import argparse
import datetime
import json
import random
import time
import pickle
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
import torch.distributed as dist
import re

from Custom_Dataset import *
from custom_utils import *
from custom_prints import *
from custom_buffer_manager import *
from custom_training import rehearsal_training

from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import get_models
from glob import glob


def init(args):
        utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(utils.get_sha()))
        if args.frozen_weights is not None:
            assert args.masks, "Frozen training is meant for segmentation only"
        print(args)
        
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        


class TrainingPipeline:
    def __init__(self, args):
        init(args)
        self.set_directory(args)
        self.args = args
        self.device = torch.device(args.device)
        self.Divided_Classes, self.dataset_name, self.start_epoch, self.start_task, self.tasks = self._incremental_setting()
        self.model, self.model_without_ddp, self.criterion, self.postprocessors, self.teacher_model = self._build_and_setup_model(task_idx=args.start_task)
        if self.args.Branch_Incremental and not args.eval:
            self.make_branch(self.start_task, self.args, replay=True)
        self.optimizer, self.lr_scheduler = self._setup_optimizer_and_scheduler()
        # self._load_state()
        self.output_dir = Path(args.output_dir)
        self.load_replay, self.rehearsal_classes = self._load_replay_buffer()
        self.DIR = os.path.join(self.output_dir, 'mAP_TEST.txt')
        self.Task_Epochs = args.Task_Epochs
    
    def set_directory(self, args):
        if args.pretrained_model_dir is not None:
            if 'checkpoints' not in args.pretrained_model_dir:
                args.pretrained_model_dir = os.path.join(args.pretrained_model_dir, 'checkpoints')
        if args.Rehearsal_file is not None:
            if 'replay' not in args.Rehearsal_file:
                args.Rehearsal_file = os.path.join(args.Rehearsal_file, 'replay')
    
    def set_task_epoch(self, args, idx):
        epochs = self.Task_Epochs
        if len(epochs) > 1:
            args.Task_Epochs = epochs[idx]
        else:
            args.Task_Epochs = epochs[0]
    

    def make_branch(self, task_idx, args, replay=False):
        if not replay:
            self.update_class(self, task_idx)          
            self.model, self.criterion, self.postprocessors = get_models(self.args.model_name, self.args, self.num_classes, self.current_class)
        
        if replay:
            weight_path = args.pretrained_model[0]
        else:
            base_path = '/'.join(args.Rehearsal_file.split('/')[:-1]) if replay else args.output_dir
            weight_path = os.path.join(base_path, f'checkpoints/cp_{self.tasks:02}_{task_idx:02}.pth')
        previous_weight = torch.load(weight_path)
        print(colored(f"Branch_incremental weight path : {weight_path}", "red", "on_yellow"))

        if args.model_name == 'deform_detr':
            for idx, class_emb in enumerate(self.model.class_embed):
                init_layer_weight = torch.nn.init.xavier_normal_(class_emb.weight.data)
                previous_layer_weight = previous_weight['model'][f'class_embed.{idx}.weight']
                previous_class_len = previous_layer_weight.size(0)

                init_layer_weight[:previous_class_len] = previous_layer_weight
                
        elif args.model_name == 'dn_detr':
            class_emb = self.model.class_embed
            label_enc = self.model.label_enc
            
            init_class_weight = torch.nn.init.xavier_normal_(class_emb.weight.data)
            init_label_weight = torch.nn.init.xavier_normal_(label_enc.weight.data)
            previous_class_weight = previous_weight['model']['class_embed.weight']
            previous_label_weight = previous_weight['model']['label_enc.weight']
            previous_class_len = previous_class_weight.size(0)
            previous_label_len = previous_label_weight.size(0)
            init_class_weight[:previous_class_len] = previous_class_weight
            init_label_weight[:previous_label_len] = previous_label_weight

    def update_class(self, task_idx):
        if self.args.Branch_Incremental is False:
            # Because original classes(whole classes) is 60 to LG, COCO is 91.
            num_classes = 60 if self.args.LG else 91
            current_class = None
        else:
            if self.args.eval:
                task_idx = len(self.Divided_Classes)
            current_class = sum(self.Divided_Classes[:task_idx+1], [])
            num_classes = len(current_class) + 1
        self.current_class = current_class
        self.num_classes = num_classes        

    def _build_and_setup_model(self, task_idx, replay=False):
        self.update_class(task_idx)

        model, criterion, postprocessors = get_models(self.args.model_name, self.args, self.num_classes, self.current_class)
        pre_model = copy.deepcopy(model)
        model.to(self.device)
        if self.args.pretrained_model is not None and not self.args.eval and not replay:
            model = load_model_params("main", model, self.args.pretrained_model)
        model_without_ddp = model
        
        teacher_model = None
        if self.args.Distill:    
            teacher_model = load_model_params("teacher", pre_model, self.args.teacher_model)
            print(f"teacher model load complete !!!!")
            return model, model_without_ddp, criterion, postprocessors, teacher_model
            
        return model, model_without_ddp, criterion, postprocessors, None
    

    def _setup_optimizer_and_scheduler(self):
        args = self.args
        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out

        param_dicts = [
            {
                "params":
                    [p for n, p in self.model_without_ddp.named_parameters()
                        if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr,
            },
            {
                "params": [p for n, p in self.model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
                "lr": args.lr * args.lr_linear_proj_mult,
            },
            {
                "params": [p for n, p in self.model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
                "lr": args.lr_backbone,
            },
        ]
            
        if args.sgd:
            optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                        weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                        weight_decay=args.weight_decay)
        lr_scheduler = ContinualStepLR(optimizer, args.lr_drop, gamma = 0.5)

        return optimizer, lr_scheduler


    def _load_state(self):
        args = self.args
        # For extra epoch training, because It's not affected to DDP.
        if args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu])
            self.model_without_ddp = self.model.module

        if args.frozen_weights is not None:
            checkpoint = torch.load(args.frozen_weights, map_location='cpu')
            self.model_without_ddp.detr.load_state_dict(checkpoint['model'])


    def _incremental_setting(self):
        args = self.args
        Divided_Classes = []
        start_epoch = 0
        start_task = 0
        tasks = args.Task
        Divided_Classes = DivideTask_for_incre(args.Task, args.Total_Classes, args.Total_Classes_Names, args.eval, args.test_file_list)
        if args.Total_Classes_Names == True :
            tasks = len(Divided_Classes)    
        
        if args.start_epoch != 0:
            start_epoch = args.start_epoch
        
        if args.start_task != 0:
            start_task = args.start_task
            
        dataset_name = "Original"
        if args.AugReplay == True:
            dataset_name = "AugReplay"

        return Divided_Classes, dataset_name, start_epoch, start_task, tasks
    

    def _load_replay_buffer(self):
        '''
            you should check more then two task splits
            criteria : task >= 2
        '''
        load_replay = []
        rehearsal_classes = {}
        args = self.args
        for idx in range(self.start_task):
            load_replay.extend(self.Divided_Classes[idx])
            
        #* Load for Replay
        if (args.Rehearsal and (self.start_task >= 1)) or args.Construct_Replay:
            rehearsal_classes = load_rehearsal(args.Rehearsal_file, 0, args.limit_image)
        
            try:
                if len(rehearsal_classes) == 0:
                    print(f"No rehearsal file. Initialization rehearsal dict")
                    rehearsal_classes = {}
            except:
                print(f"Rehearsal File Error. Generate new empty rehearsal dict.")
                rehearsal_classes = {}

        
        print(f"old class list : {load_replay}")
        return load_replay, rehearsal_classes


    def evaluation_only_mode(self):
        print(colored(f"evaluation only mode start !!", "red"))
        args = self.args
        dir_list = []
        filename_list = [test_file.split('+') if '+' in test_file else test_file for test_file in args.test_file_list]
        try:
            filename_list = sum(filename_list, [])
        except:
            pass
        
        if args.all_data == True:
            # eval과 train의 coco_path를 다르게 설정
            dir_list = glob(os.path.join(args.coco_path, '*'))
            if os.path.isfile(self.DIR):
                os.remove(self.DIR) # self.DIR = args.output_dir + 'mAP_TEST.txt'
        else:
            dir_list = [args.coco_path] # "/home/user/Desktop/vscode"+ 
        
        # FIXME: change directory list
        # filename_list = ["didtest", "pztest", "VE2021", "VEmultisingle", "VE10test"] # for DID, PZ, VE, VE, VE
        def load_all_files(directory):
            all_files = []
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)
            return all_files
        
        def extract_last_number(filepath):
            filename = filepath.split('/')[-1]  # get the last part after '/'
            numbers = re.findall(r'\d+', filename)  # find all number sequences
            if numbers:
                return int(numbers[-1])  # return the last number
            else:
                return 0  # return 0 if there are no numbers

        # load all files in data
        if self.args.pretrained_model_dir is not None:
            self.args.pretrained_model = load_all_files(self.args.pretrained_model_dir)
            print(f"test directory list : {len(self.args.pretrained_model)}")
            args.pretrained_model.sort(key=extract_last_number)
            print(f"test directory examples : {self.args.pretrained_model}")
            
        for enum, predefined_model in enumerate(self.args.pretrained_model):
            print(colored(f"current predefined_model : {enum}, defined model name : {predefined_model}", "red"))
            
            if predefined_model is not None:
                self.model = load_model_params("eval", self.model, predefined_model)
                self.make_branch(self.start_task, self.args, replay=True)
                
            print(colored(f"check filename list : {filename_list}", "red"))
            with open(self.DIR, 'a') as f:
                f.write(f"\n-----------------------pth file----------------------\n")
                f.write(f"file_name : {str(predefined_model)}\n")
                
            for task_idx, cur_file_name in enumerate(filename_list):
                
                # TODO: VE - eval인 경우도 고려하기
                file_link = [name for name in dir_list if cur_file_name in os.path.basename(name)]
                args.coco_path = file_link[0]
                print(colored(f"now evaluating file name : {args.coco_path}", "red"))
                print(colored(f"now eval classes: {self.Divided_Classes[task_idx]}", "red"))
                dataset_val, data_loader_val, _, _  = Incre_Dataset(task_idx, args, self.Divided_Classes)
                base_ds = get_coco_api_from_dataset(dataset_val)
                
                with open(self.DIR, 'a') as f:
                    f.write(f"-----------------------task working----------------------\n")
                    f.write(f"NOW TASK num : {task_idx}, checked classes : {self.Divided_Classes[task_idx]} \t ")
                    
                _, _ = evaluate(self.model, self.criterion, self.postprocessors,
                                                data_loader_val, base_ds, self.device, args.output_dir, self.DIR, args)


    def incremental_train_epoch(self, task_idx, last_task, dataset_train, data_loader_train, sampler_train, list_CC):
        args = self.args
        if isinstance(dataset_train, list):
            temp_dataset, temp_loader, temp_sampler = copy.deepcopy(dataset_train), copy.deepcopy(data_loader_train), copy.deepcopy(sampler_train)
        T_epochs = args.Task_Epochs[0] if isinstance(args.Task_Epochs, list) else args.Task_Epochs
        for epoch in range(self.start_epoch, T_epochs): #어차피 Task마다 훈련을 진행해야 하고, 중간점음 없을 것이므로 TASK마다 훈련이 되도록 만들어도 상관이 없음
            if args.MixReplay and args.Rehearsal and task_idx >= 1:
                dataset_index = epoch % 2 
                self.dataset_name = ["AugReplay", "Original"]
                dataset_train = temp_dataset[dataset_index]
                data_loader_train = temp_loader[dataset_index]
                sampler_train = temp_sampler[dataset_index]
                self.dataset_name = self.dataset_name[dataset_index]
                
            if args.distributed:
                sampler_train.set_epoch(epoch)#TODO: 추후에 epoch를 기준으로 batch sampler를 추출하는 행위 자체가 오류를 일으킬 가능성이 있음 Incremental Learning에서                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
            print(f"task id : {task_idx} / {self.tasks-1}")
            print(f"each epoch id : {epoch} , Dataset length : {len(dataset_train)}, current classes :{list_CC}")
            print(f"Task is Last : {last_task}")
            
            # Training process
            train_one_epoch(args, last_task, epoch, self.model, self.teacher_model, self.criterion, 
                            data_loader_train, self.optimizer, self.lr_scheduler,
                            self.device, self.dataset_name, list_CC, self.rehearsal_classes)
            
            # set a lr scheduler.
            self.lr_scheduler.step()

            # Save model each epoch
            save_model_params(self.model_without_ddp, self.optimizer, self.lr_scheduler, args, args.output_dir, 
                              task_idx, int(self.tasks), epoch)
        
        # For generating buffer with extra epoch
        if last_task == False and args.Rehearsal:
            print(f"model update for generating buffer list")
            self.rehearsal_classes = construct_replay_extra_epoch(args=self.args, Divided_Classes=self.Divided_Classes, model=self.model,
                                                                criterion=self.criterion, device=self.device, rehearsal_classes=self.rehearsal_classes,
                                                                data_loader_train=data_loader_train, list_CC=list_CC)
            print(f"complete save and merge replay's buffer process")
            print(f"next replay buffer list : {self.rehearsal_classes.keys()}")
            
        # For task information
        save_model_params(self.model_without_ddp, self.optimizer, self.lr_scheduler, args, args.output_dir, 
                          task_idx, int(self.tasks), -1)
        self.load_replay.extend(self.Divided_Classes[task_idx])
        self.teacher_model = self.model_without_ddp #Trained Model Change in change TASK 
        self.teacher_model = teacher_model_freeze(self.teacher_model)


    # when only construct replay buffer    
    def construct_replay_buffer(self):
        construct_replay_extra_epoch(args=self.args, Divided_Classes=self.Divided_Classes, model=self.model,
                                    criterion=self.criterion, device=self.device, rehearsal_classes=self.rehearsal_classes)


    # No incremental learning process    
    def only_one_task_training(self):
        dataset_train, data_loader_train, sampler_train, list_CC = Incre_Dataset(0, self.args, self.Divided_Classes)
        
        print(f"Normal Training Process \n \
                Classes : {self.Divided_Classes}")
        
        # Normal training with each epoch
        self.incremental_train_epoch(task_idx=0, last_task=True, dataset_train=dataset_train,
                                         data_loader_train=data_loader_train, sampler_train=sampler_train,
                                         list_CC=list_CC)

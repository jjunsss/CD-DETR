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
from Custom_Dataset import *
from custom_utils import *
from custom_prints import *
from custom_buffer_manager import *
from custom_training import rehearsal_training

from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model

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
        self.args = args
        self.device = torch.device(args.device)
        self.Divided_Classes, self.dataset_name, self.start_epoch, self.start_task, self.tasks = self._incremental_setting()
        self.model, self.model_without_ddp, self.criterion, self.postprocessors, self.teacher_model = self._build_and_setup_model(len(self.Divided_Classes[0]))
        self.optimizer, self.lr_scheduler = self._setup_optimizer_and_scheduler()
        self._load_state()
        self.output_dir = Path(args.output_dir)
        self.load_replay, self.rehearsal_classes = self._load_replay_buffer()
        self.DIR = './mAP_TEST.txt'


    def make_branch(self, task_idx, class_len, args):
        self.model, self.model_without_ddp, self.criterion, \
            self.postprocessors, self.teacher_model = self._build_and_setup_model(class_len)
        
        weight_path = os.path.join(args.output_dir, f'cp_{self.tasks:02}_{task_idx:02}.pth')
        previous_weight = torch.load(weight_path)

        for idx, class_emb in enumerate(self.model.class_embed):
            init_layer_weight = torch.nn.init.xavier_normal_(class_emb.weight.data)
            previous_layer_weight = previous_weight['model'][f'class_embed.{idx}.weight']
            previous_class_len = previous_layer_weight.size(0)

            init_layer_weight[:previous_class_len] = previous_layer_weight

    def _build_and_setup_model(self, num_classes):
        if self.args.Branch_Incremental is False:
            # Because original classes(whole classes) is 60 to LG, COCO is 91.
            num_classes = 60 if self.args.LG else 91 
            
        model, criterion, postprocessors = build_model(self.args, num_classes)
        pre_model = copy.deepcopy(model)
        model.to(self.device)
        if self.args.pretrained_model is not None:
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
        Divided_Classes = DivideTask_for_incre(args.Task, args.Total_Classes, args.Total_Classes_Names)
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
        if args.Rehearsal and (self.start_task >= 1):
            rehearsal_classes = load_rehearsal(args.Rehearsal_file, 0, args.Memory)
        
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
        args = self.args
        expand_classes = []

        for task_idx in range(int(self.tasks)):
            expand_classes.extend(self.Divided_Classes[task_idx])
            print(f"trained task classes: {self.Divided_Classes[task_idx]}\t  we check all classes {expand_classes}")
            dataset_val, data_loader_val, sampler_val, current_classes  = Incre_Dataset(task_idx, args, expand_classes, False)
            base_ds = get_coco_api_from_dataset(dataset_val)
            with open(self.DIR, 'a') as f:
                f.write(f"NOW TASK num : {task_idx}, checked classes : {expand_classes} \t file_name : {str(os.path.basename(args.pretrained_model))} \n")
                
            _, _ = evaluate(self.model, self.criterion, self.postprocessors,
                                            data_loader_val, base_ds, self.device, args.output_dir, self.DIR)


    def incremental_train_epoch(self, task_idx, last_task, dataset_train, data_loader_train, sampler_train, list_CC):
        args = self.args
        if isinstance(dataset_train, list):
            temp_dataset, temp_loader, temp_sampler = copy.deepcopy(dataset_train), copy.deepcopy(data_loader_train), copy.deepcopy(sampler_train)
            
        for epoch in range(self.start_epoch, args.Task_Epochs): #어차피 Task마다 훈련을 진행해야 하고, 중간점음 없을 것이므로 TASK마다 훈련이 되도록 만들어도 상관이 없음
            if args.MixReplay and args.Rehearsal and task_idx >= 1:
                dataset_index = epoch % 2 
                self.dataset_name = ["AugReplay", "Original"]
                dataset_train = temp_dataset[dataset_index]
                data_loader_train = temp_loader[dataset_index]
                sampler_train = temp_sampler[dataset_index]
                self.dataset_name = self.dataset_name[dataset_index]
                
            if args.distributed:
                sampler_train.set_epoch(epoch)#TODO: 추후에 epoch를 기준으로 batch sampler를 추출하는 행위 자체가 오류를 일으킬 가능성이 있음 Incremental Learning에서                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
            print(f"task id : {task_idx}")
            print(f"each epoch id : {epoch} , Dataset length : {len(dataset_train)}, current classes :{list_CC}")
            print(f"Task is Last : {last_task}")
            print(f"args task : : {self.tasks}")
            
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
            self.rehearsal_classes = contruct_replay_extra_epoch(args=self.args, Divided_Classes=self.Divided_Classes, model=self.model,
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
        
    def construct_replay_buffer(self):
        contruct_replay_extra_epoch(args=self.args, Divided_Classes=self.Divided_Classes, model=self.model,
                                    criterion=self.criterion, device=self.device, rehearsal_classes=self.rehearsal_classes)
        
    def only_one_task_training(self):
        dataset_train, data_loader_train, sampler_train, list_CC = Incre_Dataset(0, self.args, self.Divided_Classes)
        
        print(f"Normal Training Process \n \
                Classes : {self.Divided_Classes}")
        
        # Normal training with each epoch
        self.incremental_train_epoch(task_idx=0, last_task=True, dataset_train=dataset_train,
                                         data_loader_train=data_loader_train, sampler_train=sampler_train,
                                         list_CC=list_CC)

import random
from pathlib import Path
import os
import numpy as np
import torch
import util.misc as utils

import re

from Custom_Dataset import *
from custom_utils import *
from custom_prints import *
from custom_buffer_manager import *

from datasets import get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import get_models
from glob import glob
import torch.backends.cudnn as cudnn


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
        cudnn.benchmark = False
        cudnn.deterministic = True
        


class TrainingPipeline:
    def __init__(self, args):
        init(args)
        self.set_directory(args)
        self.args = args
        self.device = torch.device(args.device)
        self.Divided_Classes, self.dataset_name, self.start_epoch, self.start_task, self.tasks = self._incremental_setting()
        if self.args.eval:
            self.args.start_task = 0
        self.model, self.model_without_ddp, self.criterion, self.postprocessors, self.teacher_model = self._build_and_setup_model(task_idx=self.args.start_task)
        if self.args.Branch_Incremental and not args.eval and args.pretrained_model is not None:
            self.make_branch(self.start_task, self.args, is_init=True)
        self.optimizer, self.lr_scheduler = self._setup_optimizer_and_scheduler()
        self.output_dir = Path(args.output_dir)
        self.load_replay, self.rehearsal_classes = self._load_replay_buffer()
        self.DIR = os.path.join(self.output_dir, 'mAP_TEST.txt')
        self.Task_Epochs = args.Task_Epochs
    
    def set_directory(self, args):
        '''
            pretrained_model and rehearsal file should be contrained "checkpoints" and "replay", respectively.
        '''        
        if args.pretrained_model_dir is not None:
            if 'checkpoints' not in args.pretrained_model_dir and not args.eval :
                args.pretrained_model_dir = os.path.join(args.pretrained_model_dir, 'checkpoints')
                print(colored(f"args.pretrained_model_dir : {args.pretrained_model_dir}", "red", "on_yellow"))
        if args.Rehearsal_file is not None:
            if 'replay' not in args.Rehearsal_file and not args.eval :
                args.Rehearsal_file = os.path.join(args.Rehearsal_file, 'replay')
                print(colored(f"args.Rehearsal_file : {args.Rehearsal_file}", "red", "on_yellow"))
    
    def set_task_epoch(self, args, idx):
        epochs = self.Task_Epochs
        if len(epochs) > 1:
            args.Task_Epochs = epochs[idx]
        else:
            args.Task_Epochs = epochs[0]
    

    def make_branch(self, task_idx, args, is_init=False):
        self.update_class(task_idx)
        
        ## 고려하고 있는 case ##
        # case 1) start_task=0부터 시작해서 차근차근 task를 진행하는 경우
        #    이 경우는 task가 끝날 때마다 해당 task의 weight가 output_dir에 저장되므로,
        #    previous_weight를 output_dir에서 불러옴
        #
        # case 2) start_task=1, args.Rehearsal_file에서 task 0의 데이터 불러오는 경우 (전체 task 2개)
        #    이 경우는 previous_weight가 현재 output_dir에 없음
        #    따라서 args.pretrained_model에서 previous_weight를 불러옴
        #
        # case 3) start_task=1, args.Rehearsal_file에서 task 0의 데이터 불러오는 경우 (전체 task 3개 이상)
        #    이 경우, 초기에는 args.pretrained_model에서 previous_weight를 불러와야 하지만,
        #    task가 변할 경우 args.output_dir에서 previous_weight를 불러와야 함
        #
        # case 1, 2, 3를 모두 충족시키는 방법)
        #    main_component에서 make_branch가 참조되는 경우 is_init을 True로, main에서 참조되는 경우 False로 설정함.
        #    case 1의 경우는 어차피 args.pretrained_model이 선언되어 있지 않기 때문에 is_init이 항상 False임
        #    case 2,3의 경우 args.pretrained_model이 존재하기 때문에, is_init이 True인 경우와 False인 경우가 둘 다 존재함
        #        1) is_init==True
        #              해당 경우는 args.pretrained_model에서 previous_weight를 불러옴
        #        2) is_init==False
        #              해당 경우는 args.output_dir에서 previous_weight를 불러옴
        
        if is_init:
            weight_path = args.pretrained_model[0]
        else:
            weight_path = os.path.join(args.output_dir, f'checkpoints/cp_{self.tasks:02}_{task_idx:02}.pth')
            self.model, self.model_without_ddp, self.criterion, self.postprocessors, self.teacher_model = \
                self._build_and_setup_model(task_idx=task_idx)
            self.model = self.model_without_ddp = load_model_params("main", self.model, weight_path)
            
        previous_weight = torch.load(weight_path)
        print(colored(f"Branch_incremental weight path : {weight_path}", "red", "on_yellow"))

        try:
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
        except:
            # LG pretrained model이 아니라 coco pretrained model을 사용할 때는 class, label weight 안가져옴
            print(colored(f"Num of class does not matched! : {weight_path}", "yellow", "on_red"))

    def update_class(self, task_idx):
        if self.args.Branch_Incremental is False:
            # Because original classes(whole classes) is 60 to LG, COCO is 91.
            num_classes = 60 if self.args.LG else 91
            current_class = None
        else:
            idx = len(self.Divided_Classes) if self.args.LG and self.args.eval else task_idx+1
            current_class = sum(self.Divided_Classes[:idx], [])
            num_classes = len(current_class) + 1
            
        previous_classes = sum(self.Divided_Classes[:task_idx], []) # For distillation options.
        self.previous_classes = previous_classes
        self.current_class = current_class
        self.num_classes = num_classes

    def _build_and_setup_model(self, task_idx):
        self.update_class(task_idx)

        model, criterion, postprocessors = get_models(self.args.model_name, self.args, self.num_classes, self.current_class)
        if self.args.fisher_model is not None:
            print(colored(f"fisher model loading : {self.args.fisher_model}", "blue", "on_yellow"))
            if self.args.Branch_Incremental is False:
                self.fisher_model, self.fisher_criterion, _ = get_models(self.args.model_name, self.args, self.num_classes, self.current_class)
            else:
                self.fisher_model, self.fisher_criterion, _ = get_models(self.args.model_name, self.args, len(self.previous_classes)+1, self.previous_classes)
                
            self.fisher_model = load_model_params("eval", self.fisher_model, self.args.fisher_model)
            self.fisher_ddp_state(self.fisher_model)
        if self.args.Distill:
            pre_model, _, _ = get_models(self.args.model_name, self.args, self.num_classes, self.current_class)
        #FIXME: If we use the pre_model option, we need to load the pre-trained model architecture.
        #FIXME: because previous version of the model does not match the current version(branch incremental option)
        if self.args.pretrained_model is not None and not self.args.eval:
            model = load_model_params("main", model, self.args.pretrained_model)
        # if self.args.AugReplay :
        #     self.fisher_model = load_model_params("eval", model, self.args.fisher_model)
        model_without_ddp = model
        
        teacher_model = None
        if self.args.Distill:
            teacher_model = load_model_params("teacher", pre_model, self.args.teacher_model)
            print(f"teacher model load complete !!!!")
            return model, model_without_ddp, criterion, postprocessors, teacher_model
            
        return model, model_without_ddp, criterion, postprocessors, teacher_model
    

    def _setup_optimizer_and_scheduler(self):
        args = self.args
        
        def match_name_keywords(n, name_keywords):
            out = False
            for b in name_keywords:
                if b in n:
                    out = True
                    break
            return out
        

        # if args.model_name == "deform_detr" :
        #     total_batch_size = args.batch_size * utils.get_world_size()
        #     lr_ratio = total_batch_size / 32
        #     args.lr = args.lr * round(lr_ratio, 2)
        #     args.lr_backbone = args.lr_backbone * round(lr_ratio, 2)
        #     print(colored(f"args LR : {args.lr}", "blue"))
        #     print(colored(f"args LR backbone : {args.lr_backbone}", "blue"))
            
        # if args.model_name == "dn_detr" :
        #     total_batch_size = args.batch_size * utils.get_world_size()
        #     lr_ratio = total_batch_size / 16
        #     args.lr = args.lr * round(lr_ratio, 2)
        #     args.lr_backbone = args.lr_backbone * round(lr_ratio, 2)
        #     print(colored(f"args LR : {args.lr}", "blue"))
        #     print(colored(f"args LR backbone : {args.lr_backbone}", "blue"))

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
        # lr_scheduler = ContinualStepLR(optimizer, args.lr_drop, gamma = 0.5)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

        return optimizer, lr_scheduler


    def load_ddp_state(self):
        args = self.args
        # For extra epoch training, because It's not affected to DDP.
        self.model = self.model.to(self.device)
        if args.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu])
            self.model_without_ddp = self.model.module

        if args.frozen_weights is not None:
            checkpoint = torch.load(args.frozen_weights, map_location='cpu')
            self.model_without_ddp.detr.load_state_dict(checkpoint['model'])
            

    def fisher_ddp_state(self, fisher_model):
        args = self.args
        if args.distributed and fisher_model != None:
            self.fisher_model = self.fisher_model.to(self.device)
            self.fisher_model = torch.nn.parallel.DistributedDataParallel(self.fisher_model, device_ids=[args.gpu])
            
            
    def _incremental_setting(self):
        args = self.args
        Divided_Classes = []
        start_epoch = 0
        start_task = 0
        tasks = args.Task
        Divided_Classes = DivideTask_for_incre(args, args.Task, args.Total_Classes, args.Total_Classes_Names, args.eval, args.test_file_list)
        if args.Total_Classes_Names == True :
            # If you use the Total Classes names, you don't need to write args.tasks(you can use the any value)
            tasks = len(Divided_Classes)    
        
        if args.start_epoch != 0:
            start_epoch = args.start_epoch
        
        if args.start_task != 0:
            start_task = args.start_task
            
        dataset_name = "Original"
        if args.AugReplay :
            dataset_name = "AugReplay"
        elif args.Mosaic :
            dataset_name = "Mosaic"

        return Divided_Classes, dataset_name, start_epoch, start_task, tasks
    

    def _load_replay_buffer(self):
        '''
            you should check more then two task splits. because It is used in incremental tasks
            1. criteria : tasks >= 2
            2. args.Rehearsal : True
            3. args.
        '''
        load_replay = []
        rehearsal_classes = {}
        args = self.args
        for idx in range(self.start_task):
            load_replay.extend(self.Divided_Classes[idx])
        
        load_task = 0 if args.start_task == 0 else args.start_task - 1
        
        #* Load for Replay
        if args.Rehearsal:
            rehearsal_classes = load_rehearsal(args.Rehearsal_file, load_task, args.limit_image)
            try:
                if len(list(rehearsal_classes.keys())) == 0:
                    print(f"No rehearsal file. Initialization rehearsal dict")
                    rehearsal_classes = {}
                else:
                    print(f"replay keys length :{len(list(rehearsal_classes.keys()))}")
            except:
                print(f"Rehearsal File Error. Generate new empty rehearsal dict.")
                rehearsal_classes = {}

        return load_replay, rehearsal_classes


    def evaluation_only_mode(self,):
        print(colored(f"evaluation only mode start !!", "red"))
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
            
        args = self.args
        dir_list = []
        if args.LG :
            filename_list = [test_file.split('+') if '+' in test_file else test_file for test_file in args.test_file_list]
            try:
                filename_list = sum(filename_list, [])
            except:
                pass
    
        if args.all_data == True:
            # eval과 train의 coco_path를 다르게 설정
            dir_list = [f for f in glob(os.path.join(args.coco_path, '*')) if os.path.isdir(f)]
            if os.path.isfile(self.DIR):
                os.remove(self.DIR) # self.DIR = args.output_dir + 'mAP_TEST.txt'
        else:
            dir_list = [args.coco_path] # "/home/user/Desktop/vscode"+ 
        
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
            
            if args.LG and 've' in filename_list:
                ve_idx = filename_list.index('ve')
                filename_list.pop(ve_idx)
                filename_list.extend(['ve10', 've2021', 'vemulti']) # 실제 파일 이름에 해당 키워드가 포함되어 있어야 함
        
                print(colored(f"check filename list : {filename_list}", "red"))
                
            with open(self.DIR, 'a') as f:
                f.write(f"\n-----------------------pth file----------------------\n")
                f.write(f"file_name : {os.path.basename(predefined_model)}\n")  # 파일 이름
                f.write(f"file_path : {os.path.abspath(os.path.dirname(predefined_model))}\n")  # 파일 절대 경로

            if args.LG:  
                for task_idx, cur_file_name in enumerate(filename_list):
                    if 've' in cur_file_name:
                        task_idx = ve_idx
                    elif args.orgcocopath:
                        cur_file_name = 'val'
                    elif 'coco' in cur_file_name and not arg.orgcocopath:
                        cur_file_name = 'test'
                        
                    # TODO: VE - eval인 경우도 고려하기
                    file_link = [name for name in dir_list if cur_file_name in os.path.basename(name).lower()]
                    args.coco_path = file_link[0]
                    print(colored(f"now evaluating file name : {args.coco_path}", "red"))
                    print(colored(f"now eval classes: {self.Divided_Classes[task_idx]}", "red"))
                    dataset_val, data_loader_val, _, _  = Incre_Dataset(task_idx, args, self.Divided_Classes)
                    base_ds = get_coco_api_from_dataset(dataset_val)
                    
                    with open(self.DIR, 'a') as f:
                        f.write(f"-----------------------task working----------------------\n")
                        f.write(f"NOW TASK num : {task_idx} , checked classes : {self.Divided_Classes[task_idx]} \t ")
                        
                    _, _ = evaluate(self.model, self.criterion, self.postprocessors,
                                                    data_loader_val, base_ds, self.device, args.output_dir, self.DIR, args)
            else:
                test_epoch = 1 if args.Total_Classes != args.Test_Classes else args.Task
                for task_idx in range(test_epoch) :
                    print(colored(f"evaluation task number {task_idx + 1} / {test_epoch}", "blue", "on_yellow"))
                    Divided_Classes = DivideTask_for_incre(args, self.tasks, args.Total_Classes, args.Total_Classes_Names, False, args.test_file_list)
                    dataset_val, data_loader_val, _, _  = Incre_Dataset(task_idx, args, Divided_Classes)
                    base_ds = get_coco_api_from_dataset(dataset_val)
                    with open(self.DIR, 'a') as f:
                        f.write(f"-----------------------task working----------------------\n")
                        f.write(f"NOW TASK num : {task_idx + 1} / {test_epoch}, checked classes : {sum(Divided_Classes[:task_idx+1], [])} \t ")
                        
                    _, _ = evaluate(self.model, self.criterion, self.postprocessors,
                                                    data_loader_val, base_ds, self.device, args.output_dir, self.DIR, args)
                    
                    if args.FPP :
                        #TODO: We should provide two pth file for calculating difference value (M1, M2 foretting)
                        _, _ = evaluate(self.model, self.criterion, self.postprocessors,
                                                    data_loader_val, base_ds, self.device, args.output_dir, self.DIR, args)


    def incremental_train_epoch(self, task_idx, last_task, dataset_train, data_loader_train, sampler_train, list_CC, first_training=False):
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
            print(colored(f"task id : {task_idx} / {self.tasks-1}", "blue", "on_white"))
            print(colored(f"each epoch id : {epoch} , Dataset length : {len(dataset_train)}, current classes :{list_CC}", "blue", "on_white"))
            print(colored(f"Task is Last : {last_task}", "blue", "on_white"))
            
            # Training process
            train_one_epoch(args, task_idx, last_task, epoch, self.model, self.teacher_model, self.criterion, dataset_train,
                            data_loader_train, self.optimizer, self.lr_scheduler,
                            self.device, self.dataset_name, list_CC, self.rehearsal_classes, first_training)
            
            # set a lr scheduler.
            self.lr_scheduler.step()

            # Save model each epoch
            save_model_params(self.model_without_ddp, self.optimizer, self.lr_scheduler, args, args.output_dir, 
                            task_idx, int(self.tasks), epoch)
        
        # If task change, training epoch should be zero.
        self.start_epoch = 0
        
        # For generating buffer with extra epoch
        if last_task == False and args.Rehearsal:
            print(f"model update for generating buffer list")
            self.rehearsal_classes = construct_replay_extra_epoch(args=self.args, Divided_Classes=self.Divided_Classes, model=self.model,
                                                                criterion=self.criterion, device=self.device, rehearsal_classes=self.rehearsal_classes,
                                                                task_num=task_idx)
            print(f"complete save and merge replay's buffer process")
            print(f"next replay buffer list : {self.rehearsal_classes.keys()}")
            
        # For task information
        save_model_params(self.model_without_ddp, self.optimizer, self.lr_scheduler, args, args.output_dir, 
                        task_idx, int(self.tasks), -1)
        self.load_replay.extend(self.Divided_Classes[task_idx])
        self.teacher_model = self.model_without_ddp # teacher model change to new trained model architecture before next task
        self.teacher_model = teacher_model_freeze(self.teacher_model)

        if utils.get_world_size() > 1:
            dist.barrier()


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
        
from copy import deepcopy
def generate_dataset(first_training, task_idx, args, pipeline):
    # Generate new dataset(current classes)
    dataset_train, data_loader_train, sampler_train, list_CC = Incre_Dataset(task_idx, args, pipeline.Divided_Classes)

    if not first_training and args.Rehearsal:
        
        # Ready for replay training strategy
        temp_replay_dataset = deepcopy(pipeline.rehearsal_classes)
        replay_dataset = dict(sorted(temp_replay_dataset.items(), key=lambda x: x[0]))
        previous_classes = sum(pipeline.Divided_Classes[:task_idx], []) # Not now current classe
        if args.AugReplay:
            #TODO: need to Fisher condition
            if args.fisher_model != None:
                if args.CER == "fisher" :
                    fisher_dict = calc_fisher_process(args, pipeline.rehearsal_classes, previous_classes, 
                                                    pipeline.fisher_criterion, pipeline.fisher_model, pipeline.optimizer)
                elif args.CER == "uniform":
                    fisher_dict = None
            else :
                if args.CER == "fisher" :    
                    fisher_dict = calc_fisher_process(args, pipeline.rehearsal_classes, previous_classes, 
                                                    pipeline.criterion, pipeline.model, pipeline.optimizer)
                elif args.CER == "uniform":
                    fisher_dict = None
                    
            AugRplay_dataset, AugRplay_loader, AugRplay_sampler = CombineDataset(
                args, replay_dataset, dataset_train, args.num_workers, args.batch_size, 
                old_classes=previous_classes, fisher_dict=fisher_dict, MixReplay="AugReplay")
        else:
            fisher_dict = None
            AugRplay_dataset, AugRplay_loader, AugRplay_sampler = None, None, None
            
        assert (args.Mosaic and not args.AugReplay) or (not args.Mosaic and args.AugReplay) or (not args.Mosaic and not args.AugReplay)
            
        if args.Mosaic and not args.AugReplay:
            mosaic_dataset, mosaic_loader, mosaic_sampler = CombineDataset(
                args, replay_dataset, dataset_train, args.num_workers, args.batch_size, 
                old_classes=previous_classes, fisher_dict=None)
            return mosaic_dataset, mosaic_loader, mosaic_sampler, list_CC

        # Combine dataset for original and AugReplay(Circular)
        original_dataset, original_loader, original_sampler = CombineDataset(
            args, replay_dataset, dataset_train, args.num_workers, args.batch_size, 
            old_classes=previous_classes, fisher_dict=fisher_dict, MixReplay="Original")


        # Set a certain configuration
        dataset_train, data_loader_train, sampler_train = dataset_configuration(
            args, original_dataset, original_loader, original_sampler, AugRplay_dataset, AugRplay_loader, AugRplay_sampler)

        # Task change for learning rate scheduler
        # this lr changed value
        # pipeline.lr_scheduler.task_change()

    return dataset_train, data_loader_train, sampler_train, list_CC
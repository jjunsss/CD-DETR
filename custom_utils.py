from ast import arg
from pyexpat import model
from sched import scheduler
from xmlrpc.client import Boolean
import torch
import util.misc as utils
import numpy as np
from typing import Tuple, Dict, List, Optional
import os
from custom_prints import over_label_checker, check_components
from termcolor import colored

def decompose(func):
    def wrapper(no_use_count: int, samples: utils.NestedTensor, targets: Dict, 
                origin_samples: List, origin_targets: Dict ,used_number: List) \
            -> Tuple[torch.Tensor, list, bool]:
        """
            origin_samples : List
        """
        output = func(no_use_count, samples, targets, origin_samples, origin_targets, used_number) #Batch 3개라고 가정하고 문제풀이하기
        batch_size = output[0]
        no_use = output[1]
        samples = output[2]
        targets = output[3]
        origin_samples = output[4]
        origin_targets = output[5]
        yes_use = output[6]
        if batch_size == 1:
            if no_use == 1:
                return samples, targets, origin_samples, origin_targets, False
            return samples, targets, origin_samples, origin_targets, True
        
        if batch_size == 2: #! used to divide rehearsal or main Trainer
            if no_use == 2:
                return samples, targets, origin_samples, origin_targets, False
            if no_use == 1:
                new_targets = []
                origin_new_samples = []
                origin_new_targets = []
                useit0 = yes_use[0]
                ten, mask = samples.decompose()
                if ten.shape[0] > 1 :
                    ten0 = torch.unsqueeze(ten[useit0], 0)
                    mask0 = torch.unsqueeze(mask[useit0], 0)
                    samples = utils.NestedTensor(ten0, mask0)
                    new_targets.append(targets[useit0])
                    targets = [{k: v for k, v in t.items()} for t in new_targets]
                origin_new_samples.append(origin_samples[useit0])
                origin_new_targets.append(origin_targets[useit0])
                origin_targets = [{k: v for k, v in t.items()} for t in origin_new_targets]
                return samples, targets, origin_new_samples, origin_targets, True
            
            #origin_samples, _ = origin_samples.decompose()
            return samples, targets, origin_samples, origin_targets, True
            
        if batch_size == 3 :
            if no_use == 3:
                return samples, targets, origin_samples, origin_targets, False
            if no_use == 2:
                new_targets = []
                origin_new_targets = []
                origin_new_samples = []
                useit0 = yes_use[0]
                ten, mask = samples.decompose()
                
                ten0 = torch.unsqueeze(ten[useit0], 0)
                mask0 = torch.unsqueeze(mask[useit0], 0)

                
                samples = utils.NestedTensor(ten0, mask0)
                
                new_targets.append(targets[useit0])
                origin_new_samples.append(origin_samples[useit0])
                origin_new_targets.append(origin_targets[useit0])
                
                targets = [{k: v for k, v in t.items()} for t in new_targets]
                origin_targets = [{k: v for k, v in t.items()} for t in origin_new_targets]
                return samples, targets, origin_new_samples, origin_targets, True
            
            if no_use == 1:
                new_targets = []
                origin_new_targets = []
                origin_new_samples = []
                useit0 = yes_use[0]
                useit1 = yes_use[1]
                ten, mask = samples.decompose()
                
                ten0 = torch.unsqueeze(ten[useit0], 0)
                mask0 = torch.unsqueeze(mask[useit0], 0)
                ten1 = torch.unsqueeze(ten[useit1], 0)
                mask1 = torch.unsqueeze(mask[useit1], 0)
                
                ten = torch.cat([ten0,ten1], dim = 0) 
                mask = torch.cat([mask0,mask1], dim = 0) 
                
                samples = utils.NestedTensor(ten, mask)
                
                new_targets.append(targets[useit0])
                new_targets.append(targets[useit1])
                targets = [{k: v for k, v in t.items()} for t in new_targets]
                
                origin_new_samples.append(origin_samples[useit0])
                origin_new_samples.append(origin_samples[useit1])
                origin_new_targets.append(origin_targets[useit0])
                origin_new_targets.append(origin_targets[useit1])
                origin_targets = [{k: v for k, v in t.items()} for t in origin_new_targets]
                
                return samples, targets, origin_new_samples, origin_targets, True 
            
            return samples, targets, origin_samples, origin_targets, True #return original
        
        if batch_size == 4 :
            if no_use == 4:
                return samples, targets, origin_samples, origin_targets, False
            if no_use == 3:
                new_targets = []
                origin_new_targets = []
                origin_new_samples = []
                useit0 = yes_use[0]
                ten, mask = samples.decompose()
                
                ten0 = torch.unsqueeze(ten[useit0], 0)
                mask0 = torch.unsqueeze(mask[useit0], 0)
                
                samples = utils.NestedTensor(ten0, mask0)
                
                new_targets.append(targets[useit0])
                origin_new_targets.append(origin_targets[useit0])
                origin_new_samples.append(origin_samples[useit0])
                
                
                targets = [{k: v for k, v in t.items()} for t in new_targets]
                origin_targets = [{k: v for k, v in t.items()} for t in origin_new_targets]
                return samples, targets, origin_new_samples, origin_targets, True
            if no_use == 2:
                new_targets = []
                origin_new_targets = []
                origin_new_samples = []
                useit0 = yes_use[0]
                useit1 = yes_use[1]
                ten, mask = samples.decompose()
                
                ten0 = torch.unsqueeze(ten[useit0], 0)
                mask0 = torch.unsqueeze(mask[useit0], 0)
                ten1 = torch.unsqueeze(ten[useit1], 0)
                mask1 = torch.unsqueeze(mask[useit1], 0)
                
                ten = torch.cat([ten0,ten1], dim = 0) 
                mask = torch.cat([mask0,mask1], dim = 0) 
                
                samples = utils.NestedTensor(ten, mask)
                
                
                new_targets.append(targets[useit0])
                new_targets.append(targets[useit1])
                targets = [{k: v for k, v in t.items()} for t in new_targets]
                
                origin_new_targets.append(origin_targets[useit0])
                origin_new_targets.append(origin_targets[useit1])
                origin_targets = [{k: v for k, v in t.items()} for t in origin_new_targets]
                
                origin_new_samples.append(origin_samples[useit0])
                origin_new_samples.append(origin_samples[useit1])
                
                return samples, targets, origin_new_samples, origin_targets, True 
            
            if no_use == 1:
                new_targets = []
                origin_new_targets = []
                origin_new_samples = []
                useit0 = yes_use[0]
                useit1 = yes_use[1]
                useit2 = yes_use[2]
                ten, mask = samples.decompose()
                
                ten0 = torch.unsqueeze(ten[useit0], 0)
                mask0 = torch.unsqueeze(mask[useit0], 0)
                ten1 = torch.unsqueeze(ten[useit1], 0)
                mask1 = torch.unsqueeze(mask[useit1], 0)
                ten2 = torch.unsqueeze(ten[useit2], 0)
                mask2 = torch.unsqueeze(mask[useit2], 0)
                
                ten = torch.cat([ten0,ten1, ten2], dim = 0) 
                mask = torch.cat([mask0,mask1, mask2], dim = 0) 
                
                samples = utils.NestedTensor(ten, mask)
                new_targets.append(targets[useit0])
                new_targets.append(targets[useit1])
                new_targets.append(targets[useit2])
                targets = [{k: v for k, v in t.items()} for t in new_targets]
                
                origin_new_samples.append(origin_samples[useit0])
                origin_new_samples.append(origin_samples[useit1])
                origin_new_samples.append(origin_samples[useit2])
                
                origin_new_targets.append(origin_targets[useit0])
                origin_new_targets.append(origin_targets[useit1])
                origin_new_targets.append(origin_targets[useit2])
                origin_targets = [{k: v for k, v in t.items()} for t in origin_new_targets]
                
                return samples, targets, origin_new_samples, origin_targets, True 
            
            return samples, targets, origin_samples, origin_targets, True #return original
        
        raise Exception("set your batch_size : 1 or 2 or 3 or 4")
    return wrapper

# how many classes in targets for couning each isntances 
def check_class(verbose, LG_Dataset: bool, targets: Dict, label_dict: Dict, current_classes,
                DID_COUNT: int =4000, VE_COUNT: int=7000, PZ_COUNT: int=4000, CL_Limited: int = 0):
    no_use = []
    yes_use = []
    check_list = []
    
    if LG_Dataset :
        limit2 = [28, 32, 35, 41, 56] #photozone
        limit3 = [22, 23, 24, 25, 26, 27, 29, 30,  31, 33, 34, 36, 37,38, 39, 40,42,43,44, 45, 46,47, 48, 49,50, 51, 52,53,\
                    54, 55, 57, 58, 59] #VE 
            
        for enum, target in enumerate(targets):
            
            label_tensor = target['labels']
            label_tensor = label_tensor.cpu()
            label_tensor_unique = torch.unique(label_tensor)
            
            # Normal Limited Training (for equivalent Performance)
            if CL_Limited is 0 :
                check_list = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() <= 21 and label_dict[idx.item()] > DID_COUNT] #did
                check_list2 = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() in limit2 and label_dict[idx.item()] > PZ_COUNT] #pz
                check_list3 = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() in limit3 and label_dict[idx.item()] > VE_COUNT] #ve (more)
                check_list4 = []
            else:
                # Continual Limited Training (for equivalent Performance)
                check_list = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() <= 21 and label_dict[idx.item()] > CL_Limited] #did
                check_list2 = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() in limit2 and label_dict[idx.item()] > CL_Limited] #pz
                check_list3 = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() in limit3 and label_dict[idx.item()] > CL_Limited] #ve (more)
                check_list4 = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if label_dict[idx.item()] > CL_Limited] #ve (other)
                
            #TODO : before checking overlist Process outputdim and continue iter
            
            if len(check_list) > 0 or len(check_list2) > 0 or len(check_list3) > 0 or len(check_list4) > 0:
                if verbose == True:
                    over_label_checker(check_list, check_list2, check_list3, check_list4)           
                no_use.append(enum)
            else:
                yes_use.append(enum)
                label_tensor_count = label_tensor.numpy()
                bin = np.bincount(label_tensor_count)
                for idx in label_tensor_unique:
                    idx = idx.item()
                    if idx in label_dict.keys():
                        label_dict[idx] += bin[idx]
                    else :
                        label_dict[idx] = bin[idx]
    else:
        if CL_Limited == 0 :
            CL_Limited = 99999999 #No use Limited Training
            
        for enum, target in enumerate(targets):
            label_tensor = target['labels']
            label_tensor = label_tensor.cpu()
            label_tensor_unique = torch.unique(label_tensor)
            
            check_list = [idx.item() for idx in label_tensor_unique if idx.item() in label_dict  if idx.item() in current_classes and label_dict[idx.item()] > CL_Limited] #* 데이터가 몇 개로 나눠져도 상관 없도록 구성해야함
            if len(check_list) > 0 :
                over_label_checker(check_list)           
                no_use.append(enum)
            else:
                yes_use.append(enum)
                label_tensor_count = label_tensor.numpy()
                bin = np.bincount(label_tensor_count)
                for idx in label_tensor_unique:
                    idx = idx.item()
                    if idx in label_dict.keys():
                        label_dict[idx] += bin[idx]
                    else :
                        label_dict[idx] = bin[idx]
    return no_use, yes_use, label_dict

from torch.utils.data import DataLoader
from datasets import build_dataset, get_coco_api_from_dataset
def new_dataLoader(saved_dict, args):
    dataset_idx_list = []
    for _, value in saved_dict.items():
        if len(value) > 0 :
            np_idx_list = np.array(value, dtype=object)
            dataset_idx_list.extend(np.unique(np_idx_list[:, 3]).astype(np.uint8).tolist())
    
    custom_dataset = build_dataset(image_set='train', args=args, img_ids=dataset_idx_list)
    
    custom_loader = DataLoader(custom_dataset, args.batch_size, shuffle=True, collate_fn=utils.collate_fn, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    return custom_loader


def _dataset_for_memory_check(*args):
    
    check_sample = memory_usage_check(args[2])
    check_target = memory_usage_check(args[3])
    total_memory = check_target + check_sample


    return total_memory

def _divide_targetset(target: Dict, index: int)-> Dict:
    
    changed_target_boxes = target['boxes'][target["labels"] == index]
    changed_target_area = target['area'][target["labels"] == index]
    changed_target_iscrowd = target['iscrowd'][target["labels"] == index]
    changed_target_labels = target['labels'][target["labels"] == index]
    if changed_target_boxes.shape[0] == 0:
        raise Exception("Tensor has not any value. check the Tensor value")
    return {"boxes" : changed_target_boxes, "labels" : changed_target_labels, "image_id" : target["image_id"], "area" : changed_target_area,
            "iscrowd" : changed_target_iscrowd, "orig_size" : target["orig_size"], "size" : target["size"]}

@decompose #For Divide Dataset
def _rearrange_targets(no_use_count: int, samples: utils.NestedTensor, targets: Dict, origin_samples: utils.NestedTensor, origin_targets: Dict, 
                      used_number: List) -> Tuple[int, List, utils.NestedTensor, Dict, utils.NestedTensor, Dict, List]:
    no_use_count = 1 
    batch_size = 2
    return (batch_size, no_use_count, samples, targets, origin_samples, origin_targets, used_number)


def load_model_params(mode, model: model,
                      dir: str = "/home/user/Desktop/jjunsss/CL_DDETR/baseline_ddetr.pth"):
    new_model_dict = model.state_dict()
    
    if isinstance(dir, list):
        dir = dir[0]
    #temp dir
    checkpoint = torch.load(dir)
    pretraind_model = checkpoint["model"]
    name_list = [name for name in new_model_dict.keys() if name in pretraind_model.keys()]
    name_list = list(filter(lambda x : "class" not in x, name_list))
    pretraind_model_dict = {k : v for k, v in pretraind_model.items() if k in name_list } # if "class" not in k => this method used in diff class list
    
    new_model_dict.update(pretraind_model_dict)
    model.load_state_dict(new_model_dict)


    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    #No parameter update
    for name, params in model.named_parameters():
        if name in pretraind_model_dict.keys():
            if mode == "teacher":
                params.requires_grad = False #if you wanna set frozen the pre parameters for specific Neuron update, so then you could set False
        else:
            if mode == "teacher":
                params.requires_grad = False
    
    print(colored(f"Done every model params", "red", "on_yellow"))
            
    return model

def teacher_model_freeze(model):
    for _, params in model.named_parameters():
            params.requires_grad = False
                
    return model

def save_model_params(model_without_ddp:model, optimizer:torch.optim, lr_scheduler:scheduler,
                     args:arg, output_dir: str, task_index:int, task_total:int, epoch):
    '''
        Save the model for each task
    '''
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        print(f"Directroy created")
    
    if epoch == -1:
        checkpoint_paths = output_dir + f'cp_{task_total:02}_{task_index+1:02}.pth'
    else:
        checkpoint_paths = output_dir + f'cp_{task_total:02}_{task_index+1:02}_{epoch}.pth'
    utils.save_on_master({
        'model': model_without_ddp.state_dict(), 
        'optimizer': optimizer.state_dict(),
        'lr_scheduler': lr_scheduler.state_dict(),
        'args': args,
    }, checkpoint_paths)


import torch.distributed as dist
def check_training_gpu(train_check):
    world_size = utils.get_world_size()
    
    
    if world_size < 2:
        return True
    
    gpu_control_value = torch.tensor(1.0, device=torch.device("cuda"))
    temp_list = [torch.tensor(0.0, device=torch.device("cuda")) for _ in range(4)]
    
    if train_check == False:
        gpu_control_value = torch.tensor(0.0, device=torch.device("cuda"))
        
    dist.all_gather(temp_list, gpu_control_value)
    gpu_control_value = sum([ten_idx.item() for ten_idx in temp_list])
    print(f"used gpu counts : {int(gpu_control_value)}")
    if int(gpu_control_value) == 0:
        print("current using GPU counts is 0, so it's not traing")
        return False

    return True

def buffer_checker(rehearsal):
    #print text file
    check_components(rehearsal, True)
        
        
def control_lr_backbone(args, optimizer, frozen):
    if frozen is True:
        lr = 0.0
    else:
        lr = args.lr_backbone
        
    optimizer.param_groups[-1]['lr'] = lr
            
    return optimizer

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import copy
from Custom_Dataset import CombineDataset

def dataset_configuration(args, original_dataset, original_loader, original_sampler,
                          AugRplay_dataset, AugRplay_loader, AugRplay_sampler):
    
    if args.AugReplay :
        return AugRplay_dataset, AugRplay_loader, AugRplay_sampler
    
    elif args.MixReplay:
        return [AugRplay_dataset, original_dataset], [AugRplay_loader, original_loader], [AugRplay_sampler, original_sampler] 
    
    else :
        return original_dataset, original_loader, original_sampler    

#* Just CL_StepLR(CLStepLR)
class ContinualStepLR(StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, task_gamma=0.75, replay_gamma=10, last_epoch=-1, verbose=False):
        super(ContinualStepLR, self).__init__(optimizer, step_size, gamma, last_epoch, verbose)
        self.task_gamma = task_gamma

        
    def task_change(self):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = param_group['lr'] * self.task_gamma
            if self.verbose:
                print(f'Task changed: Decreasing Group {i} lr to {param_group["lr"]:.4e}')

#* for adaptive lR  
# class ContinualStepLR(StepLR):
#     def __init__(self, optimizer, step_size, gamma=0.1, task_gamma=0.5, replay_gamma=10, last_epoch=-1, verbose=False):
#         super(ContinualStepLR, self).__init__(optimizer, step_size, gamma, last_epoch, verbose)
#         self.task_gamma = task_gamma
#         self.replay_gamma =  replay_gamma
#         self.base_lr = None
        
#     def task_change(self):
#         for i, param_group in enumerate(self.optimizer.param_groups):
#             param_group['lr'] = param_group['lr'] * self.task_gamma
#             if self.verbose:
#                 print(f'Task changed: Decreasing Group {i} lr to {param_group["lr"]:.4e}')
#         # Save different learning rate in changed task.
#         self.base_lr = copy.deepcopy(self.optimizer)
#         print(f"base learning rate : {self.base_lr}")
        
#     def replay_step(self):
#         if self.base_lr is None:
#             raise Exception("First, use the original learning rate and then you can use the replay learning rate.")
#         replay_gamma = float(1.0 / self.task_gamma)
#         for i, (param_group, base_param_group) in enumerate(zip(self.optimizer.param_groups, self.base_lr.param_groups)):
#             param_group['lr'] = base_param_group['lr'] * replay_gamma
#             if self.verbose:
#                 print(f'Task changed: Increasing Group {i} lr to {param_group["lr"]:.4e}')
                
#     def original_step(self):
#         self.optimizer = copy.deepcopy(self.base_lr)

# # #* For Diff StepLR between step 1 and step2 -> ADStepLR in Training process
# class ContinualStepLR(StepLR):
#     def __init__(self, optimizer, step_size, gamma=0.1, task_gamma=1, replay_gamma=2, last_epoch=-1, verbose=False):
#         super(ContinualStepLR, self).__init__(optimizer, step_size, gamma, last_epoch, verbose)
#         self.task_gamma = task_gamma
#         self.replay_gamma =  replay_gamma
#         self.base_lr = None
#         self.replay_lr = None
        
#     def task_change(self):
#         for i, param_group in enumerate(self.optimizer.param_groups):
#             param_group['lr'] = param_group['lr'] * self.task_gamma
#             if self.verbose:
#                 print(f'Task changed: Decreasing Group {i} lr to {param_group["lr"]:.4e}')
#         # Save different learning rate in changed task.
#         self.base_lr = copy.deepcopy(self.optimizer)
#         print(f"base learning rate : {self.base_lr}")
        
#     def replay_step(self, idx):
#         if self.replay_lr is None :
#             for i, param_group in enumerate(self.optimizer.param_groups):
#                 param_group['lr'] = param_group['lr'] * self.replay_gamma
                
#                 if self.verbose:
#                     print(f'Task changed: Increasing Group {i} lr to {param_group["lr"]:.4e}')
#             self.replay_lr = copy.deepcopy(self.optimizer)
#         else:
#             self.optimizer = self.replay_lr
#             if self.verbose and (idx % 30) == 0:
#                 print(f"optimizer group setting is {self.optimizer}")
            
#     def original_step(self, idx):
#         self.optimizer = self.base_lr
#         if self.verbose and (idx % 30) == 0:
#             print(f"optimizer group setting is {self.optimizer}")
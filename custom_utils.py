from ast import arg
from pyexpat import model
from sched import scheduler
from xmlrpc.client import Boolean
import torch
import util.misc as utils
import numpy as np
from typing import Tuple, Dict, List, Optional
import os
from custom_prints import over_label_checker

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
    #print(f"{dist.get_rank()}gpu training saved dict : {saved_dict}")
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

from pympler import asizeof, summary
def contruct_rehearsal(losses_value: float, lower_limit: float, upper_limit: float, targets,
                       rehearsal_classes: List, Current_Classes: List[int], Rehearsal_Memory: int = 300) -> Dict:
    # Check if losses_value is within the specified range
    if losses_value > lower_limit and losses_value < upper_limit : 
        ex_device = torch.device("cpu")
        
        for enum, target in enumerate(targets): #! 배치 개수 ex) 4개 
            # Get the unique labels and the count of each label
            label_tensor = target['labels']
            image_id = target['image_id'].item()
            label_tensor_unique = torch.unique(label_tensor)
            label_tensor_unique_list = label_tensor_unique.tolist()
            if set(label_tensor_unique_list).issubset(Current_Classes) is False: #if unique tensor composed by Old Dataset, So then Continue iteration
                continue
            
            label_tensor_count = label_tensor.numpy()
            bin = np.bincount(label_tensor_count)
            if image_id in rehearsal_classes.keys():
                rehearsal_classes[image_id][-1].extend(label_tensor_unique_list)
                continue
            
            
            if _check_rehearsal_size(Rehearsal_Memory, rehearsal_classes, *label_tensor_unique_list) == True:
                rehearsal_classes[image_id] = [losses_value, label_tensor_unique_list]
            else:
                print(f"**** Memory over ****")
                high_loss_rehearsal = _change_rehearsal_size(Rehearsal_Memory, rehearsal_classes, *label_tensor_unique_list)
                if high_loss_rehearsal == False: #!얘를들어 unique index를 모두 포함하고 있는 rehearsal 데이터 애들이 존재하지 않는 경우에 해당 상황이 발생할 수 있다.
                    continue
                
                if high_loss_rehearsal[0] > losses_value:
                    print(f"chagne rehearsal value")
                    del rehearsal_classes[high_loss_rehearsal[0]]
                    rehearsal_classes[image_id] = [losses_value, label_tensor_unique_list]
    
    return rehearsal_classes

def _check_rehearsal_size(limit_memory_size, rehearsal_classes, *args, ):
    if len(rehearsal_classes.keys()) == 0:
        return True
    
    check_list = [len(list(filter(lambda x: index in x[1], list(rehearsal_classes.values())))) for index in args]
    
    check = all([value < limit_memory_size for value in check_list])
    return check

def _change_rehearsal_size(limit_memory_size, rehearsal_classes, *args, ): 
    check_list = [len(list(filter(lambda x: index in x[1], list(rehearsal_classes.values())))) for index in args]
    temp_array = np.array(check_list)
    temp_array = temp_array < limit_memory_size 
    
    over_list = []
    for t, arg in zip(temp_array, args):
        if t == False:
            over_list.append(arg)
            
    check_list = list(filter(lambda x: all(item in x[1][1] for item in over_list), list(rehearsal_classes.items())))
    sorted_result = sorted(check_list, key = lambda x : x[1][0])
    if len(sorted_result) == 0 :
        return False
    
    sorted_result = sorted_result[-1]

    return sorted_result

def rearrange_rehearsal(rehearsal_classes: dict, current_classes: list) -> dict:
    '''
        delete bincount, loss_value in Current class 
    '''
    for key, contents in rehearsal_classes.items():
        if key in current_classes:
            for content in contents:
                del content[:2]
            
    return rehearsal_classes

def load_model_params(model: model,
                      dir: str = "/data/LG/real_dataset/total_dataset/test_dir/Continaul_DETR/baseline_ddetr.pth"):
    new_model_dict = model.state_dict()
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
            params.requires_grad = True #if you wanna set frozen the pre parameters for specific Neuron update, so then you could set False
        else:
            params.requires_grad = True
    
    print(f"$$$$$$$ Done every model params $$$$$$$$$$")
            
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

import pickle
def save_rehearsal(task, dir, rehearsal, epoch):
    #* save the capsulated dataset(Boolean, image_id:int)
    if not os.path.exists(dir):
        os.mkdir(dir)
        print(f"Directroy created")
        
    dir = dir + str(dist.get_rank()) + "_gpu_rehearsal" + "_task_" + str(task) + "_ep_" + str(epoch)
    with open(dir, 'wb') as f:
        pickle.dump(rehearsal, f)


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
    
    
def memory_usage_check(byte_usage):
    """
        for checking memory usage in instace. 
        output : x.xx MB usage
    """
    instances_bytes = asizeof.asizeof(byte_usage) 
    memory_usage_MB = instances_bytes * 0.00000095367432
    
    return memory_usage_MB

import pickle
import os
def multigpu_rehearsal(dir, limit_memory_size, gpu_counts, task_num, epoch=0, *args):
    '''
        limit_memory_size : args.memory
        rehearsal_classes: Rehearsal classes
        args : old classes (Not Now classes)
    '''
    limit_memory_size = limit_memory_size * gpu_counts
    
    dir_list = [dir + str(num) +"_gpu_rehearsal_task_" + str(task_num) + "_ep_" + str(epoch) for num in range(gpu_counts)]
    for each_dir in dir_list:
        if os.path.isfile(each_dir) == False:
            raise Exception("No rehearsal file")
        
    merge_dict = {}
    for idx, dictionary_dir in enumerate(dir_list):
        with open(dictionary_dir, 'rb') as f :
            temp = pickle.load(f)
            merge_dict = {**merge_dict, **temp}
    
    while True:
        check_list = [len(list(filter(lambda x: index in x[1], list(merge_dict.values())))) for index in args]
        #print(check_list)
        temp_array = np.array(check_list)
        temp_array = temp_array < limit_memory_size
        #print(temp_array)
        if all(temp_array) == True:
            print(f"********** Done Combined dataset ***********")
            #dist.barrier()
            return merge_dict
        
        over_list = []
        for t, arg in zip(temp_array, args):
            if t == False:
                over_list.append(arg)
                
        check_list = list(filter(lambda x: all(item in x[1][1] for item in over_list), list(merge_dict.items())))
        sorted_result = sorted(check_list, key = lambda x : x[1][0])
        if len(sorted_result) == 0 :
            check_list = list(filter(lambda x: any(item in x[1][1] for item in over_list), list(merge_dict.items())))
            sorted_result = sorted(check_list, key = lambda x : x[1][0])
            del merge_dict[sorted_result[-1][0]]
            continue
        
        del merge_dict[sorted_result[-1][0]]
        
        
def control_lr_backbone(args, optimizer, frozen):
    if frozen is True:
        lr = 0.0
    else:
        lr = args.lr_backbone
        
    optimizer.param_groups[-1]['lr'] = lr
            
    return optimizer
import pickle
import copy
from ast import arg
from xmlrpc.client import Boolean
import torch
import util.misc as utils
import numpy as np
from typing import Tuple, Dict, List, Optional
import os
import torch.distributed as dist

def _save_rehearsal_for_combine(task, dir, rehearsal, epoch):
    #* save the capsulated dataset(Boolean, image_id:int)
    if not os.path.exists(dir):
        os.mkdir(dir)
        print(f"Directroy created")

    temp_dict = copy.deepcopy(rehearsal)
    for key, value in rehearsal.items():
        if len(value[-1]) == 0:
            del temp_dict[key]
            
    dir = dir + str(dist.get_rank()) + "_gpu_rehearsal_task_" + str(task) + "_ep_" + str(epoch)
    with open(dir, 'wb') as f:
        pickle.dump(temp_dict, f)
        
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
            if set(label_tensor_unique_list).issubset(Current_Classes) is False: #if unique tensor composed by Old Dataset, So then pass iteration
                continue
            
            label_tensor_count = label_tensor.numpy()
            bin = np.bincount(label_tensor_count)
            if image_id in rehearsal_classes.keys():
                temp = set(rehearsal_classes[image_id][-1])
                temp = temp.union(set(label_tensor_unique_list))
                rehearsal_classes[image_id][-1] = list(temp)
                continue
            
            
            if _check_rehearsal_size(Rehearsal_Memory, rehearsal_classes, *label_tensor_unique_list) == True:
                rehearsal_classes[image_id] = [losses_value, label_tensor_unique_list]
            else:
                print(f"**** Memory over ****")
                high_loss_rehearsal = _change_rehearsal_size(Rehearsal_Memory, rehearsal_classes, *label_tensor_unique_list)
                if high_loss_rehearsal == False: #!얘를들어 unique index를 모두 포함하고 있는 rehearsal 데이터 애들이 존재하지 않는 경우에 해당 상황이 발생할 수 있다.
                    continue
                
                #replay dictionary ~ [image id][Loss_value, [Unique_classes]]
                if len(high_loss_rehearsal[1][1]) < len(label_tensor_unique_list): #Low buffer construct
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

def _change_rehearsal_size(limit_memory_size, rehearsal_classes, *args,): 
    check_list = [len(list(filter(lambda x: index in x[1], list(rehearsal_classes.values())))) for index in args]
    temp_array = np.array(check_list)
    temp_array = temp_array < limit_memory_size 

    over_list = []
    for t, arg in zip(temp_array, args):
        if t == False:
            over_list.append(arg)

    check_list = list(filter(lambda x: any(item in x[1][1] for item in over_list), list(rehearsal_classes.items())))
    sorted_result = sorted(check_list, key = lambda x : len(x[1][1]), reverse=True) # low unique counts for training
    if len(sorted_result) == 0 :
        return False

    sorted_result = sorted_result[-1]

    return sorted_result

import pickle
import os
def _save_rehearsal(rehearsal, dir, task, memory):
    all_memory = memory * 4
    all_dir = dir  + "ALL_gpu_rehearsal_task_" + str(task) +"_" + str(all_memory)
    if not os.path.exists(dir):
        os.mkdir(dir)
        print(f"Directroy created")

    with open(all_dir, 'wb') as f:
        pickle.dump(rehearsal, f)
        print(f"save all rehearsal data complete")


def load_rehearsal(dir, task, memory):
    all_memory = memory * 4
    all_dir = dir  + "ALL_gpu_rehearsal_task_" + str(task) + "_" + str(all_memory)
    print(f"load replay file name : {all_dir}")
    #all_dir = "/data/LG/real_dataset/total_dataset/test_dir/Continaul_DETR/Rehearsal_dict/0_gpu_rehearsal_task_0_ep_9"
    if os.path.exists(all_dir) :
        with open(all_dir, 'rb') as f :
            temp = pickle.load(f)
            print(f"********** Done Combined replay data ***********")
            return temp

def _handle_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, current_classes, include_all=False):
    def load_dictionaries_from_files(dir_list):
        merged_dict = {}
        for dictionary_dir in dir_list:
            with open(dictionary_dir, 'rb') as f :
                temp = pickle.load(f)
                merged_dict = {**merged_dict, **temp}
        return merged_dict

    def get_over_list(temp_array, limit_memory_size, current_classes):
        over_list = {}
        temp_array = temp_array - limit_memory_size
        temp_array = np.clip(temp_array, 0, np.Infinity)
        for t, arg in zip(temp_array <= limit_memory_size, current_classes):
            if not t:
                over_list[arg] = temp_array[arg]
        return over_list

    limit_memory_size *= gpu_counts

    dir_list = [dir + str(num) +"_gpu_rehearsal_task_" + str(task) + "_ep_" + str(epoch) for num in range(gpu_counts)]

    if include_all:
        all_dir = dir  + "ALL_gpu_rehearsal_task_" + str(task) + "_" + str(limit_memory_size)
        dir_list.append(all_dir)

    for each_dir in dir_list:
        if not os.path.exists(each_dir):
            raise Exception("No rehearsal file")
            
    merge_dict = load_dictionaries_from_files(dir_list)
    
    while True:
        check_list = [len(list(filter(lambda x: index in x[1], list(merge_dict.values())))) for index in current_classes]
        print(f"check list : {check_list}")
        
        temp_array_bool = np.array(check_list) <= limit_memory_size
        print(f"merged keys : {len(list(merge_dict.keys()))}")
        
        if all(temp_array_bool):
            print(f"********** Done Combined replay data ***********")
            return merge_dict

        over_list = get_over_list(np.array(check_list), limit_memory_size, current_classes)
        sorted_over_list = sorted(over_list.items(), key=lambda x: x[1])
        print(f"sorted_overlist : {sorted_over_list}")
        
        del_classes = sorted_over_list[0][0]
        del_counts = sorted_over_list[0][1]
        print("del_indexes", del_classes)
        
        sorted_result = sorted(filter(lambda x: del_classes in x[1][1], list(merge_dict.items())), key=lambda x: x[1][0], reverse=True)
        
        del_count = int(del_counts * 1)
        print("Deleting {} elements:".format(del_count))
        
        deleted_count = 0
        for img_index, _ in sorted_result:
            if deleted_count >= del_count:
                break
            del merge_dict[img_index]
            deleted_count += 1

def _multigpu_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, current_classes):
    return _handle_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, current_classes, include_all=False)

def _merge_replay_for_multigpu(args, dir, limit_memory_size, gpu_counts, task, epoch, current_classes):
    return _handle_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, current_classes, include_all=True)
    

def construct_combined_rehearsal(args, task:int ,dir:str ,rehearsal:dict ,epoch:int 
                                 ,limit_memory_size:int ,list_CC:list, gpu_counts:int, ) -> dict:
    all_memory = limit_memory_size * 4
    all_dir = dir  + "ALL_gpu_rehearsal_task_" + str(task) +"_" + str(all_memory)
    
    #file save of each GPUs
    _save_rehearsal_for_combine(task, dir, rehearsal, epoch)

    if os.path.isfile(all_dir):
        # Constructing all gpu (기존에 존재하는 replay 데이터와 합치기 위해)
        rehearsal_classes = _merge_replay_for_multigpu(args, dir, limit_memory_size, gpu_counts, task, epoch, list_CC)
    else :    
        # 기존에 만들어진 합성 replay 데이터가 없을 때, 새롭게 만들어야 하는 상황을 가정
        rehearsal_classes = _multigpu_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, list_CC)

    #save combined replay buffer data for next training
    if utils.is_main_process():
        _save_rehearsal(rehearsal_classes, dir, task, limit_memory_size)
    
    return rehearsal_classes

from Custom_Dataset import *
from custom_prints import *
from engine import train_one_epoch
from custom_utils import buffer_checker
def contruct_replay_extra_epoch(args, Divided_Classes, model, criterion, device):
    rehearsal_classes = {}
    
    # 1. 현재 테스크에 맞는 적절한 데이터 셋 호출 (학습한 테스크, 0번 테스크에 해당하는 내용을 가져와야 함)
    _, data_loader_train, _, list_CC = Incre_Dataset(0, args, Divided_Classes) 
    
    # 2. Extra epoch를 통해서 모든 이미지들의 Loss를 측정
    rehearsal_classes = train_one_epoch(args, last_task = False, epo = 0, model=model, teacher_model=None,
                                        criterion=criterion, data_loader=data_loader_train, optimizer=None,
                                        lr_scheduler=None, device=device, dataset_name=None, current_classes=list_CC, 
                                        rehearsal_classes=rehearsal_classes)
    
    # 3. Extra epoch를 통해 수집된 replay data(multi-gpu로 각각 생성)를 합치고 통합하는 과정
    # TODO : 기존에 만들어진 merge dict가 있다면 합치는 동작을 해야하는데 이 부분 자동화하는 작업 수행해야 함.
    rehearsal_classes = construct_combined_rehearsal(args=args, task=0, dir=args.Rehearsal_file, rehearsal=rehearsal_classes,
                                                     epoch=0, limit_memory_size=args.Memory, gpu_counts=4, list_CC=list_CC)
    
    # 4. 수집된 replay buffer 데이터를 정리해서 확인
    buffer_checker(rehearsal=rehearsal_classes)
    
    
    print(f"Extra training process, for conducting replay dataset")
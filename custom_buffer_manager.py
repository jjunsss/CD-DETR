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

def _multigpu_rehearsal(dir, limit_memory_size, gpu_counts, task, epoch, current_classes):
    '''
        limit_memory_size : current_classes.memory
        rehearsal_classes: Rehearsal classes
        gpu_counts : the number of all GPUs
        args : old classes or current classes 
        epoch : now epoch
        task : now task
    '''
    limit_memory_size = limit_memory_size * gpu_counts

        
    dir_list = [dir + str(num) +"_gpu_rehearsal_task_" + str(task) + "_ep_" + str(epoch) for num in range(gpu_counts)]
    for each_dir in dir_list:
        if os.path.exists(each_dir) == False:
            raise Exception("No rehearsal file")
        
    merge_dict = {}
    for idx, dictionary_dir in enumerate(dir_list):
        with open(dictionary_dir, 'rb') as f :
            temp = pickle.load(f)
            merge_dict = {**merge_dict, **temp}
    
    while True:
        check_list = [len(list(filter(lambda x: index in x[1], list(merge_dict.values())))) for index in current_classes]
        temp_array = np.array(check_list)
        print(f"check list : {check_list}")
        temp_array_bool = temp_array <= limit_memory_size
        print(f"merged keys : {len(list(merge_dict.keys()))}")
        
        if all(temp_array_bool) == True:
            print(f"********** Done Combined replay data ***********")
            return merge_dict

        over_list = {}
        temp_array = temp_array - limit_memory_size
        temp_array = np.clip(temp_array, 0, np.Infinity)
        print("temp_array", temp_array)
        for idx, (t, arg) in enumerate(zip(temp_array_bool, current_classes)):
            if t == False:
                over_list[arg] = temp_array[idx]

        # 수가 적은 것부터 정렬
        sorted_over_list = sorted(over_list.items(), key=lambda x: x[1])
        print(f"sorted_overlist : {sorted_over_list}")
        
        del_classes = sorted_over_list[0][0]
        del_counts = sorted_over_list[0][1]
        
        #얘가 계속해서 변경되어야 맞음
        print("del_indexes", del_classes)
        
        check_list = list(filter(lambda x: del_classes in x[1][1], list(merge_dict.items())))
        sorted_result = sorted(check_list, key=lambda x: x[1][0], reverse=True)
        
        del_count = int(del_counts * 1)
        print("Deleting {} elements:".format(del_count))
        
        # Delete elements in merge_dict according to the sorted result and del_count
        deleted_count = 0
        for img_index, _ in sorted_result:
            if deleted_count >= del_count:
                break
            # Ensure that del_count is greater than 0 before deleting
            if del_count > 0:
                del merge_dict[img_index]
                deleted_count += 1
                
        continue

def construct_combined_rehearsal(task:int ,dir:str ,rehearsal:dict ,epoch:int ,limit_memory_size:int ,list_CC:list,
                                 gpu_counts:int, ) -> dict:
    #file save of each GPUs
    _save_rehearsal_for_combine(task, dir, rehearsal, epoch)

    #combine replay buffer data in each GPUs processes
    rehearsal_classes = _multigpu_rehearsal(dir, limit_memory_size, gpu_counts, task, epoch, list_CC)

    #save combined replay buffer data for next training
    if utils.is_main_process():
        _save_rehearsal(rehearsal_classes, dir, task, limit_memory_size)
    
    return rehearsal_classes
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
import random
from util.misc import get_world_size
from termcolor import colored

#TODO : Change calc each iamage loss and tracking each object loss avg.
def _replacment_strategy(args, loss_value, targeted, rehearsal_classes,
                       label_tensor_unique_list, image_id, num_bounding_boxes):
    if args.Sampling_strategy == "hierarchical" : 
        if ( targeted[1][0] > loss_value ): #Low buffer construct
            print(colored(f"hierarchical based buffer change strategy", "blue"))
            del rehearsal_classes[targeted[0]]
            rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list, num_bounding_boxes]
            return rehearsal_classes
        
    if args.Sampling_strategy  == "high_uniq": # This is same that "hard sampling"
        if ( len(targeted[1][1]) < len(label_tensor_unique_list) ): #Low buffer construct
            print(colored(f"high-unique counts based buffer change strategy", "blue"))
            del rehearsal_classes[targeted[0]]
            rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list, num_bounding_boxes]
            return rehearsal_classes
            
    if args.Sampling_strategy  == "random" :
        print(colored(f"random counts based buffer change strategy", "blue"))
        key_to_delete = random.choice(list(rehearsal_classes.keys()))
        del rehearsal_classes[key_to_delete]
        rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list, num_bounding_boxes]
        return rehearsal_classes
    
    if args.Sampling_strategy == "hard":
        # This is same as "hard sampling"
        if targeted[1][2] < num_bounding_boxes:  # Low buffer construct
            print(colored(f"hard sampling based buffer change strategy", "blue"))
            del rehearsal_classes[targeted[0]]
            rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list, num_bounding_boxes]
            return rehearsal_classes


    print(f"no changed")
    return rehearsal_classes

def _change_available_list_mode(mode, rehearsal_dict, need_to_include, least_image, current_classes):
    '''
        각 유니크 객체의 개수를 세어 제한하는 것은 그에 맞는 이미지가 존재해야만 모일수도 있기때문에 모두 모을 수 없을 수도 있게되는 불상사가 있다.
        따라서 객체의 개수를 제한하는 것보다는 그에 맞게 비율을 따져서 이미지를 제한하는 방법이 더 와닿을 수 있다.
    '''
    if mode == "normal":
        # no limit and no least images
        changed_available_dict = rehearsal_dict
        
    if mode == "ensure_min":
        image_counts_in_rehearsal = {class_label: sum(class_label in classes for _, (_, classes, _) in rehearsal_dict.items()) for class_label in current_classes}
        print(f"replay counts : {image_counts_in_rehearsal}")
        
        changed_available_dict = {key: (losses, classes, bboxes) for key, (losses, classes, bboxes) in rehearsal_dict.items() if all(image_counts_in_rehearsal[class_label] > least_image for class_label in classes)}
        # print(f"available counts : {changed_available_dict}")
        
        if len(changed_available_dict.keys()) == 0 :
            # this process is protected to generate error messages
            # include classes that have at least one class in need_to_include
            print(colored(f"no changed available dict, suggest to reset your least image count", "blue"))
            temp_dict = {key: len([c for c in items[1] if c in need_to_include]) for key, items in rehearsal_dict.items() if any(c in need_to_include for c in items[1])}

            # sort the temporary dictionary by values (counts of classes from need_to_include)
            sorted_temp_dict = dict(sorted(temp_dict.items(), key=lambda item: item[1]))

            # get the first key in the sorted dictionary as min_key
            min_key = next(iter(sorted_temp_dict))

            # create the new changed_available_dict with entries that have the minimum number of classes from need_to_include
            changed_available_dict = {key:items for key, items in rehearsal_dict.items() if len([c for c in items[1] if c in need_to_include]) == sorted_temp_dict[min_key]}
    
    # TODO:  in CIL method, {K / |C|} usage
    # if mode == "classification":
    #     num_classes = len(classes)
    #     initial_limit = limit_image // num_classes
    #     limit_memory = {class_index: initial_limit for class_index in classes}
        
    return changed_available_dict

def contruct_rehearsal(args, losses_dict: dict, targets, rehearsal_dict: List, 
                       current_classes: List[int], least_image: int = 3, limit_image:int = 100) -> Dict:

    loss_value = 0.0
    for enum, target in enumerate(targets): #! 배치 개수 ex) 4개 
        loss_value = losses_dict["loss_bbox"][enum] + losses_dict["loss_giou"][enum] + losses_dict["loss_labels"][enum]
        if loss_value > 10.0 :
            continue
        # Get the unique labels and the count of each label
        label_tensor = target['labels']
        bbox_counts = label_tensor.shape[0]
        image_id = target['image_id'].item()
        label_tensor_unique = torch.unique(label_tensor)
        label_tensor_unique_list = label_tensor_unique.tolist()
        #if unique tensor composed by Old Dataset, So then pass iteration [Replay constructig shuld not operate in last task training]
        if set(label_tensor_unique_list).issubset(current_classes) is False: 
            continue

        if len(rehearsal_dict.keys()) <  limit_image :
            # when under the buffer 
            rehearsal_dict[image_id] = [loss_value, label_tensor_unique_list, bbox_counts]
        else :
            if args.Sampling_strategy == "hard" : 
                rehearsal_dict = _replacment_strategy(args=args, loss_value=loss_value, targeted=targeted, 
                                                        rehearsal_classes=rehearsal_dict, label_tensor_unique_list=label_tensor_unique_list,
                                                        image_id=image_id)
                
                
            # First, generate a dictionary with counts of each class label in rehearsal_classes
            image_counts_in_rehearsal = {class_label: sum(class_label in classes for _, classes, _ in rehearsal_dict.values()) for class_label in label_tensor_unique_list}

            # Then, calculate the needed count for each class label and filter out those with a non-positive needed count
            need_to_include = {class_label: count - least_image for class_label, count in image_counts_in_rehearsal.items() if count - least_image <= 0}

            if len(need_to_include) > 0:
                changed_available_dict = _change_available_list_mode(mode=args.Sampling_mode, rehearsal_dict=rehearsal_dict,
                                            need_to_include=need_to_include, least_image=least_image, current_classes=current_classes)
                
                # all classes dont meet L requirement
                targeted = _calc_target(rehearsal_classes=changed_available_dict, replace_strategy=args.Sampling_strategy, )
                
                del rehearsal_dict[targeted[0]]
                rehearsal_dict[image_id] = [loss_value, label_tensor_unique_list, bbox_counts]
            else :
                changed_available_dict = _change_available_list_mode(mode=args.Sampling_mode, rehearsal_dict=rehearsal_dict,
                                            need_to_include=need_to_include, least_image=least_image, current_classes=current_classes)
                
                # all classes dont meet L requirement
                targeted = _calc_target(rehearsal_classes=changed_available_dict, replace_strategy=args.Sampling_strategy, )
                rehearsal_dict = _replacment_strategy(args=args, loss_value=loss_value, targeted=targeted, 
                                        rehearsal_classes=rehearsal_dict, label_tensor_unique_list=label_tensor_unique_list,
                                        image_id=image_id, num_bounding_boxes=bbox_counts)
    

    return rehearsal_dict

def _check_rehearsal_size(limit_memory_size, rehearsal_classes, unique_classes_list, ):
    if len(rehearsal_classes.keys()) == 0:
        return True
    
    check_list = [len(list(filter(lambda x: index in x[1], list(rehearsal_classes.values())))) for index in unique_classes_list]
    
    check = all([value < limit_memory_size for value in check_list])
    return check

def _calc_target(rehearsal_classes, replace_strategy="hierarchical", ): 

    if replace_strategy == "hierarchical":
        # ours for effective, mode is "ensure_min"
        min_class_length = min(len(x[1]) for x in rehearsal_classes.values())
        
        # first change condition: low unique based change
        changed_list = [(index, values) for index, values in rehearsal_classes.items() if len(values[1]) == min_class_length]
    
        # second change condition: low loss based change
        sorted_result = max(changed_list, key=lambda x: x[1][0])
        
    elif replace_strategy == "high_uniq": 
        # only high unique based change, mode is "normal" or "random"
        sorted_result = min(changed_list, key=lambda x: len(x[1][1]))
        
    elif replace_strategy == "random":
        # only random change, mode is "normal" or "random"
        sorted_result = None
        
    elif replace_strategy == "low_loss":
        # only low loss based change, mode is "normal" or "random"
        sorted_result = max(rehearsal_classes, key=lambda x: x[1][0])
        
    elif replace_strategy == "hard":
        # only high bounding box count based change, mode is "normal" or "random"
        sorted_result = min(rehearsal_classes.items(), key=lambda x: x[1][2])

    return sorted_result

def _save_rehearsal_for_combine(task, dir, rehearsal, epoch):
    #* save the capsulated dataset(Boolean, image_id:int)
    if not os.path.exists(dir) and utils.is_main_process() :
        os.mkdir(dir)
        print(f"Directroy created")
    
    if not os.path.exists(dir+"backup/") and utils.is_main_process() :
        os.mkdir(dir+"backup/")
        print(f"Backup directory created")    
        
    if utils.get_world_size() > 1:
        dist.barrier()

    temp_dict = copy.deepcopy(rehearsal)
    for key, value in rehearsal.items():
        if len(value[1]) == 0:
            del temp_dict[key]
            
    backup_dir = dir + "backup/" + str(dist.get_rank()) + "_gpu_rehearsal_task_" + str(task) + "_ep_" + str(epoch)
    dir = dir + str(dist.get_rank()) + "_gpu_rehearsal_task_" + str(task) + "_ep_" + str(epoch)
    with open(dir, 'wb') as f:
        pickle.dump(temp_dict, f)
        
    with open(backup_dir, 'wb') as f:
        pickle.dump(temp_dict, f)

import pickle
import os
def _save_rehearsal(rehearsal, dir, task, memory):
    all_dir = os.path.join(dir, "Buffer_T_" + str(task) +"_" + str(memory))
    if not os.path.exists(dir):
        os.mkdir(dir)
        print(f"Directroy created")

    with open(all_dir, 'wb') as f:
        pickle.dump(rehearsal, f)
        print(colored(f"Save task buffer", "light_red", "on_yellow"))


def load_rehearsal(dir, task=None, memory=None):
    if task==None and memory==None:
        all_dir = dir
    else:
        all_dir = os.path.join(dir, "Buffer_T_" + str(task) + "_" + str(memory))
    print(f"load replay file name : {all_dir}")
    if os.path.exists(all_dir) :
        with open(all_dir, 'rb') as f :
            temp = pickle.load(f)
            print(colored(f"********** Loading replay data ***********", "light_red", "on_yellow"))
            return temp

def _handle_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, least_image, list_CC, include_all=False):
    def load_dictionaries_from_files(dir_list):
        merged_dict = {}
        for dictionary_dir in dir_list:
            with open(dictionary_dir, 'rb') as f :
                temp = pickle.load(f)
                merged_dict = {**merged_dict, **temp}
        return merged_dict

    dir_list = [
        os.path.join(
            dir,
            str(num) +"_gpu_rehearsal_task_" + str(task) + "_ep_" + str(epoch)
        ) for num in range(gpu_counts)
    ]
    
    if include_all:
        all_dir = os.path.join(dir, "Buffer_T_" + str(task) + "_" + str(limit_memory_size))
        dir_list.append(all_dir)

    for each_dir in dir_list:
        if not os.path.exists(each_dir):
            raise Exception("No rehearsal file")
            
    
    print(colored(f"Total memory : {len(dir_list)} ", "blue"))
    merge_dict = load_dictionaries_from_files(dir_list)
    
    # For only one GPU processing, becuase effective buffer constructing
    print(colored(f"New buffer dictionary genrating for optimizing replay dataset", "dark_grey", "on_yellow"))
    new_buffer_dict = {}
    for img_idx in merge_dict.keys():
        loss_value = merge_dict[img_idx][0]
        unique_classes_list = merge_dict[img_idx][1]
        bbox_counts = merge_dict[img_idx][2]
                                                # 0 -> loss value
                                                # 1 -> unique classes list

        if len(new_buffer_dict.keys()) <  limit_memory_size :                                        
            new_buffer_dict[img_idx] = merge_dict[img_idx]
        else : 
            # First, generate a dictionary with counts of each class label in rehearsal_classes
            image_counts_in_rehearsal = {class_label: sum(class_label in classes for _, classes, _ in new_buffer_dict.values()) for class_label in unique_classes_list}

            # Then, calculate the needed count for each class label and filter out those with a non-positive needed count
            need_to_include = {class_label: count - least_image for class_label, count in image_counts_in_rehearsal.items() if (count - least_image) <= 0}
            if len(need_to_include) > 0:
                changed_available_dict = _change_available_list_mode(mode=args.Sampling_mode, rehearsal_dict=new_buffer_dict,
                                            need_to_include=need_to_include, least_image=least_image, current_classes=list_CC)
                
                # all classes dont meet L requirement
                targeted = _calc_target(rehearsal_classes=changed_available_dict, replace_strategy=args.Sampling_strategy, )
                
                del new_buffer_dict[targeted[0]]
                new_buffer_dict[img_idx] = [loss_value, unique_classes_list, bbox_counts]
                    
            else :
                changed_available_dict = _change_available_list_mode(mode=args.Sampling_mode, rehearsal_dict=new_buffer_dict,
                                            need_to_include=need_to_include, least_image=least_image, current_classes=list_CC)
            
                # all classes meet L requirement
                # Just sampling strategy and replace strategy
                targeted = _calc_target(rehearsal_classes=changed_available_dict, replace_strategy=args.Sampling_strategy,)

                new_buffer_dict = _replacment_strategy(args=args, loss_value=loss_value, targeted=targeted, 
                                                        rehearsal_classes=new_buffer_dict, label_tensor_unique_list=unique_classes_list,
                                                        image_id=img_idx, num_bounding_boxes=bbox_counts)
            
    print(colored(f"Complete generating new buffer", "dark_grey", "on_yellow"))
    return new_buffer_dict

def _multigpu_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, least_image, list_CC):
    return _handle_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, least_image, list_CC, include_all=False)

def _merge_replay_for_multigpu(args, dir, limit_memory_size, gpu_counts, task, epoch, least_image, list_CC):
    return _handle_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, least_image, list_CC, include_all=True)
    

def construct_combined_rehearsal(args, task:int ,dir:str ,rehearsal:dict ,epoch:int 
                                 ,limit_memory_size:int , list_CC:list, gpu_counts:int, ) -> dict:
    least_image = args.least_image
    # total_size = limit_memory_size * get_world_size()
    dir = os.path.join(dir, 'replay')
    all_dir = os.path.join(dir, "Buffer_T_" + str(task) +"_" + str(limit_memory_size))
    
    #file save of each GPUs
    _save_rehearsal_for_combine(task, dir, rehearsal, epoch)
    
    # All GPUs ready replay buffer combining work(protecting some errors)
    if utils.get_world_size() > 1:    
        dist.barrier()
        
    if utils.is_main_process() : 
        if os.path.isfile(all_dir):
            # Constructing all gpu (기존에 존재하는 replay 데이터와 합치기 위해), Because Multi Task Incrmental Learning
            rehearsal_classes = _merge_replay_for_multigpu(args, dir, limit_memory_size, gpu_counts, task, epoch, least_image, list_CC)
        else :    
            # 기존에 만들어진 합성 replay 데이터가 없을 때, 새롭게 만들어야 하는 상황을 가정, Becaus Binary Task Incremental Learning
            rehearsal_classes = _multigpu_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, least_image, list_CC)
        # #save combined replay buffer data for next training
        _save_rehearsal(rehearsal_classes, dir, task, limit_memory_size)
        
        buffer_checker(rehearsal=rehearsal_classes)
    
    # wait main process to finish
    if utils.get_world_size() > 1:    
        dist.barrier()

    # All GPUs ready replay dataset
    rehearsal_classes = load_rehearsal(all_dir)
    return rehearsal_classes

from Custom_Dataset import *
from custom_prints import *
from engine import train_one_epoch
from custom_utils import buffer_checker

def contruct_replay_extra_epoch(args, Divided_Classes, model, criterion, device, rehearsal_classes={}, data_loader_train=None, list_CC=None):
    
    # 0. Initialization
    extra_epoch = True
    
    # 1. 현재 테스크에 맞는 적절한 데이터 셋 호출 (학습한 테스크, 0번 테스크에 해당하는 내용을 가져와야 함)
    #    하나의 GPU로 Buffer 구성하기 위해서(더 정확함) 모든 데이터 호출
    _, data_loader_train, _, list_CC = Incre_Dataset(0, args, Divided_Classes, extra_epoch) 
    
    # 2. Extra epoch, 모든 이미지들의 Loss를 측정
    rehearsal_classes = train_one_epoch(args, last_task=False, epo=0, model=model, teacher_model=None,
                                        criterion=criterion, data_loader=data_loader_train, optimizer=None,
                                        lr_scheduler=None, device=device, dataset_name="", current_classes=list_CC, 
                                        rehearsal_classes=rehearsal_classes, extra_epoch=extra_epoch)

    # 3. 수집된 Buffer를 특정 파일에 저장
    rehearsal_classes = construct_combined_rehearsal(args=args, task=0, dir=args.Rehearsal_file, rehearsal=rehearsal_classes,
                                                    epoch=0, limit_memory_size=args.limit_image, gpu_counts=utils.get_world_size(), list_CC=list_CC)
    
    print(colored(f"Complete constructing buffer","red", "on_yellow"))
    
    return rehearsal_classes
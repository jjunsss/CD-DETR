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
def _sampling_strategy(args, loss_value, targeted, rehearsal_classes,
                       label_tensor_unique_list, image_id):
    if args.Sampling_strategy == "low_loss" : 
        if ( targeted[1][0] > loss_value ): #Low buffer construct
            print(f"change rehearsal value")
            print(colored(f"low-loss based buffer change strategy", "blue"))
            del rehearsal_classes[targeted[0]]
            rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list]
            return rehearsal_classes
        
    if args.Sampling_strategy  == "high_uniq":
        if ( len(targeted[1][1]) < len(label_tensor_unique_list) ): #Low buffer construct
            print(f"change rehearsal value")
            print(colored(f"high-unique counts based buffer change strategy", "blue"))
            del rehearsal_classes[targeted[0]]
            rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list]
            return rehearsal_classes
            
    if args.Sampling_strategy  == "random" :
        print(f"change rehearsal value")
        print(colored(f"random counts based buffer change strategy", "blue"))
        key_to_delete = random.choice(list(rehearsal_classes.keys()))
        del rehearsal_classes[key_to_delete]
        rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list]
        return rehearsal_classes

    print(f"no changed")
    return rehearsal_classes

def _limit_classes(mode, classes, limit_image):
    '''
        각 유니크 객체의 개수를 세어 제한하는 것은 그에 맞는 이미지가 존재해야만 모일수도 있기때문에 모두 모을 수 없을 수도 있게되는 불상사가 있다.
        따라서 객체의 개수를 제한하는 것보다는 그에 맞게 비율을 따져서 이미지를 제한하는 방법이 더 와닿을 수 있다.
    '''
    if mode == "normal":
        num_classes = len(classes)
        initial_limit = limit_image // num_classes
        limit_memory = {class_index: initial_limit for class_index in classes}
        
    if mode == "random":
        limit_memory = None
        
    if mode == "rate":
        num_classes = len(classes)
        initial_limit = limit_image // num_classes
        limit_memory = {class_index: initial_limit for class_index in classes}
        
    return limit_memory

def contruct_rehearsal(args, losses_dict: dict, targets, rehearsal_classes: List, 
                       current_classes: List[int], limit_memory: int = 300, limit_image:int = 100) -> Dict:
    # TODO 1: COCO 데이터셋 전체에서의 각 클래스별 객체의 비율을 계산 -> 이건 데이터셋 호출하는 부분에서 진행해야함
    # TODO 1-2: 전체 객체들의 개수로 Normalize 진행
    # TODO 1-3: Normalize한 비율을 각 클래스들을 제한할 Dictionary에 저장
    # limit_memory = {"class index" : limit number}
    loss_value = 0.0
    increment_value = 2 # hyper parameters
    limit_memory = _limit_classes(args.Limit_strategy, current_classes, limit_image)
    
    # Function to update class limit
    def update_class_limits(limit_memory, increment):
        for class_index in limit_memory.keys():
            limit_memory[class_index] += increment
        return limit_memory
        
    # Function to check if there's space in buffer
    def buffer_has_space(rehearsal_classes, limit_image):
        return len(rehearsal_classes) < limit_image

    for enum, target in enumerate(targets): #! 배치 개수 ex) 4개 
        loss_value = losses_dict["loss_bbox"][enum] + losses_dict["loss_giou"][enum] + losses_dict["loss_labels"][enum]
        if loss_value > 10.0 :
            continue
        
        # Get the unique labels and the count of each label
        label_tensor = target['labels']
        image_id = target['image_id'].item()
        label_tensor_unique = torch.unique(label_tensor)
        label_tensor_unique_list = label_tensor_unique.tolist()
        
        #if unique tensor composed by Old Dataset, So then pass iteration [Replay constructig shuld not operate in last task training]
        if set(label_tensor_unique_list).issubset(current_classes) is False: 
            continue
        
        # In your construct_rehearsal loop or any other suitable place
        while buffer_has_space(rehearsal_classes, limit_image):
            # If the buffer still has space, increment the class limits
            if buffer_has_space(rehearsal_classes, limit_image):
                limit_memory = update_class_limits(limit_memory, increment_value) # increment_value is hyps
                
                
        # 이미지 제한 확인 (정해진 양 만큼은 절대 넘을 수 없도록 설정)
        if len(rehearsal_classes.keys()) >= limit_image :
            #! 교체 전략 수행 | TODO : 모든 객체들의 클래스를 대표하는 이미지들을 수집해서 형성
            # 이미지 제한을 초과한다면 -> 교체전략 혹은 유지 전략
            # TODO 1: limit_memory를 dictionary로 설정해서 현재 버퍼 내의 클래스별로 개수를 다양하게 설정하도록 만들기 -> 
                    # 제한값 설정 없음(랜덤)/균등/데이터분포량
            targeted = _calc_to_be_changed_target(limit_memory_size=limit_memory, rehearsal_classes=rehearsal_classes,
                       replace_strategy=args.Sampling_strategy, args=label_tensor_unique_list)
            
            # Real replacement strategy (loss-based, unique_classe-based, random)
            rehearsal_classes = _sampling_strategy(args=args, loss_value=loss_value, targeted=targeted, 
                                                rehearsal_classes=rehearsal_classes, label_tensor_unique_list=label_tensor_unique_list,
                                                image_id=image_id)  

        else : 
            if _check_rehearsal_size(limit_memory, rehearsal_classes, label_tensor_unique_list) :
                rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list]
            else :
                targeted = _calc_to_be_changed_target(limit_memory_size=limit_memory, rehearsal_classes=rehearsal_classes,
                       replace_strategy=args.Sampling_strategy, args=label_tensor_unique_list)
            
                # Real replacement strategy (loss-based, unique_classe-based, random)
                rehearsal_classes = _sampling_strategy(args=args, loss_value=loss_value, targeted=targeted, 
                                                    rehearsal_classes=rehearsal_classes, label_tensor_unique_list=label_tensor_unique_list,
                                                    image_id=image_id)  
    
    return rehearsal_classes

def _check_rehearsal_size(limit_memory_size, rehearsal_classes, unique_classes_list, ):
    if len(rehearsal_classes.keys()) == 0:
        return True
    
    check_list = [len(list(filter(lambda x: index in x[1], list(rehearsal_classes.values())))) for index in unique_classes_list]
    
    check = all([value < limit_memory_size for value in check_list])
    return check

def _calc_to_be_changed_target(limit_memory_size, rehearsal_classes, replace_strategy="low_loss", args=[]): 
    '''
        rehearsal_classes : replay data in buffer before change statement
        args : unique classes in a data(image)
    '''
    check_list = [len([item for item in rehearsal_classes.values() if index in item]) for index in args]
    # check_list = [len(list(filter(lambda x: index in x[1], list(rehearsal_classes.values())))) for index in args]
    temp_array = np.array(check_list)
    # compate to each unique class counts (ex. overlist = [1: 100, 2: 50, ...])
    check_temp_array = (temp_array < limit_memory_size)

    over_list = []
    for t, arg in zip(check_temp_array, args):
        if t == False:
            over_list.append(arg)

    # Hierarchcal replacement strategy. (1. all same thing change, 2. the most included changed classes, 3. max loss-based change)
    ## First, all same unique class list 
    same_list = list(filter(lambda x: set(x[1][1]) == set(over_list), list(rehearsal_classes.items())))
    
    ## sec, any unique classes that included a over unique classes at least
    any_check_list = list(filter(lambda x: any(item in x[1][1] for item in over_list), list(rehearsal_classes.items())))
    
    # rehearsal_classes.items() -> [image index, [loss_value, unique object classes]]
    # find all items that include any class from over_list
    if len(any_check_list) == 0 :
        raise Exception("NO CAHNGED DATA SAMPLE IN BUFFER, PLZ CHECK DICTIONARY")
    
    if len(same_list) > 0 :
        changed_list = same_list
    else :
        # find the item(s) with the maximum count of over_classes
        # if not use this term, so replacement strategy able change any items
        max_over_count = max([len([item in x[1][1] for item in over_list]) for x in any_check_list])
        changed_list = list(filter(lambda x: len([item in x[1][1] for item in over_list]) == max_over_count, any_check_list))

    if replace_strategy == "low_loss":
        # among the items with max_over_count, find the one with the highest loss value
        sorted_result = max(changed_list, key=lambda x: x[1][0])
        
    elif replace_strategy == "high_uniq":
        # among the items with max_over_count, find the one with the smallest unique class count
        sorted_result = min(changed_list, key=lambda x: len(x[1][1]))
        
    elif replace_strategy == "random":
        # because we don't need last sample with random sampling.
        sorted_result = None


    return sorted_result

def _save_rehearsal_for_combine(task, dir, rehearsal, epoch):
    #* save the capsulated dataset(Boolean, image_id:int)
    if not os.path.exists(dir) and utils.is_main_process() :
        os.mkdir(dir)
        print(f"Directroy created")
    dist.barrier()

    temp_dict = copy.deepcopy(rehearsal)
    for key, value in rehearsal.items():
        if len(value[-1]) == 0:
            del temp_dict[key]
            
    dir = dir + str(dist.get_rank()) + "_gpu_rehearsal_task_" + str(task) + "_ep_" + str(epoch)
    if not os.path.exists(dir) : 
        with open(dir, 'wb') as f:
            pickle.dump(temp_dict, f)

import pickle
import os
def _save_rehearsal(rehearsal, dir, task, memory):
    all_dir = dir  + "Buffer_T_" + str(task) +"_" + str(memory)
    if not os.path.exists(dir):
        os.mkdir(dir)
        print(f"Directroy created")

    with open(all_dir, 'wb') as f:
        pickle.dump(rehearsal, f)
        print(colored(f"Save task buffer", "light_red", "on_yellow"))


def load_rehearsal(dir, task, memory):
    all_dir = dir  + "Buffer_T_" + str(task) + "_" + str(memory)
    print(f"load replay file name : {all_dir}")
    #all_dir = "/data/LG/real_dataset/total_dataset/test_dir/Continaul_DETR/Rehearsal_dict/0_gpu_rehearsal_task_0_ep_9"
    if os.path.exists(all_dir) :
        with open(all_dir, 'rb') as f :
            temp = pickle.load(f)
            print(colored(f"********** Loading replay data ***********", "light_red", "on_yellow"))
            return temp

def _handle_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, current_classes, include_all=False):
    def load_dictionaries_from_files(dir_list):
        merged_dict = {}
        for dictionary_dir in dir_list:
            with open(dictionary_dir, 'rb') as f :
                temp = pickle.load(f)
                merged_dict = {**merged_dict, **temp}
        return merged_dict

    dir_list = [dir + str(num) +"_gpu_rehearsal_task_" + str(task) + "_ep_" + str(epoch) for num in range(gpu_counts)]

    if include_all:
        all_dir = dir  + "Buffer_T_" + str(task) + "_" + str(limit_memory_size)
        dir_list.append(all_dir)

    for each_dir in dir_list:
        if not os.path.exists(each_dir):
            raise Exception("No rehearsal file")
            
    merge_dict = load_dictionaries_from_files(dir_list)
    
    # For only one GPU processing, becuase effective buffer constructing
    print(colored(f"New buffer dictionary genrating for optimizing replay dataset", "dark_grey", "on_yellow"))
    new_buffer_dict = {}
    for img_idx in merge_dict.keys():
        loss_value = merge_dict[img_idx][0]
        unique_classes_list = merge_dict[img_idx][1]
                                                # 0 -> loss value
                                                # 1 -> unique classes list
        if _check_rehearsal_size(limit_memory_size, new_buffer_dict, unique_classes_list):
            new_buffer_dict[img_idx] = merge_dict[img_idx]
        else : 
            targeted = _calc_to_be_changed_target(limit_memory_size=limit_memory_size, rehearsal_classes=new_buffer_dict,
                       replace_strategy=args.Sampling_strategy, args=unique_classes_list)
            
            # Real replacement strategy (loss-based, unique_classe-based, random)
            new_buffer_dict = _sampling_strategy(args=args, loss_value=loss_value, targeted=targeted, 
                                                rehearsal_classes=new_buffer_dict, label_tensor_unique_list=unique_classes_list,
                                                image_id=img_idx)  
            
    print(colored(f"Complete generating new buffer", "dark_grey", "on_yellow"))
    return new_buffer_dict

def _multigpu_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, current_classes):
    return _handle_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, current_classes, include_all=False)

def _merge_replay_for_multigpu(args, dir, limit_memory_size, gpu_counts, task, epoch, current_classes):
    return _handle_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, current_classes, include_all=True)
    

def construct_combined_rehearsal(args, task:int ,dir:str ,rehearsal:dict ,epoch:int 
                                 ,limit_memory_size:int ,list_CC:list, gpu_counts:int, ) -> dict:
    
    # total_size = limit_memory_size * get_world_size()
    all_dir = dir  + "Buffer_T_" + str(task) +"_" + str(limit_memory_size)
    
    #file save of each GPUs
    _save_rehearsal_for_combine(task, dir, rehearsal, epoch)

    if utils.is_main_process() : 
        if os.path.isfile(all_dir):
            # Constructing all gpu (기존에 존재하는 replay 데이터와 합치기 위해), Because Multi Task Incrmental Learning
            rehearsal_classes = _merge_replay_for_multigpu(args, dir, limit_memory_size, gpu_counts, task, epoch, list_CC)
        else :    
            # 기존에 만들어진 합성 replay 데이터가 없을 때, 새롭게 만들어야 하는 상황을 가정, Becaus Binary Task Incremental Learning
            rehearsal_classes = _multigpu_rehearsal(args, dir, limit_memory_size, gpu_counts, task, epoch, list_CC)

        # #save combined replay buffer data for next training
        _save_rehearsal(rehearsal_classes, dir, task, limit_memory_size)
        
        buffer_checker(rehearsal=rehearsal_classes)
        
    dist.barrier()
    # All GPUs ready replay dataset
    rehearsal_classes = load_rehearsal(dir=args.Rehearsal_file, task=0, memory=args.Memory)
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
    _, data_loader_train, _, list_CC = Incre_Dataset(0, args, Divided_Classes ) 
    
    # 2. Extra epoch, 모든 이미지들의 Loss를 측정
    rehearsal_classes = train_one_epoch(args, last_task=False, epo=0, model=model, teacher_model=None,
                                        criterion=criterion, data_loader=data_loader_train, optimizer=None,
                                        lr_scheduler=None, device=device, dataset_name="", current_classes=list_CC, 
                                        rehearsal_classes=rehearsal_classes, extra_epoch=extra_epoch)

    # 3. 수집된 Buffer를 특정 파일에 저장
    rehearsal_classes = construct_combined_rehearsal(args=args, task=0, dir=args.Rehearsal_file, rehearsal=rehearsal_classes,
                                                    epoch=0, limit_memory_size=args.Memory, gpu_counts=4, list_CC=list_CC)
    
    print(colored(f"Complete constructing buffer","red", "on_yellow"))
    
    return rehearsal_classes
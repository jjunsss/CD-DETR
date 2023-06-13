import pickle
import copy
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
# This is various experiments function for comparing performance

def contruct_rehearsal(args, losses_dict: dict, targets, rehearsal_dict: List, 
                       current_classes: List[int], least_image: int = 3, limit_image:int = 100) -> Dict:

    # limit_memory = {"class index" : limit number}
    loss_value = 0.0


    for enum, target in enumerate(targets): #! 배치 개수 ex) 4개 
        loss_value = losses_dict["loss_bbox"][enum] + losses_dict["loss_giou"][enum] + losses_dict["loss_labels"][enum]
        if loss_value > 10.0 :
            continue
        # Get the unique labels and the count of each label
        label_tensor = target['labels']
        num_bounding_boxes = label_tensor.shape[0]

        image_id = target['image_id'].item()
        label_tensor_unique = torch.unique(label_tensor)
        label_tensor_unique_list = label_tensor_unique.tolist()
        #if unique tensor composed by Old Dataset, So then pass iteration [Replay constructig shuld not operate in last task training]
        if set(label_tensor_unique_list).issubset(current_classes) is False: 
            continue

        if len(rehearsal_dict.keys()) <  limit_image :
            # when under the buffer 
            rehearsal_dict[image_id] = [loss_value, label_tensor_unique_list, num_bounding_boxes]
        else :
            
            if args.Sampling_mode == "ensure_min":
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
                    rehearsal_dict[image_id] = [loss_value, label_tensor_unique_list, num_bounding_boxes]
                else :
                    changed_available_dict = _change_available_list_mode(mode=args.Sampling_mode, rehearsal_dict=rehearsal_dict,
                                                need_to_include=need_to_include, least_image=least_image, current_classes=current_classes)
                    
                    # all classes dont meet L requirement
                    targeted = _calc_target(rehearsal_classes=changed_available_dict, replace_strategy=args.Sampling_strategy, )
                    rehearsal_dict = _replacment_strategy(args=args, loss_value=loss_value, targeted=targeted, 
                                            rehearsal_classes=rehearsal_dict, label_tensor_unique_list=label_tensor_unique_list,
                                            image_id=image_id, num_bounding_boxes=num_bounding_boxes)
            
            if args.Sampling_mode == "normal":
                changed_available_dict = _change_available_list_mode(mode=args.Sampling_mode, rehearsal_dict=rehearsal_dict,
                            need_to_include=need_to_include, least_image=least_image, current_classes=current_classes)
                targeted = _calc_target(rehearsal_classes=changed_available_dict, replace_strategy=args.Sampling_strategy, )
                rehearsal_dict = _replacment_strategy(args=args, loss_value=loss_value, targeted=targeted, 
                                            rehearsal_classes=rehearsal_dict, label_tensor_unique_list=label_tensor_unique_list,
                                            image_id=image_id, num_bounding_boxes=num_bounding_boxes)

    return rehearsal_dict


def _change_available_list_mode(mode, rehearsal_dict, need_to_include, least_image, current_classes):
    '''
        각 유니크 객체의 개수를 세어 제한하는 것은 그에 맞는 이미지가 존재해야만 모일수도 있기때문에 모두 모을 수 없을 수도 있게되는 불상사가 있다.
        따라서 객체의 개수를 제한하는 것보다는 그에 맞게 비율을 따져서 이미지를 제한하는 방법이 더 와닿을 수 있다.
    '''
    if mode == "normal":
        # no limit and no least images
        changed_available_dict = rehearsal_dict
        
    if mode == "ensure_min":
        image_counts_in_rehearsal = {class_label: sum(class_label in classes for _, (_, classes) in rehearsal_dict.items()) for class_label in current_classes}
        print(f"replay counts : {image_counts_in_rehearsal}")
        
        changed_available_dict = {key: (losses, classes) for key, (losses, classes) in rehearsal_dict.items() if all(image_counts_in_rehearsal[class_label] > least_image for class_label in classes)}
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


#TODO : Change calc each iamage loss and tracking each object loss avg.
def _replacment_strategy(args, loss_value, targeted, rehearsal_classes,
                       label_tensor_unique_list, image_id, ):
    if args.Sampling_strategy == "hierarchical" : 
        if ( targeted[1][0] > loss_value ): #Low buffer construct
            print(colored(f"hierarchical based buffer change strategy", "blue"))
            del rehearsal_classes[targeted[0]]
            rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list]
            return rehearsal_classes
        
    if args.Sampling_strategy  == "high_uniq": # This is same that "hard sampling"
        if ( len(targeted[1][1]) < len(label_tensor_unique_list) ): #Low buffer construct
            print(colored(f"high-unique counts based buffer change strategy", "blue"))
            del rehearsal_classes[targeted[0]]
            rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list]
            return rehearsal_classes
            
    if args.Sampling_strategy  == "random" :
        print(colored(f"random counts based buffer change strategy", "blue"))
        key_to_delete = random.choice(list(rehearsal_classes.keys()))
        del rehearsal_classes[key_to_delete]
        rehearsal_classes[image_id] = [loss_value, label_tensor_unique_list]
        return rehearsal_classes

    print(f"no changed")
    return rehearsal_classes


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

    return sorted_result
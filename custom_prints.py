from typing import Tuple, Dict, List, Optional
import os
import sys
import torch
from datetime import datetime

def write_to_addfile(filename):
    def decorator(func):
        def wrapper(*args, **kwargs):
            original_stdout = sys.stdout
            if os.path.exists("./" + filename.split('/')[1]) is False:
                os.makedirs(filename.split('/')[1],)                
            with open(filename, "a") as f:
                sys.stdout = f
                func(*args, **kwargs)
                sys.stdout = original_stdout
        return wrapper
    return decorator

@write_to_addfile("./check/check_replay_limited.txt")
def check_components(rehearsal_classes: Dict, print_stat: bool=False):
    '''
        1. check each instance usage capacity
        2. print each classes counts
        3. Instance Summary 
        4. Save information
    '''
    if len(rehearsal_classes) == 0:
        raise Exception("No replay classes")
        
    temp_list = [index for _, index in list(rehearsal_classes.values())]
    replay_classes = set()
    for value in temp_list:
        replay_classes.update(value)
    list(replay_classes).sort()
    
    if print_stat == True:
        # check each instance usage capacity
        check_list = [len(list(filter(lambda x: index in x[1], list(rehearsal_classes.values())))) for index in replay_classes]
        
        # To print the current time
        print(f"--------------------------------------------------------\n")
        print("Current Time =", datetime.now())
        for i, c in enumerate(replay_classes):
            print(f"**** class num : {c}, counts : {check_list[i]} ****")
            
def Memory_checker():
    '''
        To check memory capacity
        To check memory cache capacity
    '''
    print(f"*" * 50)
    print(f"allocated Memory : {torch.cuda.memory_allocated()}")
    print(f"max allocated Memory : {torch.cuda.max_memory_allocated()}")
    print(f"*" * 50)
    print(f"cache allocated Memory : {torch.cuda.memory_allocated()}")
    print(f"max allocated Memory : {torch.cuda.max_memory_cached()}")
    print(f"*" * 50)
    
def over_label_checker(check_list:List , check_list2:List = None, check_list3:List = None, check_list4:List = None):
    if check_list2 is None:
        print("Only one overlist: ", check_list)    
    else :
        print("overlist: ", check_list, check_list2, check_list3, check_list4)

@write_to_addfile("./check/loss_check.txt")
def check_losses(epoch, index, losses, epoch_loss, count, training_class, rehearsal=None, dtype=None):
    '''
        protect to division zero Error.
        print (epoch, losses, losses of epoch, training count, training classes now, rehearsal check, CBB format check)
    '''

    try :
        epoch_total_loss = epoch_loss / count
    except ZeroDivisionError:
        epoch_total_loss = 0
            
    if index % 30 == 0: 
        print(f"epoch : {epoch}, losses : {losses:05f}, epoch_total_loss : {epoch_total_loss:05f}, count : {count}")
        if rehearsal is not None:
            print(f"total examplar counts : {len(list(rehearsal.keys()))}")
        if dtype is not None:
            print(f"Now, CBB is {dtype}")    
        
    if index % 30 == 0:
        print(f"current classes is {training_class}")
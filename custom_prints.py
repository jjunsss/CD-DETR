from pympler import summary
from pympler import asizeof
from typing import Tuple, Dict, List, Optional
import os
import sys
import torch

def check_rehearsal_components(rehearsal_classes: Dict, output_dir: str, print_stat: bool=True, save: bool=False):
    '''
        1. check each instance usage capacity
        2. print each classes counts
        3. Instance Summary 
        4. Save information
    '''
    if print_stat == True:
        # check each instance usage capacity
        print("check instance memory capacity")
        for dict_key in rehearsal_classes.keys():
            instances_bytes = asizeof.asizeof(rehearsal_classes[dict_key])
            memory_usage_MB = instances_bytes * 0.00000095367432
            print(f"instance memory capacity {dict_key} : {memory_usage_MB} MB")
            
        # print each clas s's counts
        current_class_sort = sorted(list(rehearsal_classes.keys()))
        for idx in current_class_sort:
            print(f"key : {idx}, counts : {len(rehearsal_classes[idx])}")
            
        print("check rehearsal dictionary memory capacity")
        print(summary.print_(summary.summarize(rehearsal_classes)))
        
    if save == True:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # redirect the output of the print() function to a file
        output_file = os.path.join(output_dir, 'output.txt')
        sys.stdout = open(output_file, 'w')

        # check each instance usage capacity
        print("check instance memory capacity")
        for dict_key in rehearsal_classes.keys():
            instances_bytes = asizeof.asizeof(rehearsal_classes[dict_key])
            memory_usage_MB = instances_bytes * 0.00000095367432
            print(f"instance memory capacity {dict_key} : {memory_usage_MB} MB")

        # print each class's counts
        current_class_sort = sorted(list(rehearsal_classes.keys()))
        for idx in current_class_sort:
            print(f"key : {idx}, counts : {len(rehearsal_classes[idx])}")

        print("check rehearsal dictionary memory capacity")
        print(summary.print_(summary.summarize(rehearsal_classes)))

        sys.stdout.close()

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
    
def over_label_checker(check_list:List , check_list2:List, check_list3:List, check_list4:List):
    print("overlist: ", check_list, check_list2, check_list3, check_list4)
    
    
def check_losses(epoch, index, losses, epoch_loss, count, training_class, rehearsal=None, dtype=None):
    '''
        protect to division zero Error.
        print (epoch, losses, losses of epoch, training count, training classes now, rehearsal check, CBB format check)
    '''
    try :
        epoch_total_loss = epoch_loss / count
    except ZeroDivisionError:
        epoch_total_loss = 0
        
    if index % 10 == 0: 
        print(f"epoch : {epoch}, losses : {losses:05f}, epoch_total_loss : {epoch_total_loss:05f}, count : {count}")
        if rehearsal is not None:
            print(f"total examplar counts : {sum([len(contents) for contents in list(rehearsal.values())])}")
        if dtype is not None:
            print(f"Now, CBB is {dtype}")    
    
    if index % 30 == 0:
        print(f"current classes is {training_class}")
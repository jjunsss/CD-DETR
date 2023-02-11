from pympler import summary
from pympler import asizeof
from typing import Tuple, Dict, List, Optional
import os
import sys
import torch

def check_rehearsal_components(task_number: int, rehearsal_classes: Dict, current_classes: List, output_dir: str, print_stat: bool=True, save: bool=False):
    '''
        1. check each instance usage capacity
        2. print each classes counts
        3. Instance Summary 
        4. Save information
    '''
    if print_stat == True:
        # check each instance usage capacity
        check_list = [len(list(filter(lambda x: current in x, list(rehearsal_classes.values())[1]))) for current in current_classes]
        for current, check in zip(current_classes, check_list):
            print(f"current classes : {current} memory size : {check}")
        
    if save == True:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # redirect the output of the print() function to a file
        output_file = os.path.join(output_dir, (task_number+1) +'_output.txt')
        sys.stdout = open(output_file, 'w')
        for current, check in zip(current_classes, check_list):
            print(f"current classes : {current} memory size : {check}")
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
    
def over_label_checker(check_list:List , check_list2:List = [], check_list3:List = [], check_list4:List = []):
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
            print(f"total examplar counts : {len(list(rehearsal.keys()))}")
        if dtype is not None:
            print(f"Now, CBB is {dtype}")    
        
    if index % 30 == 0:
        print(f"current classes is {training_class}")
from pympler import summary
from pympler import asizeof
from typing import Tuple, Dict, List, Optional
import os
import sys

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
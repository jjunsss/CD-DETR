"""
Train and eval functions used in main.py
"""
import math
import numpy as np
import os
import sys
import random
from typing import Iterable
from tqdm import tqdm
import torch.distributed as dist
import pickle
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher import data_prefetcher
from torch.utils.data import DataLoader
from pympler import summary
from pympler import asizeof
from custom_utils import *
from custom_prints import check_losses
import os
import time
from pycocotools.coco import COCO
from typing import Tuple, Collection, Dict, List
from datasets import build_dataset, get_coco_api_from_dataset
import torch
import util.misc as utils
import numpy as np
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
from GPUtil import showUtilization as gpu_usage
from torch.cuda.amp import autocast, GradScaler

@decompose
def decompose_dataset(no_use_count: int, samples: utils.NestedTensor, targets: Dict, origin_samples: utils.NestedTensor, 
                      origin_targets: Dict, used_number: List) -> Tuple[int, List, utils.NestedTensor, Dict, utils.NestedTensor, Dict, List]:
    batch_size = len(targets)
    return (batch_size, no_use_count, samples, targets, origin_samples, origin_targets, used_number)


def Original_training(args, epo, idx, count, sum_loss, samples, targets, origin_sam, origin_tar, 
                      model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,
                      rehearsal_classes, train_check, current_classes): 
    '''
        Only Training Original Data or (Transformed image, Transformed Target).
        This is not Mosaic Data.
    '''
    torch.cuda.empty_cache()
    if train_check :
        model.train()
    else:
        model.eval()
    
    device = torch.device("cuda")
    ex_device = torch.device("cpu")
    criterion.train()
    model.to(device)
    scaler = GradScaler()
    
    with autocast():
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        #with autocast(False):
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        losses_value = losses.item()
               
        with torch.no_grad():
            if train_check == True and args.Rehearsal == True: #* I will use this code line. No delete.
                targets = [{k: v.to(ex_device) for k, v in t.items()} for t in targets]
                rehearsal_classes = contruct_rehearsal(losses_value=losses_value, lower_limit=0.1, upper_limit=100, 
                                                    targets=targets,
                                                    rehearsal_classes=rehearsal_classes, 
                                                    Current_Classes=current_classes, 
                                                    Rehearsal_Memory=args.Memory)
                
            del samples, targets
            loss_dict_reduced = utils.reduce_dict(loss_dict, train_check)
            if loss_dict_reduced == False:
                losses_reduced_scaled = 0
                loss_dict_reduced_scaled = 0
                
            if loss_dict_reduced != False:
                count += 1
                loss_dict_reduced_scaled = {v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
                losses_reduced_scaled = sum(loss_dict_reduced_scaled)
        
            sum_loss += losses_reduced_scaled
            if utils.is_main_process(): #sum_loss가 GPU의 개수에 맞춰서 더해주고 있으니,
                check_losses(epo, idx, losses_reduced_scaled, sum_loss, count, current_classes, rehearsal_classes)
        if args.clip_max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), args.clip_max_norm)
        
        scaler.scale(losses).backward()
        

    optimizer.zero_grad()
    scaler.step(optimizer)
    scaler.update()
    dist.barrier()
    
    del origin_sam, origin_tar, losses_reduced_scaled, loss_dict_reduced_scaled, loss_dict, outputs, loss_dict_reduced, losses
    return rehearsal_classes, sum_loss, count

def Mosaic_training(args, epo, idx, count, sum_loss, samples, targets,
                    model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, current_classes, data_type):
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    ex_device = torch.device("cpu")
    
    model.train()
    criterion.train()
    scaler = GradScaler()
    with autocast():
        
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        count += 1
        del samples, targets
        
        loss_dict_reduced = utils.reduce_dict(loss_dict, True)
            
        loss_dict_reduced_scaled = {v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled)
    
        sum_loss += losses_reduced_scaled
        if args.clip_max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), args.clip_max_norm) #* inplace format 

        if utils.is_main_process(): #sum_loss가 GPU의 개수에 맞춰서 더해주고 있으니,
            check_losses(epo, idx, losses_reduced_scaled, sum_loss, count, current_classes, None, data_type)
            if idx % 10 == 0:
                print(f"current classes is {current_classes}")
        scaler.scale(losses).backward()
    optimizer.zero_grad()
    scaler.step(optimizer)
    scaler.update()
    dist.barrier()
    del loss_dict, outputs, losses
    
    return count, sum_loss
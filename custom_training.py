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
    device = torch.device("cuda")
    ex_device = torch.device("cpu")
    model.train()
    criterion.train()
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    outputs = model(samples)
    loss_dict = criterion(outputs, targets)
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    losses_value = losses.item()
    
    with torch.no_grad():
        if train_check == True and args.Rehearsal == True: #* I will use this code line. No delete.
            targets = [{k: v.to(ex_device) for k, v in t.items()} for t in targets]
            rehearsal_classes = contruct_rehearsal(losses_value=losses_value, lower_limit=0.1, upper_limit=100, samples=samples, targets=targets, 
                                    origin_samples=origin_sam, origin_targets=origin_tar, rehearsal_classes=rehearsal_classes, Current_Classes=current_classes, Rehearsal_Memory=args.Memory)

    loss_dict_reduced = utils.reduce_dict(loss_dict, train_check)
    if loss_dict_reduced == False:
                losses_reduced_scaled = 0
                loss_dict_reduced_scaled = 0
                losses = torch.tensor(0, device=torch.device("cuda"), requires_grad=True, dtype=torch.float32)
                
    if loss_dict_reduced != False:
        count += 1
        loss_dict_reduced_scaled = {k: v.item() * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        
    sum_loss += losses_reduced_scaled
    
    if utils.is_main_process(): #sum_loss가 GPU의 개수에 맞춰서 더해주고 있으니,
        print(f"epoch : {epo}, losses : {losses_reduced_scaled:05f}, epoch_total_loss : {(sum_loss / count):05f}, count : {count}")
        print(f"total examplar counts : {sum([len(contents) for contents in list(rehearsal_classes.values())])}")
        if idx % 10 == 0:
            print(f"current classes is {current_classes}")

    if not math.isfinite(losses_reduced_scaled):
        print("Loss is {}, Dagerous training".format(losses_reduced_scaled))
        print(f"all reduce GPU Params : {loss_dict_reduced}")
        
    optimizer.zero_grad()
    losses.backward()
    
    if args.clip_max_norm > 0:
        grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
    else:
        grad_total_norm = utils.get_total_grad_norm(model.parameters(), args.clip_max_norm)
    optimizer.step()

    del samples, targets, origin_sam, origin_tar, losses_reduced_scaled, loss_dict_reduced_scaled, loss_dict, outputs, loss_dict_reduced #무조건 마지막까지 함께훈련이 되도록 유도
    torch.cuda.empty_cache()
    
    return rehearsal_classes, sum_loss, count

def Mosaic_training(args, epo, idx, count, sum_loss, samples, targets,
                    model: torch.nn.Module, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, current_classes):

    device = torch.device("cuda")
    ex_device = torch.device("cpu")
    model.train()
    criterion.train()
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    outputs = model(samples)
    loss_dict = criterion(outputs, targets)
    weight_dict = criterion.weight_dict
    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    losses_value = losses.item()
    
    loss_dict_reduced = utils.reduce_dict(loss_dict, True)
    
    if loss_dict_reduced != False:
        count += 1
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
    sum_loss += losses_reduced_scaled
    
    if utils.is_main_process(): #sum_loss가 GPU의 개수에 맞춰서 더해주고 있으니,
        print(f"epoch : {epo}, losses : {losses_reduced_scaled:05f}, epoch_total_loss : {(sum_loss / count):05f}, count : {count}")
        if idx % 10 == 0:
            print(f"current classes is {current_classes}")

    if not math.isfinite(losses_reduced_scaled):
        print("Loss is {}, Dagerous training".format(losses_reduced_scaled))
        print(f"all reduce GPU Params : {loss_dict_reduced}")
        
    optimizer.zero_grad()
    losses.backward()
    
    if args.clip_max_norm > 0:
        grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
    else:
        grad_total_norm = utils.get_total_grad_norm(model.parameters(), args.clip_max_norm) #* inplace format 
    optimizer.step()

    del samples, targets , losses_reduced_scaled, loss_dict_reduced_scaled, loss_dict, outputs, loss_dict_reduced
    torch.cuda.empty_cache()
    
    return count, sum_loss
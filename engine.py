# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

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

@decompose
def decompose_dataset(no_use_count: int, samples: utils.NestedTensor, targets: Dict, origin_samples: utils.NestedTensor, origin_targets: Dict, 
                      used_number: List) -> Tuple[int, List, utils.NestedTensor, Dict, utils.NestedTensor, Dict, List]:
    batch_size = len(targets)
    return (batch_size, no_use_count, samples, targets, origin_samples, origin_targets, used_number)

def train_one_epoch(args, epo, model: torch.nn.Module,criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, memory: int = 300, max_norm: float = 0,
                    current_classes: List = [], rehearsal_classes: Dict = {}, limited_counts: int = 0):
    
    label_dict = {}
    model.train()
    criterion.train()
    ex_device = torch.device("cpu")
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))

    prefetcher = data_prefetcher(data_loader, device, prefetch=True)
    samples, targets, origin_samples, origin_targets = prefetcher.next() 
    
    sum_loss = 0.0
    set_tm = time.time()
    count = 0
    for idx in tqdm(range(len(data_loader))): #targets 
        train_check = True
        samples = samples.to(ex_device)
        origin_samples = origin_samples.to(ex_device)
        targets = [{k: v.to(ex_device) for k, v in t.items()} for t in targets]
        
        #TODO : one samples no over / one samples over solve this ! 
        if idx < 100000:
            with torch.no_grad():
                no_use, yes_use, label_dict = check_class(True, targets, label_dict, CL_Limited=limited_counts)
                samples, targets, _, _ , train_check = decompose_dataset(no_use_count=len(no_use), samples= samples, targets = targets, origin_samples=origin_samples, origin_targets= origin_targets ,used_number= yes_use)

            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            outputs = model(samples)
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            losses_value = losses.item()
            
            #! 여기서 리허설을 위한 데이터를 모집해야 함. construct rehearsal dataset
            with torch.no_grad():
                if train_check == True and args.Rehearsal == True:
                    #origin_samples = origin_samples.to(ex_device)
                    targets = [{k: v.to(ex_device) for k, v in t.items()} for t in targets]
                    #origin_targets = [{k: v.to(ex_device) for k, v in t.items()} for t in origin_targets]
                    rehearsal_classes = contruct_rehearsal(losses_value=losses_value, lower_limit=0.1, upper_limit=0.1,
                                        targets=targets, origin_samples=origin_samples, origin_targets=origin_targets, rehearsal_classes=rehearsal_classes, Current_Classes=current_classes, Rehearsal_Memory=memory)

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = utils.reduce_dict(loss_dict, train_check)
            if loss_dict_reduced == False:
                samples, targets, origin_samples, origin_targets = prefetcher.next() 
                print(f'Total GPU not working... so passed \n')
                continue
            
            count += 1
            loss_dict_reduced_scaled = {k: v.item() * weight_dict[k]
                                        for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            #loss_value = losses_reduced_scaled.item()
            sum_loss += losses_reduced_scaled
            
            if utils.is_main_process(): #sum_loss가 GPU의 개수에 맞춰서 더해주고 있으니,
                print(f"epoch : {epo}, losses : {losses_reduced_scaled:05f}, epoch_total_loss : {(sum_loss / count):05f}, count : {count}")
                print(f"total examplar counts : {sum([len(contents) for contents in list(rehearsal_classes.values())])}")
                if idx % 10 == 0:
                    print(f"current classes is {current_classes}")

            if not math.isfinite(losses_reduced_scaled):
                print("Loss is {}, stopping training".format(losses_reduced_scaled))
                print(loss_dict_reduced)
                samples, targets, origin_samples, origin_targets = prefetcher.next() 
                continue
            
            optimizer.zero_grad()
            losses.backward()
            
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
            optimizer.step()
            # print(f"allocated Memory : {torch.cuda.memory_allocated()}")
            # print(f"max allocated Memory : {torch.cuda.max_memory_allocated()}")
            # print(f"cache allocated Memory : {torch.cuda.memory_allocated()}")
            # print(f"max allocated Memory : {torch.cuda.max_memory_cached()}")
            del samples, targets, origin_samples, origin_targets
            
            if torch.cuda.memory_allocated() > torch.cuda.max_memory_reserved() * 0.99:
                torch.cuda.empty_cache()
                
            samples, targets, origin_samples, origin_targets = prefetcher.next()
        else:
            break
        
    if utils.is_main_process():
        print("all loss: ", loss_dict_reduced)
        print("Total Time : ", time.time() - set_tm)

    return rehearsal_classes

@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir):
    model.eval()
    criterion.eval()
    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
        print('output', outputs)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
        if coco_evaluator is not None:
            coco_evaluator.update(res)

        if panoptic_evaluator is not None:
            res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
            for i, target in enumerate(targets):
                image_id = target["image_id"].item()
                file_name = f"{image_id:012d}.png"
                res_pano[i]["image_id"] = image_id
                res_pano[i]["file_name"] = file_name

            panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    #metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()
    if panoptic_evaluator is not None:
        panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()
    panoptic_res = None
    if panoptic_evaluator is not None:
        panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    if panoptic_res is not None:
        stats['PQ_all'] = panoptic_res["All"]
        stats['PQ_th'] = panoptic_res["Things"]
        stats['PQ_st'] = panoptic_res["Stuff"]
    return stats, coco_evaluator

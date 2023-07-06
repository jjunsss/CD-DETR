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
from typing import Iterable
from tqdm import tqdm
import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.data_prefetcher import data_prefetcher
import os
import time
from typing import Tuple, Collection, Dict, List
from datasets import build_dataset, get_coco_api_from_dataset
import torch
import util.misc as utils
import numpy as np
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm
from custom_training import *
from custom_prints import check_components

from models import inference_model

@decompose
def decompose_dataset(no_use_count: int, samples: utils.NestedTensor, targets: Dict, origin_samples: utils.NestedTensor, origin_targets: Dict, 
                      used_number: List) -> Tuple[int, List, utils.NestedTensor, Dict, utils.NestedTensor, Dict, List]:
    batch_size = len(targets)
    return (batch_size, no_use_count, samples, targets, origin_samples, origin_targets, used_number)


def _extra_epoch_for_replay(args, dataset_name: str, data_loader: Iterable, model: torch.nn.Module, criterion: torch.nn.Module,
                                 device: torch.device, rehearsal_classes, current_classes):

    '''
        Run additional epoch to collect replay buffer. 
        1. initialize prefeter, (icarl) feature extractor and prototype.
        2. run rehearsal training.
        3. (icarl) detach values in rehearsal_classes.
    '''

    prefetcher = create_prefetcher(dataset_name, data_loader, device, args)
    if args.Sampling_strategy == "icarl":
        fe = icarl_feature_extractor_setup(args, model)
        proto = icarl_prototype_setup(args, fe, device, current_classes)

    with torch.no_grad():
        for idx in tqdm(range(len(data_loader)), disable=not utils.is_main_process()): #targets
            samples, targets, _, _ = prefetcher.next()
                
            # extra training을 통해서 replay 데이터를 수집하도록 설정
            if args.Sampling_strategy == "icarl":
                rehearsal_classes = icarl_rehearsal_training(args, samples, targets, fe, proto, device,
                                                   rehearsal_classes, current_classes)
            else:
                rehearsal_classes = rehearsal_training(args, samples, targets, model, criterion, 
                                                   rehearsal_classes, current_classes)
            if idx % 100 == 0:
                torch.cuda.empty_cache()
            
            # 정완 디버그
            if args.debug:
                if idx == args.num_debug_dataset:
                    break

        if args.Sampling_strategy == "icarl":
            ''' 
                rehearsal_classes : [feature_sum, [[image_ids, difference] ...]] 
            '''
            for key, val in rehearsal_classes.items():
                val[0] = val[0].detach().cpu()

    return rehearsal_classes


def create_prefetcher(dataset_name: str, data_loader: Iterable, device: torch.device, args: any) \
        -> data_prefetcher:
    if dataset_name == "Original":    
        return data_prefetcher(data_loader, device, prefetch=True, Mosaic=False)
    elif dataset_name == "AugReplay":
        return data_prefetcher(data_loader, device, prefetch=True, Mosaic=True, Continual_Batch=args.Continual_Batch_size)
    else:
        return data_prefetcher(data_loader, device, prefetch=True, Mosaic=False)


def train_one_epoch(args, last_task, epo, model: torch.nn.Module, teacher_model, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler: ContinualStepLR,
                    device: torch.device, dataset_name: str,  
                    current_classes: List = [], rehearsal_classes: Dict = {},
                    extra_epoch: bool = False):
    ex_device = torch.device("cpu")
    prefetcher = create_prefetcher(dataset_name, data_loader, device, args)
    
    set_tm = time.time()
    sum_loss = 0.0
    count = 0
    for idx in tqdm(range(len(data_loader)), disable=not utils.is_main_process()): #targets
        if idx % 100 == 0:
            torch.cuda.empty_cache()
        samples, targets, _, _ = prefetcher.next()

        train_check = True
        samples = samples.to(ex_device)
        targets = [{k: v.to(ex_device) for k, v in t.items()} for t in targets]
        #Stage 1 -> T1에 대한 모든 훈련
        #Stage 2 -> T2에 대한 모든 훈련, AugReplay 사용하지 않을 때에는 일반적인 Replay 전략과 동일한 형태로 훈련을 수행
        sum_loss, count = Original_training(args, last_task, epo, idx, count, sum_loss, samples, targets,  
                                                               model, teacher_model, criterion, optimizer,
                                                               rehearsal_classes, train_check, current_classes)

        if dataset_name == "AugReplay" and args.Rehearsal and last_task:
            # this process only replay strategy, AugReplay is same to "Circular Training"
            samples, targets, _, _ = prefetcher.next() #* Different
            # lr_scheduler.replay_step(idx)
            count, sum_loss = Circular_training(args, last_task, epo, idx, count, sum_loss, samples, targets,
                                                model, teacher_model, criterion, optimizer,
                                                current_classes)
        del samples, targets, train_check

        # 정완 디버그
        if args.debug:
            if count == args.num_debug_dataset:
                break
        
    if utils.is_main_process():
        print("Total Time : ", time.time() - set_tm)


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, DIR, args) :
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    
    #FIXME: check your cocoEvaluator function for writing the results (I'll give you code that changed)
    coco_evaluator = CocoEvaluator(base_ds, iou_types, DIR)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
    
    cnt = 0 # for debug
        
    for samples, targets, _, _ in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = inference_model(args, model, samples, targets, eval=True)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        
        
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict, True)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                    for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = postprocessors['bbox'](outputs, orig_target_sizes, args.model_name)
        #print(results)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        
        gt = outputs[0]['gt'] if args.model_name == 'dn_detr' else outputs['gt']
        
        # cocoeval에서 gt와 dt를 맞추어주기 위함
        if gt is not None :
            for r in res.values():
                labels = r['labels'].cpu().numpy()
                r['labels'] = torch.tensor([
                    gt[tgt_id-1] for tgt_id in labels
                ], dtype=torch.int64).cuda()
        
        if coco_evaluator is not None:
            coco_evaluator.update(res)
            
        if args.debug:
            cnt += 1
            if cnt == args.num_debug_dataset:
                break

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    if coco_evaluator is not None:
        if 'bbox' in postprocessors.keys():
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in postprocessors.keys():
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    return stats, coco_evaluator

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

@decompose
def decompose_dataset(no_use_count: int, samples: utils.NestedTensor, targets: Dict, origin_samples: utils.NestedTensor, origin_targets: Dict, 
                      used_number: List) -> Tuple[int, List, utils.NestedTensor, Dict, utils.NestedTensor, Dict, List]:
    batch_size = len(targets)
    return (batch_size, no_use_count, samples, targets, origin_samples, origin_targets, used_number)


def train_one_epoch(args, last_task, epo, model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer, device: torch.device, 
                    MosaicBatch: Boolean, current_classes: List = [], 
                    rehearsal_data: Dict = {}, prototypes: Dict = {}, data_loader_icarl=None):

    if MosaicBatch == False:    
        prefetcher = data_prefetcher(data_loader, device, prefetch=True, Mosaic=False)
    else:
        prefetcher = data_prefetcher(data_loader, device, prefetch=True, Mosaic=True, Continual_Batch=args.Continual_Batch_size)


    set_tm = time.time()
    sum_loss = 0.0
    if args.only_icarl and not last_task:
        pass
    else:
        for idx in tqdm(range(len(data_loader))): #targets 
            train_check = True
            trainable = True
            samples, targets, origin_samples, origin_targets = prefetcher.next()
            del origin_samples, origin_targets
            # with torch.no_grad():
            #     torch.cuda.empty_cache()
            #     samples, targets, origin_samples, origin_targets = prefetcher.next()
            #     #print(f"target value: {targets}")
                
            #     train_check = True
            #     trainable = True
            #     samples = samples.to(ex_device)
            #     targets = [{k: v.to(ex_device) for k, v in t.items()} for t in targets
                #TODO : one samples no over / one samples over solve this ! 
                # #* because MosaicAugmentation Data has not original data
                # no_use, yes_use, label_dict = check_class(args.verbose, args.LG , targets, label_dict, current_classes, CL_Limited=args.CL_Limited) #! Original에 한해서만 Limited Training(현재 Task의 데이터에 대해서만 가정)
                # samples, targets, origin_samples, origin_targets, train_check = decompose_dataset(no_use_count=len(no_use), samples= samples, targets = targets, origin_samples=origin_samples, origin_targets= origin_targets ,used_number= yes_use)
                # trainable = check_training_gpu(train_check=train_check)
                # if trainable == False :
                #     del samples, targets, origin_samples, origin_targets, train_check
                #     torch.cuda.empty_cache()
                #     early_stopping_count += 1
                #     if MosaicBatch == True :
                #         _, _, _, _ = prefetcher.next(new = True)        
                #     continue

            # if trainable == True:
            # #contruct rehearsal buffer in main training
            # rehearsal_data, sum_loss, count = Original_training(args, last_task, epo, idx, count, 
            #                                         sum_loss, samples, targets, origin_samples, origin_targets, 
            #                                         model, criterion, optimizer, rehearsal_data, train_check, 
            #                                         current_classes)
            model.train()
            criterion.train()
            # model.to(device)        
            # samples = samples.to(device)
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            with autocast(True):
                outputs = model(samples)
                if args.Fake_Query == True:
                    targets = normal_query_selc_to_target(outputs, targets, current_classes)
                loss_dict = criterion(outputs, targets)
                weight_dict = criterion.weight_dict
                losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            del samples, targets
                    
            # with torch.no_grad():
            #     if train_check and args.Rehearsal and last_task == False: #* I will use this code line. No delete.
            #         targets = [{k: v.to(ex_device) for k, v in t.items()} for t in targets]
            #         rehearsal_classes = contruct_rehearsal(losses_value=losses_value, lower_limit=0.1, upper_limit=10, 
            #                                             targets=targets,
            #                                             rehearsal_classes=rehearsal_classes, 
            #                                             Current_Classes=current_classes, 
            #                                             Rehearsal_Memory=args.Memory)
                    
            #     del samples, targets

            loss_dict_reduced = utils.reduce_dict(loss_dict, train_check)
            loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
            losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
            losses_value = losses_reduced_scaled
            sum_loss += losses_value

            if utils.is_main_process(): #sum_loss가 GPU의 개수에 맞춰서 더해주고 있으니,
                check_losses(epo, idx, losses_reduced_scaled, sum_loss, idx, current_classes, rehearsal_data)
                if idx % 50 == 0:
                    print(f"Epoch[{epo}]: \t loss: {losses_value} \t total_loss: {sum_loss/(idx+1)}")
                
            # optimizer = control_lr_backbone(args, optimizer=optimizer, frozen=False)
            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            optimizer.step()
            
            del trainable, train_check, losses_reduced_scaled, loss_dict_reduced_scaled, loss_dict, outputs, loss_dict_reduced, losses
            torch.cuda.empty_cache()

    # iCaRL data setup
    if not last_task:
        prefetcher_prototype = data_prefetcher(data_loader_icarl, device, prefetch=True, Mosaic=False)
        prefetcher_exemplar = data_prefetcher(data_loader_icarl, device, prefetch=True, Mosaic=False)
        # icarl copy feature extracter
        feature_extracter = copy.deepcopy(model.module.backbone)
        for n, p in feature_extracter.named_parameters():
            p.requires_grad = False
        feature_extracter.eval()
        # construct prototype (1)- generate prototype
        with torch.no_grad():
            _cnt = 0
            for idx in tqdm(range(len(data_loader_icarl)), desc='Construct_prototype:'):
                _, _, origin_sample, origin_target = prefetcher_prototype.next()
                origin_sample = origin_sample.to(device)
                origin_target = origin_target[0]
                feature, pos = feature_extracter(origin_sample)
                feature_0 = feature[0].tensors.squeeze(dim=0)

                label_tensor = origin_target['labels']
                label_tensor_unique = torch.unique(label_tensor)
                label_list_unique = label_tensor_unique.tolist()
                for label in label_list_unique:
                    try:
                        if not prototypes[label]:
                            prototypes[label] = [feature_0, 1]
                        else:
                            prototypes[label][0] += feature_0
                            prototypes[label][1] += 1
                    except KeyError:
                        print('The label isnot in this Task!!')
                _cnt += 1
                if args.debug and _cnt >= 100:
                    print('BREAKBREAKBREAKBREAKBREAK')
                    break
        
        # construct prototype (2) - gather exemplar 
            _cnt = 0
            for idx in tqdm(range(len(data_loader_icarl)), desc='Construct_exemplar:'):
                _, _, origin_sample, origin_target = prefetcher_exemplar.next()
                origin_target = origin_target[0]
                origin_sample = origin_sample.to(device)
                feature, pos = feature_extracter(origin_sample)
                feature_0 = feature[0].tensors.squeeze(dim=0).cpu()

                label_tensor = origin_target['labels']
                label_tensor_unique = torch.unique(label_tensor)
                label_list_unique = label_tensor_unique.tolist()
                for label in label_list_unique:
                    try:
                        class_mean = (prototypes[label][0] / prototypes[label][1]).cpu()
                    except KeyError:
                        print(f'label: {label} donot in prototype: {prototypes.keys()}')
                        continue

                    if rehearsal_data[label]:
                        exemplar_mean = (rehearsal_data[label][0].cpu() + feature_0) / (len(rehearsal_data[label]) + 1)
                        difference = torch.mean(torch.sqrt(torch.sum((class_mean - exemplar_mean)**2, axis=1))).item()

                        rehearsal_data[label][0] = rehearsal_data[label][0].cpu()                    
                        rehearsal_data[label][0]+= feature_0
                        rehearsal_data[label][1].append([origin_target['image_id'].item(), difference])
                    else:
                        difference = torch.argmin(torch.sqrt(torch.sum((class_mean - feature_0)**2, axis=0))).item() # argmin is true????
                        rehearsal_data[label] = [feature_0, [[origin_target['image_id'].item(), difference], ]]
                    
                    rehearsal_data[label][1].sort(key=lambda x: x[1]) # sort with difference
                _cnt += 1
                if args.debug and _cnt >=100:
                    print('BREAKBREAKBREAKBREAKBREAK')
                    break

        # construct rehearsal (3) - reduce exemplar set
            for label, data in tqdm(rehearsal_data.items(), desc='Reduce_exemplar:'):
                try:
                    data[1] = data[1][:args.Memory]
                except:
                    continue

    #for checking limited Classes Learning
    if utils.is_main_process():
        print("Total Time : ", time.time() - set_tm)
    return rehearsal_data



@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, DIR) :
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    coco_evaluator = CocoEvaluator(base_ds, iou_types, DIR)
    # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]
        
    for samples, targets, _, _ in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(samples)
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
        results = postprocessors['bbox'](outputs, orig_target_sizes)
        #print(results)
        if 'segm' in postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        if coco_evaluator is not None:
            coco_evaluator.update(res)

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

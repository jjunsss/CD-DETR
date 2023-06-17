"""
Train and eval functions used in main.py
"""
import math
import numpy as np
import os
import sys
import random
import torch.distributed as dist
import torch
import util.misc as utils
from custom_utils import *
from custom_buffer_manager import *
from custom_prints import check_losses
import os
from typing import Tuple, Collection, Dict, List
import torch
import util.misc as utils
import numpy as np
from typing import Tuple, Dict, List, Optional

from torch.cuda.amp import autocast
from custom_fake_target import normal_query_selc_to_target, only_oldset_mosaic_query_selc_to_target
from models import inference_model

@decompose
def decompose_dataset(no_use_count: int, samples: utils.NestedTensor, targets: Dict, origin_samples: utils.NestedTensor, 
                      origin_targets: Dict, used_number: List) -> Tuple[int, List, utils.NestedTensor, Dict, utils.NestedTensor, Dict, List]:
    batch_size = len(targets)
    return (batch_size, no_use_count, samples, targets, origin_samples, origin_targets, used_number)

def Original_training(args, last_task, epo, idx, count, sum_loss, samples, targets, 
                      model: torch.nn.Module, teacher_model, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,  
                      rehearsal_classes, train_check, current_classes): 

    device = torch.device("cuda")
    ex_device = torch.device("cpu")
    count, sum_loss = _common_training(args, epo, idx, last_task, count, sum_loss, 
                                        samples, targets, model, optimizer,
                                        teacher_model, criterion, device, ex_device, current_classes, "original")

    del samples, targets

    return sum_loss, count

def Circular_training(args, last_task, epo, idx, count, sum_loss, samples, targets, 
                    model: torch.nn.Module, teacher_model, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                    current_classes): 
    
    device = torch.device("cuda")
    ex_device = torch.device("cpu")
    count, sum_loss = _common_training(args, epo, idx, last_task, count, sum_loss, 
                                        samples, targets, model, optimizer,
                                        teacher_model, criterion, device, ex_device, current_classes, "circular")

    del samples, targets  
    return count, sum_loss



def _common_training(args, epo, idx, last_task, count, sum_loss, 
                     samples, targets, model: torch.nn.Module, optimizer:torch.optim.Optimizer,
                     teacher_model, criterion: torch.nn.Module, device, ex_device, current_classes, t_type=None):
    model.train()
    criterion.train()

    samples, targets = _process_samples_and_targets(samples, targets, device)

    with autocast(False):
        if last_task and args.Distill:
            teacher_model.eval()
            teacher_model.to(device)
            with torch.no_grad():
                t_encoder = []
                hook = teacher_model.transformer.encoder.layers[-1].self_attn.attention_weights.register_forward_hook(
                    lambda module, input, output: t_encoder.append(output)
                )

                _ = teacher_model(samples)
                teacher_model.to(ex_device)
                hook.remove()
                pre_encodre = t_encoder[0]

            s_encoder = []
            hook = model.module.transformer.encoder.layers[-1].self_attn.attention_weights.register_forward_hook(
                lambda module, input, output: s_encoder.append(output)
            )
            outputs = inference_model(args, model, samples, targets)
            hook.remove()
            new_encoder = s_encoder[0]

            location_loss = torch.nn.functional.mse_loss(new_encoder.detach(), pre_encodre)
            del t_encoder, s_encoder, new_encoder, pre_encodre

        else:              
            outputs = inference_model(args, model, samples, targets)

        if args.Fake_Query:
            targets = normal_query_selc_to_target(outputs, targets, current_classes)  # Adjust this line as necessary

        loss_dict = criterion(outputs, targets)
            
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        losses_value = losses.item()
        
    if last_task and args.Distill:  
        losses = losses + location_loss * 0.5  # alpha
        
    loss_dict_reduced = utils.reduce_dict(loss_dict, train_check=True)
    if loss_dict_reduced != False:
        count += 1
        loss_dict_reduced_scaled = {v * weight_dict[k] for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled)

        sum_loss += losses_reduced_scaled
        if utils.is_main_process(): #sum_loss가 GPU의 개수에 맞춰서 더해주고 있으니,
            check_losses(epo, idx, losses_reduced_scaled, sum_loss, count, current_classes, None)
            print(f" {t_type} \t {{ epoch : {epo} \t Loss : {losses_value:.4f} \t Total Loss : {sum_loss/count:.4f} }}")
    
    optimizer.zero_grad()
    losses.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
    optimizer.step()

    return count, sum_loss


@torch.no_grad()
def icarl_rehearsal_training(args, samples, targets, model: torch.nn.Module, criterion: torch.nn.Module, 
                       rehearsal_classes, current_classes):

    from tqdm import tqdm
    from copy import deepcopy

    def feature_extractor_setup(model):
        feature_extractor = deepcopy(model.module.backbone)
        for n, p in feature_extractor.named_parameters():
            p.requires_grad = False

        return feature_extractor

    def prototype_setup(feature_extractor, data_loader, device, args):
   
        feature_extractor.eval()
        prototypes = {}
        with torch.no_grad():
            for sample, target in tqdm(data_loader, desc='Construct_prototype:'):
                sample = sample.to(device)
                target = target[0]
                feature, _ = feature_extractor(sample)
                feature_0 = feature[0].tensors.squeeze(dim=0)

                label_tensor = target['labels']
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

        return prototypes

    fe = feature_extractor_setup(model)
    fe.eval()
    proto = prototype_setup()
    feature, pos = fe(samples)
    feature_0 = feature[0].tensors.squeeze(dim=0).cpu()

    label_tensor = targets['labels']
    label_tensor_unique = torch.unique(label_tensor)
    label_list_unique = label_tensor_unique.tolist()
    for label in label_list_unique:
        try:
            class_mean = (proto[label][0] / proto[label][1]).cpu()
        except KeyError:
            print(f'label: {label} donot in prototype: {proto.keys()}')
            continue

        if rehearsal_classes[label]:
            exemplar_mean = (rehearsal_classes[label][0].cpu() + feature_0) / (len(rehearsal_classes[label]) + 1)
            difference = torch.mean(torch.sqrt(torch.sum((class_mean - exemplar_mean)**2, axis=1))).item()

            rehearsal_classes[label][0] = rehearsal_classes[label][0].cpu()                    
            rehearsal_classes[label][0]+= feature_0
            rehearsal_classes[label][1].append([targets['image_id'].item(), difference])
        else:
            difference = torch.argmin(torch.sqrt(torch.sum((class_mean - feature_0)**2, axis=0))).item() # argmin is true????
            rehearsal_classes[label] = [feature_0, [[targets['image_id'].item(), difference], ]]
        
        rehearsal_classes[label][1].sort(key=lambda x: x[1]) # sort with difference

    # construct rehearsal (3) - reduce exemplar set
    for label, data in tqdm(rehearsal_classes.items(), desc='Reduce_exemplar:'):
        try:
            data[1] = data[1][:args.Memory]
        except:
            continue

        return rehearsal_classes


def rehearsal_training(args, samples, targets, model: torch.nn.Module, criterion: torch.nn.Module, 
                       rehearsal_classes, current_classes):
    '''
        replay를 위한 데이터를 수집 시에 모델은 영향을 받지 않도록 설정
    '''
    model.eval()
    criterion.eval()
    device = torch.device("cuda")
    ex_device = torch.device("cpu")
    model.to(device)
    samples, targets = _process_samples_and_targets(samples, targets, device)
    for_replay = True

    outputs = inference_model(args, model, samples, targets, eval=True)
    # TODO : new input to model. plz change dn-detr model input (self.buffer_construct_loss)
    _ = criterion(outputs, targets, buffer_construct_loss=True)
    
        
    with torch.no_grad():
        batch_loss_dict = {}
        
        # Transform tensor to scarlar value for rehearsal step
        # TODO : Undecided, but whether to input Term to control Loss factors
        batch_loss_dict["loss_bbox"] = [loss.item() for loss in criterion.losses_for_replay["loss_bbox"]]
        batch_loss_dict["loss_giou"] = [loss.item() for loss in criterion.losses_for_replay["loss_giou"]]
        batch_loss_dict["loss_labels"] = [loss.item() for loss in criterion.losses_for_replay["loss_labels"]]
    
        targets = [{k: v.to(ex_device) for k, v in t.items()} for t in targets]
        rehearsal_classes = contruct_rehearsal(args, losses_dict=batch_loss_dict, targets=targets,
                                                rehearsal_dict=rehearsal_classes, 
                                                current_classes=current_classes,
                                                least_image=args.least_image,
                                                limit_image=args.limit_image)
    if utils.get_world_size() > 1:    
        dist.barrier()
    return rehearsal_classes


def _process_samples_and_targets(samples, targets, device):
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return samples, targets
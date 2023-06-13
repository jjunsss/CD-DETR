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
from models import _prepare_denoising_args

@decompose
def decompose_dataset(no_use_count: int, samples: utils.NestedTensor, targets: Dict, origin_samples: utils.NestedTensor, 
                      origin_targets: Dict, used_number: List) -> Tuple[int, List, utils.NestedTensor, Dict, utils.NestedTensor, Dict, List]:
    batch_size = len(targets)
    return (batch_size, no_use_count, samples, targets, origin_samples, origin_targets, used_number)

def Original_training(args, last_task, epo, idx, count, sum_loss, samples, targets, 
                      model: torch.nn.Module, teacher_model, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer,  
                      rehearsal_classes, train_check, current_classes): 

    last_epoch_check = epo == (args.Task_Epochs - 1)
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
    
    last_epoch_check = epo == (args.Task_Epochs - 1)
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

    # Add denoising arguments
    if args.model_name == 'dn_detr':
        model = _prepare_denoising_args(model, targets, args=args)

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
            outputs = model(samples)
            hook.remove()
            new_encoder = s_encoder[0]

            location_loss = torch.nn.functional.mse_loss(new_encoder.detach(), pre_encodre)
            del t_encoder, s_encoder, new_encoder, pre_encodre

        else:
            outputs = model(samples)

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

    # Add denoising arguments
    if args.model_name == 'dn_detr':
        model = _prepare_denoising_args(model, targets, args=args)

    outputs = model(samples)
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
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
from tqdm import tqdm
from copy import deepcopy


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
def icarl_feature_extractor_setup(args, model):
    '''
        In iCaRL, buffer manager collect samples closed to the mean of features of corresponding class.
        This function set up feature extractor for collecting.
    '''
    if args.distributed:
        feature_extractor = deepcopy(model.module.backbone)
    else:
        feature_extractor = deepcopy(model.backbone) # distributed:model.module.backbone
    
    for n, p in feature_extractor.named_parameters():
        p.requires_grad = False

    return feature_extractor


@torch.no_grad()
def icarl_prototype_setup(args, feature_extractor, device, current_classes):
    '''
        In iCaRL, buffer manager collect samples closed to the mean of features of corresponding class.
        This function set up prototype-mean of features of corresponding class-.
        Prototype can be the 'criteria' to select closest samples.
    '''
    
    feature_extractor.eval()
    proto = defaultdict(int)

    for cls in current_classes:
        _dataset, _data_loader, _sampler = IcarlDataset(args=args, single_class=cls)
        if _dataset == None:
            continue
        
        _cnt = 0
        for samples, targets, _, _ in tqdm(_data_loader, desc=f'Prototype:class_{cls}', disable=not utils.is_main_process()):
            samples = samples.to(device)
            feature, _ = feature_extractor(samples)
            feature_0 = feature[0].tensors
            proto[cls] += feature_0
            _cnt += 1
            if args.debug and _cnt == 10:
                break

        try:
            proto[cls] = proto[cls] / _dataset.__len__()
        except ZeroDivisionError:
            pass
        if args.debug and cls == 10:
            break

    return proto


@torch.no_grad()
def icarl_rehearsal_training(args, samples, targets, fe: torch.nn.Module, proto: Dict, device:torch.device,
                       rehearsal_classes, current_classes):
    '''
        iCaRL buffer collection.

        rehearsal_classes : [feature_sum, [[image_ids, difference] ...]]
        TODO: move line:200~218 to construct_rehearsal
    '''

    fe.eval()
    samples.to(device)

    feature, pos = fe(samples)
    feat_tensor = feature[0].tensors # TODO: cpu or cuda?

    for bt_idx in range(feat_tensor.shape[0]):
        feat_0 = feat_tensor[bt_idx]
        target = targets[bt_idx]
        label_tensor = targets[bt_idx]['labels']
        label_tensor_unique = torch.unique(label_tensor)
        label_list_unique = label_tensor_unique.tolist()

        for label in label_list_unique:
            try:
                class_mean = proto[label]
            except KeyError:
                print(f'label: {label} don\'t in prototype: {proto.keys()}')
                continue

            try: # rehearsal_classes[label] exist
                rehearsal_classes[label][0] = rehearsal_classes[label][0].to(device)

                exemplar_mean = (rehearsal_classes[label][0] + feat_0) / (len(rehearsal_classes[label]) + 1)
                difference = torch.mean(torch.sqrt(torch.sum((class_mean - exemplar_mean)**2, axis=1))).item()

                rehearsal_classes[label][0] = rehearsal_classes[label][0]                   
                rehearsal_classes[label][0]+= feat_0
                rehearsal_classes[label][1].append([target['image_id'].item(), difference])

            except KeyError:
                difference = torch.argmin(torch.sqrt(torch.sum((class_mean - feat_0)**2, axis=0))).item() # argmin is true????
                rehearsal_classes[label] = [feat_0, [[target['image_id'].item(), difference], ]]
            
            rehearsal_classes[label][1].sort(key=lambda x: x[1]) # sort with difference

    # construct rehearsal (3) - reduce exemplar set
    for label, data in tqdm(rehearsal_classes.items(), desc='Reduce_exemplar:', disable=not utils.is_main_process()):
        try:
            data[1] = data[1][:args.limit_image]
        except:
            continue

        return rehearsal_classes


def rehearsal_training(args, samples, targets, model: torch.nn.Module, criterion: torch.nn.Module, 
                       rehearsal_classes, current_classes):
    '''
        replay를 위한 데이터를 수집 시에 모델은 영향을 받지 않도록 설정
    '''
    model.eval() # For Fisher informations
    criterion.eval()
    
    device = torch.device("cuda")
    ex_device = torch.device("cpu")
    model.to(device)
    samples, targets = _process_samples_and_targets(samples, targets, device)

    outputs = inference_model(args, model, samples, targets, eval=True)
    # TODO : new input to model. plz change dn-detr model input (self.buffer_construct_loss)
    
    _ = criterion(outputs, targets, buffer_construct_loss=True)
    
    # This is collect replay buffer
    with torch.no_grad():
        batch_loss_dict = {}
        
        # Transform tensor to scarlar value for rehearsal step
        # This values sorted by batch index so first add all loss and second iterate each batch loss for update and lastly 
        # calculate all fisher information for updating all parameters
        batch_loss_dict["loss_bbox"] = [loss.item() for loss in criterion.losses_for_replay["loss_bbox"]]
        batch_loss_dict["loss_giou"] = [loss.item() for loss in criterion.losses_for_replay["loss_giou"]]
        batch_loss_dict["loss_labels"] = [loss.item() for loss in criterion.losses_for_replay["loss_labels"]]
    
        targets = [{k: v.to(ex_device) for k, v in t.items()} for t in targets]
        rehearsal_classes = construct_rehearsal(args, losses_dict=batch_loss_dict, targets=targets,
                                                rehearsal_dict=rehearsal_classes, 
                                                current_classes=current_classes,
                                                least_image=args.least_image,
                                                limit_image=args.limit_image)
    
    
    if utils.get_world_size() > 1:    
        dist.barrier()
    return rehearsal_classes

def fisher_training(args, samples, targets, model: torch.nn.Module, criterion: torch.nn.Module, 
                    optimizer, fisher_dict):
    '''
        replay buffer 내의 데이터들을 fisher 정보를 통해서 정렬하기 위해서 사용하려고 만들었음
        TODO: only training a one GPU processing (uitls.main_process())
    '''
    model.train() # For Fisher informations
    criterion.train()
    optimizer.zero_grad()
    device = torch.device("cuda")
    model.to(device)
    samples, targets = _process_samples_and_targets(samples, targets, device)

    outputs = inference_model(args, model, samples, targets)
    _ = criterion(outputs, targets, buffer_construct_loss=True)
    
    lbbox = criterion.losses_for_replay["loss_bbox"]
    lgiou = criterion.losses_for_replay["loss_giou"]
    llabels = criterion.losses_for_replay["loss_labels"]
    
    losses = sum(lbbox) + sum(lgiou) + sum(llabels)
    losses.backward()
    # Now, for each parameter in the model...
    FIM_value = 0
    for _, param in model.named_parameters():
        if param.grad is not None:
            # The gradient of the loss w.r.t. this parameter gives us information about how changing this parameter would affect the loss.
            # We square the gradient and sum over all elements to get a scalar quantity.
            FIM_value += torch.sum(param.grad.data.clone().pow(2)).item()
            param.grad = None
            
    # Use the overall index to get the correct key
    fisher_dict[targets[0]["image_id"].item()] = FIM_value

    return fisher_dict


def _process_samples_and_targets(samples, targets, device):
    samples = samples.to(device)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return samples, targets
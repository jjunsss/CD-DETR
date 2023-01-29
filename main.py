# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
import pickle
from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
import torch.distributed as dist
from Custom_Dataset import *
from custom_utils import *
from custom_prints import *


from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--lr_drop', default=500, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    #TODO : clip max grading usually set value 1 or 5 but this therory used to value 0.1 originally
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')


    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=True, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=3, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=3, type=float, # GIOU is Normalized IOU -> False일 때에도, 거리 차이에를 반영할 수 있음(기존의 IOU는 틀린 경우는 전부 0으로써 결과를 예상할 수 없었는데, GIOU는 실제 존재하는 GT BBOX와 Pred BBOX의 거리를 예측하도록 노력하게 됨.)
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=3, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=3, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='/data/LG/real_dataset/total_dataset/didvepz/', type=str)
    parser.add_argument('--file_name', default='./saved_rehearsal', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='CL_TEST',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--pretrained_model', default=None, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=328, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    #* Continual Learning 
    parser.add_argument('--Task', default=5, type=int, help='TASK is the number that divides the entire dataset.Like Domain') #if Task is 1, so then you could use it for normal training.
    parser.add_argument('--Task_Epochs', default=10, type=int, help='each Task epoch. like 1 task is 5 of 10 epoch training.. ')
    parser.add_argument('--Total_Classes', default=59, type=int, help='classes counts in custom COCODataset')
    parser.add_argument('--Total_Classes_Names', default=True, action='store_true', help="division classes through class names (DID,PZ,VE)")
    parser.add_argument('--Continual_Batch_size', default=3, type=int, help='continual batch training method')
    parser.add_argument('--CL_Limited', default=5, type=int, help='Use Limited Training in CL')#IF you choose False, you should meet data imbalancing in training.

    #* Rehearsal method
    parser.add_argument('--Rehearsal', default=False, action='store_true', help="use Rehearsal starategy in diverse CL method")
    parser.add_argument('--Memory', default=500, type=int, help='memory capacity for rehearsal training')
    return parser

def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    
    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    if args.pretrained_model is not None:
        model = load_model_params(model, args.pretrained_model)
    else:
        model = load_model_params(model,)
    
    model_without_ddp = model

    #* collate_fn : 최종 출력시에 모든 배치값에 할당해주는 함수를 말함. 여기서는 Nested Tensor 호출을 의미함.
    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        }
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma = 0.5)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])
    output_dir = Path(args.output_dir)

    print("Start training")
    start_time = time.time()
    file_name = args.file_name + "_" + str(0)
    if args.Task != 1:
        Divided_Classes = DivideTask_for_incre(args.Task, args.Total_Classes, args.Total_Classes_Names) 
        rehearsal_classes = {}
        if args.Total_Classes_Names == True :
            args.Task = len(Divided_Classes)

    #TODO : TASK 마다 훈련된 모델이 저장되게 설정해두기
    for task_idx in range(args.Task):
        #New task dataset
        dataset_train, data_loader_train, sampler_train, list_CC = Incre_Dataset(task_idx, args, Divided_Classes)
        #rehearsal + New task dataset (rehearsal Dataset은 유지하도록 설정)
        if task_idx >= 1 and len(rehearsal_classes) > 0 :
            data_loader_train = CombineDataset(args, rehearsal_classes, dataset_train, args.num_workers, args.Continual_Batch_size)#Rehearsal을 사용하는 Task 1 부터는 train_data_loader의 이름 자체를 변경
        else:
            print(f"no use rehearsal training method")
            
        #for epoch in range(args.Task_Epochs): #어차피 Task마다 훈련을 진행해야 하고, 중간점음 없을 것이므로 TASK마다 훈련이 되도록 만들어도 상관이 없음
        for epoch in range(5):
            if args.distributed:
                sampler_train.set_epoch(epoch)#TODO: 추후에 epoch를 기준으로 batch sampler를 추출하는 행위 자체가 오류를 일으킬 가능성이 있음 Incremental Learning에서                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
            print(f"task id : {task_idx}")
            print(f"each epoch id : {epoch} , Dataset length : {len(data_loader_train)}, current classes :{list_CC}")

            #original training
            #TODO: 매 에포크 마다 생성되는 save 파일과 지워지는 rehearsal 없도록 정리.
            rehearsal_classes = train_one_epoch( #save the rehearsal dataset. this method necessary need to clear dataset
                args, epoch, model, criterion, data_loader_train, optimizer, device, list_CC, rehearsal_classes)
            lr_scheduler.step()
        
        save_model_params(model_without_ddp, optimizer, lr_scheduler, args, args.output_dir, task_idx, int(args.Task))
        rehearsal_classes = rearrange_rehearsal(rehearsal_classes, list_CC)
        check_rehearsal_components(rehearsal_classes, args.output_dir, True, False)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    check_rehearsal_components(rehearsal_classes, args.output_dir, False, True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

#------------------------------------------------------------------------
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
from custom_buffer_manager import *
from custom_training import rehearsal_training

from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from main_component import TrainingPipeline

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--lr_drop', default=10, type=int)
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
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='/home/nextserver/Desktop/jjunsss/LG/LG/plustotal/', type=str)
    parser.add_argument('--file_name', default='./saved_rehearsal', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='./TEST/', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--LG', default=False, action='store_true', help="for LG Dataset process")
    
    #* CL Setting 
    parser.add_argument('--pretrained_model', default=None, help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=15, type=int, metavar='N',help='start epoch')
    parser.add_argument('--start_task', default=0, type=int, metavar='N',help='start task')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')

    #* Continual Learning 
    parser.add_argument('--Task', default=2, type=int, help='The task is the number that divides the entire dataset, like a domain.') #if Task is 1, so then you could use it for normal training.
    parser.add_argument('--Task_Epochs', default=16, type=int, help='each Task epoch, e.g. 1 task is 5 of 10 epoch training.. ')
    parser.add_argument('--Total_Classes', default=90, type=int, help='number of classes in custom COCODataset. e.g. COCO : 80 / LG : 59')
    parser.add_argument('--Total_Classes_Names', default=False, action='store_true', help="division of classes through class names (DID, PZ, VE). This option is available for LG Dataset")
    parser.add_argument('--CL_Limited', default=0, type=int, help='Use Limited Training in CL. If you choose False, you may encounter data imbalance in training.')
    parser.add_argument('--Construct_Replay', default=False, action='store_true', help="For cunstructing replay dataset")
    

    #* Rehearsal method
    parser.add_argument('--Rehearsal', default=False, action='store_true', help="use Rehearsal strategy in diverse CL method")
    parser.add_argument('--AugReplay', default=False, action='store_true', help="use Our augreplay strategy in step 2")
    parser.add_argument('--MixReplay', default=False, action='store_true', help="1:1 Mix replay solution, First Circular Training. Second Original Training")
    parser.add_argument('--Fake_Query', default=False, action='store_true', help="retaining previous task target through predict query")
    parser.add_argument('--Distill', default=False, action='store_true', help="retaining previous task target through predict query")
    parser.add_argument('--Memory', default=25, type=int, help='memory capacity for rehearsal training')
    parser.add_argument('--Continual_Batch_size', default=2, type=int, help='continual batch training method')
    parser.add_argument('--Rehearsal_file', default='./Rehearsal_LG-CL/', type=str)
    parser.add_argument('--teacher_model', default=None, type=str)
    return parser

def main(args):
    # Initializing
    pipeline = TrainingPipeline(args)
    args = pipeline.args 

    # Constructing only the replay buffer
    if args.Construct_Replay :
        pipeline.construct_replay_buffer()

    # Evaluation mode
    if args.eval:
        pipeline.evaluation_only_mode()
        
    print("Start training")
    start_time = time.time()

    # Training loop over tasks ( for incremental learning )
    for task_idx in range(pipeline.start_task, pipeline.tasks):
        # Check whether it's the first or last task
        first_training = (task_idx == 0)
        last_task = (task_idx+1 == pipeline.tasks)

        # Generate new dataset
        dataset_train, data_loader_train, sampler_train, list_CC = Incre_Dataset(task_idx, args, pipeline.Divided_Classes)

        # Ready for replay training strategy 
        if first_training is False and args.Rehearsal is True:
            if args.verbose :
                check_components(pipeline.rehearsal_classes, args.verbose)

            replay_dataset = copy.deepcopy(pipeline.rehearsal_classes)

            # Combine dataset for original and AugReplay(Circular)
            original_dataset, original_loader, original_sampler = CombineDataset(
                args, replay_dataset, dataset_train, args.num_workers, args.batch_size, old_classes=pipeline.load_replay, MixReplay="Original")

            AugRplay_dataset, AugRplay_loader, AugRplay_sampler = CombineDataset(
                args, replay_dataset, dataset_train, args.num_workers, args.batch_size, old_classes=pipeline.load_replay, MixReplay="AugReplay") 

            # Set a certain configuration
            dataset_train, data_loader_train, sampler_train = dataset_configuration(
                args, original_dataset, original_loader, original_sampler, AugRplay_dataset, AugRplay_loader, AugRplay_sampler)

            # Task change for learning rate scheduler
            pipeline.lr_scheduler.task_change()
            
            # Incremental training for each epoch
            pipeline.incremental_train_epoch(task_idx=task_idx, last_task=last_task, dataset_train=dataset_train,
                                              data_loader_train=data_loader_train, sampler_train=sampler_train,
                                              list_CC=list_CC)
            
    # Calculate and print the total time taken for training
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training completed in: ", total_time_str)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

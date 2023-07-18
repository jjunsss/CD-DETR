import argparse
import numpy as np

import argparse

def get_args_parser():
    parser = argparse.ArgumentParser('Parent parser', add_help=False)

    # ** Model
    parser.add_argument('--model_name', type=str, default='deform_detr', choices=['deform_detr', 'dn_detr']) # set model name
    parser.add_argument('--frozen_weights', type=str, default=None, help="Path to the pretrained model. If set, only the mask head will be trained")    

    # lr
    parser.add_argument('--clip_max_norm', default=0.1, type=float,help='gradient clipping max norm')
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--sgd', action='store_true')

    # * Backbone
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str, help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true', help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'), help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')    

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--hidden_dim', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int, help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--two_stage', default=False, action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true', help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false', help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_bbox', default=5, type=float, help="L1 box coefficient in the matching cost")    

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', default='/data/LG/real_dataset/total_dataset/didvepz/plustotal/', type=str)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--output_dir', default='./result/DIDPZ+VE', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)

    
    #* Setting 
    parser.add_argument('--LG', default=False, action='store_true', help="for LG Dataset process")
    parser.add_argument('--file_name', default='./saved_rehearsal', type=str)
    parser.add_argument('--verbose', default=False, action='store_true')
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--prefetch', default=2, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--eval', action='store_true')
    # parser.add_argument('--pretrained_model', default=None, help='resume from checkpoint')
    parser.add_argument('--pretrained_model', default=None, type=str, nargs='+', help='resume from checkpoint')
    parser.add_argument('--pretrained_model_dir', default=None, type=str, help='test all parameters')


    #* Continual Learning 
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='start epoch')
    parser.add_argument('--start_task', default=0, type=int, metavar='N', help='start task, if you set the construct_replay method, \
                                                                                so then you should set the start_task value. becuase start_task is task number of construct replay options ')
    parser.add_argument('--Task', default=2, type=int, help='The task is the number that divides the entire dataset, like a domain.') #if Task is 1, so then you could use it for normal training.
    parser.add_argument('--Task_Epochs', default=[16], type=int, nargs='+', help='each Task epoch, e.g. 1 task is 5 of 10 epoch training.. ')
    parser.add_argument('--Total_Classes', default=59, type=int, help='number of classes in custom COCODataset. e.g. COCO : 80 / LG : 59')
    parser.add_argument('--Total_Classes_Names', default=False, action='store_true', help="division of classes through class names (DID, PZ, VE). This option is available for LG Dataset")
    parser.add_argument('--CL_Limited', default=0, type=int, help='Use Limited Training in CL. If you choose False, you may encounter data imbalance in training.')

    #* Rehearsal method
    parser.add_argument('--Rehearsal', default=False, action='store_true', help="use Rehearsal strategy in diverse CL method")
    parser.add_argument('--AugReplay', default=False, action='store_true', help="use Our augreplay strategy in step 2")
    parser.add_argument('--MixReplay', default=False, action='store_true', help="1:1 Mix replay solution, First Circular Training. Second Original Training")
    parser.add_argument('--Mosaic', default=False, action='store_true', help="mosaic augmentation for autonomous training")
    parser.add_argument('--Rehearsal_file', default=None, type=str)
    parser.add_argument('--Construct_Replay', default=False, action='store_true', help="For cunnstructing replay dataset")
    
    parser.add_argument('--Sampling_strategy', default='hierarchical', type=str, help="hierarchical(ours), RODEO(del low unique labels), random \
                                                                                     , hier_highloss, hier_highlabels, hier_highlabels_highloss, hard(high labels)")

    parser.add_argument('--Sampling_mode', default='GM', type=str, help="normal, GM(GuaranteeMinimum, ours), ")
    parser.add_argument('--least_image', default=0, type=int, help='least image of each class, must need to exure_min mode')
    parser.add_argument('--limit_image', default=100, type=int, help='maximum image of all classes, must need to exure_min mode')
    
    parser.add_argument('--CER', default='fisher', type=str, help="fisher(ours), original, weight. This processes are used with \
                                                                   Augreplay ER")
    #* CL Strategy
    parser.add_argument('--Fake_Query', default=False, action='store_true', help="retaining previous task target through predict query")
    parser.add_argument('--Distill', default=False, action='store_true', help="retaining previous task target through predict query")
    parser.add_argument('--Branch_Incremental', default=False, action='store_true', help="MLP or something incremental with class")
    parser.add_argument('--teacher_model', default=None, type=str)
    parser.add_argument('--Continual_Batch_size', default=2, type=int, help='continual batch traiing method')
    parser.add_argument('--fisher_model', default=None, type=str, help='fisher model path')
    # 정완 디버그
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--num_debug_dataset', default=10, type=int) # 디버그 데이터셋 개수

    #* EVALUATION
    parser.add_argument('--all_data', default=False, action='store_true', help ="save your model output image") # I think this option is depreciated, so temporarily use for 79 path, and modify later ... .
    parser.add_argument('--test_file_list', default=None, type=str, nargs='+', \
        help='Test folder name')
    parser.add_argument('--FPP', default=False, action='store_true', help="Forgetting metrics")
    parser.add_argument('--Test_Classes', default=45, type=int, help="2 task eval(coco) : T1=45 / T2=90, 3task eval(coco) T1=30 T2=60 T3=90\
                                                                      this value be used to config model architecture in the adequate task")
    return parser    


def deform_detr_parser(parser):
    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')

    # lr
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=600, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')

    # * Backbone
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float, help="position / size * scale")   

    # * Transformer f
    parser.add_argument('--dim_feedforward', default=1024, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.1, type=float, help="Dropout applied in the transformer")    

    # * Matcher
    parser.add_argument('--set_cost_class', default=3, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=3, type=float, help="giou box coefficient in the matching cost")    

    # * Loss coefficients
    parser.add_argument('--cls_loss_coef', default=2, type=float)

    return parser


def dn_detr_parser(parser):
    # about dn args
    parser.add_argument('--use_dn', action="store_true",
                        help="use denoising training.")
    parser.add_argument('--scalar', default=5, type=int,
                        help="number of dn groups")
    parser.add_argument('--label_noise_scale', default=0.2, type=float,
                        help="label noise ratio to flip")
    parser.add_argument('--box_noise_scale', default=0.4, type=float,
                        help="box noise scale to shift and scale")
    parser.add_argument('--contrastive', action="store_true",
                        help="use contrastive training.")
    parser.add_argument('--use_mqs', action="store_true",
                        help="use mixed query selection from DINO.")
    parser.add_argument('--use_lft', action="store_true",
                        help="use look forward twice from DINO.")
    
    # lr
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--lr_backbone', default=1e-5, type=float, help='learning rate for backbone')
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--override_resumed_lr_drop', default=False, action='store_true')
    parser.add_argument('--drop_lr_now', action="store_true", help="load checkpoint and drop for 12epoch setting")
    parser.add_argument('--save_checkpoint_interval', default=10, type=int)

    # * Backbone
    parser.add_argument('--pe_temperatureH', default=20, type=int, help="Temperature for height positional encoding.")
    parser.add_argument('--pe_temperatureW', default=20, type=int, help="Temperature for width positional encoding.")
    parser.add_argument('--batch_norm_type', default='FrozenBatchNorm2d', type=str, 
                        choices=['SyncBatchNorm', 'FrozenBatchNorm2d', 'BatchNorm2d'], help="batch norm type for backbone")    
    
    # * Transformer 
    parser.add_argument('--dim_feedforward', default=2048, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--dropout', default=0.0, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--return_interm_layers', action='store_true', help="Train segmentation head if the flag is provided")
    parser.add_argument('--backbone_freeze_keywords', nargs="+", type=str,  help='freeze some layers in backbone. for catdet5.')
    parser.add_argument('--num_results', default=300, type=int, help="Number of detection results")
    parser.add_argument('--pre_norm', action='store_true',  help="Using pre-norm in the Transformer blocks.")    
    parser.add_argument('--num_select', default=300, type=int,  help='the number of predictions selected for evaluation')
    parser.add_argument('--transformer_activation', default='prelu', type=str)
    parser.add_argument('--num_patterns', default=0, type=int,  help='number of pattern embeddings. See Anchor DETR for more details.')
    parser.add_argument('--random_refpoints_xy', action='store_true', help="Random init the x,y of anchor boxes and freeze them.")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float, help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float, help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--eos_coef', default=0.1, type=float)
    parser.add_argument('--cls_loss_coef', default=1, type=float)

    # Traing utils
    parser.add_argument('--note', default='', help='add some notes to the experiment')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--pretrain_model_path', help='load from other checkpoint')
    parser.add_argument('--finetune_ignore', type=str, nargs='+', help="A list of keywords to ignore when loading pretrained models.")
    parser.add_argument('--find_unused_params', default=False, action='store_true')

    parser.add_argument('--save_results', action='store_true', help="For eval only. Save the outputs for all images.")
    parser.add_argument('--save_log', action='store_true', help="If save the training prints to the log file.")

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,help='number of distributed processes')
    parser.add_argument("--local_rank", type=int, help='local rank for DistributedDataParallel')
    parser.add_argument('--amp', action='store_true',help="Train with mixed precision")
    
    parser.add_argument('--orgcocopath', action='store_true', help='for original coco directory path')
    
    return parser    
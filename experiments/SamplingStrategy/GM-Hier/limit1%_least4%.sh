#!/usr/bin/env bash

set -x

# Define variables
PUS_PER_NODE=4
BATCH_SIZE=8
MODEL_NAME="dn_detr"
COCO_PATH="/home/user/sumin/paper/COCODIR/"  # /home/user/sumin/paper/COCODIR/ for 79 server. /data/LG/coco/cocodataset for 129
OUTPUT_DIR="./GM_Hier_Train_1%_least2%/"
START_TASK=1
START_EPOCH=0
TASK_EPOCHS=12
NUM_WORKERS=24
TOTAL_CLASSES=90
LIMIT_IMAGE=1200
LEAST_IMAGE=48 # 4% 
TASK=2
REHEARSAL_FILE="./GM_Hier_Train_1%_least2%/"
PRETRAINED_MODEL="./GM_Hier_Train_1%_least2%/checkpoints/cp_02_02_1.pth"
SAMPLING_STRATEGY="hierarchical"
SAMPLING_MODE="GM"

# Prepare the command
CMD="PUS_PER_NODE=4 ./tools/run_dist_launch.sh $PUS_PER_NODE ./configs/r50_dn_detr.sh \
    --batch_size $BATCH_SIZE \
    --model_name $MODEL_NAME \
    --use_dn \
    --coco_path $COCO_PATH \
    --output_dir $OUTPUT_DIR \
    --start_task $START_TASK \
    --start_epoch $START_EPOCH \
    --Task_Epochs $TASK_EPOCHS \
    --num_workers $NUM_WORKERS \
    --Total_Classes $TOTAL_CLASSES \
    --limit_image $LIMIT_IMAGE \
    --least_image $LEAST_IMAGE \
    --Branch_Incremental \
    --Task $TASK \
    --Rehearsal_file $REHEARSAL_FILE \
    --pretrained_model $PRETRAINED_MODEL \
    --Rehearsal \
    --orgcocopath \
    --Sampling_strategy $SAMPLING_STRATEGY \
    --Sampling_mode $SAMPLING_MODE \
    $@"

# Print the command
echo $CMD

# Run the command
eval $CMD
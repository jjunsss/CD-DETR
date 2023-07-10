#!/usr/bin/env bash

set -x

# Define variables
PUS_PER_NODE=4
BATCH_SIZE=16
MODEL_NAME="dn_detr"
COCO_PATH="/data/LG/coco/cocodataset/"
# OUTPUT_DIR="./GM_Hier_Train_1%/"
START_TASK=0
START_EPOCH=0
TASK_EPOCHS=12
NUM_WORKERS=32
TOTAL_CLASSES=90
LIMIT_IMAGE=1200
LEAST_IMAGE=36 # 3% value
TASK=2
REHEARSAL_FILE="./Underbound/"
PRETRAINED_MODEL="./DN_Task1_40-40.pth"
SAMPLING_STRATEGY="hierarchical"
SAMPLING_MODE="GM"

# Prepare the command
CMD="PUS_PER_NODE=4 ./tools/run_dist_launch.sh $PUS_PER_NODE ./configs/r50_dn_detr.sh \
    --batch_size $BATCH_SIZE \
    --model_name $MODEL_NAME \
    --use_dn \
    --Branch_Incremental \
    --coco_path $COCO_PATH \
    --output_dir $REHEARSAL_FILE \
    --start_task $START_TASK \
    --start_epoch $START_EPOCH \
    --Task_Epochs $TASK_EPOCHS \
    --num_workers $NUM_WORKERS \
    --Total_Classes $TOTAL_CLASSES \
    --Task $TASK \
    --Rehearsal_file $REHEARSAL_FILE"

# Print the command
echo $CMD

# Run the command
eval $CMD
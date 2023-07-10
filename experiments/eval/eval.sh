#!/usr/bin/env bash

set -x

# Define variables
PUS_PER_NODE=1
BATCH_SIZE=12
MODEL_NAME="dn_detr"
COCO_PATH="/data/coco/cocodataset/" 
#/home/user/sumin/paper/COCODIR/ for 79 server. 
#/data/LG/coco/cocodataset for 129
#/data/coco/cocodataset/ for local
OUTPUT_DIR="/data/eval/GM-Hier-Testing/limit1%-least5%/"
# OUTPUT_DIR="/data/eval/UpperUnder/Under/"
#/home/uvll/Desktop/jjunsss/dataset/cocodataset/
# OUTPUT_DIR="/data/eval/GM-Hier-Testing/limit1%-least1%/"
TASK_EPOCHS=12
NUM_WORKERS=24
TOTAL_CLASSES=90
TEST_CLASSES=90
LIMIT_IMAGE=1200
LEAST_IMAGE=12
TASK=2
REHEARSAL_FILE="/data/eval/GM-Hier-Testing/limit1%-least5%/"
PRETRAINED_MODEL="/data/eval/GM-Hier-Testing/limit1%-least5%/cp_02_02_8.pth"
SAMPLING_STRATEGY="hierarchical"
SAMPLING_MODE="GM"

# Prepare the command
CMD="PUS_PER_NODE=$PUS_PER_NODE ./tools/run_dist_launch.sh $PUS_PER_NODE ./configs/r50_dn_detr.sh \
    --batch_size $BATCH_SIZE \
    --model_name $MODEL_NAME \
    --use_dn \
    --coco_path $COCO_PATH \
    --output_dir $OUTPUT_DIR \
    --num_workers $NUM_WORKERS \
    --Total_Classes $TOTAL_CLASSES \
    --Task $TASK \
    --Rehearsal_file $REHEARSAL_FILE \
    --pretrained_model $PRETRAINED_MODEL \
    --Branch_Incremental \
    --eval \
    --Test_Classes $TEST_CLASSES "

# Print the command
echo $CMD

# Run the command
eval $CMD
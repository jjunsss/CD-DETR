#!/usr/bin/env bash

set -x

# Define variables
#* Mosaic ER mehtod
#* random sampling, normal mode(random)
#* 2 new, 2 old batchsize
# file:///home/uvllkjs/Downloads/sensors-20-06777.pdf
PUS_PER_NODE=4
BATCH_SIZE=16
MODEL_NAME="dn_detr"
COCO_PATH="/home/user/sumin/paper/COCODIR/" # /home/user/sumin/paper/COCODIR/ for 79 server. /data/LG/coco/cocodataset for 129, 60
# OUTPUT_DIR="./GM_Hier_Train_1%/"
START_TASK=0
START_EPOCH=0
TASK_EPOCHS=12
NUM_WORKERS=24
TOTAL_CLASSES=90
LIMIT_IMAGE=1200
LEAST_IMAGE=48 #4%
TASK=2
REHEARSAL_FILE="./mosaicER/"
PRETRAINED_MODEL="./DN_Task1_40-40.pth"
SAMPLING_STRATEGY="random"
SAMPLING_MODE="normal"

# Prepare the command
CMD="PUS_PER_NODE=4 ./tools/run_dist_launch.sh $PUS_PER_NODE ./configs/r50_dn_detr.sh \
    --batch_size $BATCH_SIZE \
    --model_name $MODEL_NAME \
    --use_dn \
    --coco_path $COCO_PATH \
    --output_dir $REHEARSAL_FILE \
    --start_task $START_TASK \
    --num_workers $NUM_WORKERS \
    --Total_Classes $TOTAL_CLASSES \
    --limit_image $LIMIT_IMAGE \
    --Task $TASK \
    --Rehearsal_file $REHEARSAL_FILE \
    --pretrained_model $PRETRAINED_MODEL \
    --Sampling_strategy $SAMPLING_STRATEGY \
    --Sampling_mode $SAMPLING_MODE \
    --Construct_Replay \
    --orgcocopath"

# Print the command
echo $CMD

# Run the command
eval $CMD
#!/usr/bin/env bash

set -x

EXP_DIR=exps/s6030icarl/
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --pretrained_model ./exps/s6030icarl/cp_02_02_2.pth \
    --coco_path ../COCODIR \
    --batch_size 4 \
    --with_box_refine \
    --Rehearsal \
    --Memory 25 \
    --Task 2 \
    --Task_Epochs 15 \
    --start_epoch 3 \
    --start_task 1 \
    ${PY_ARGS}

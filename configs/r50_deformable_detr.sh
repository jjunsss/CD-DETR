#!/usr/bin/env bash

set -x

EXP_DIR=exps/coco_icarl/
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --pretrained_model baseline_ddetr.pth \
    --coco_path ../COCODIR \
    --batch_size 4 \
    --with_box_refine \
    --CL_Limited 0 \
    ${PY_ARGS}

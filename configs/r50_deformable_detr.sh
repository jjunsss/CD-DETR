#!/usr/bin/env bash

set -x

EXP_DIR=exps/pre_underbound/
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --coco_path ../COCODIR \
    --batch_size 4 \
    --with_box_refine \
    --CL_Limited 0 \
    --Memory 25 \
    ${PY_ARGS}
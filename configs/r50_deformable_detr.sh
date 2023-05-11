#!/usr/bin/env bash

set -x

EXP_DIR=exps/pre_underbound/
PY_ARGS=${@:1}

python -u main.py \
    --output_dir ${EXP_DIR} \
    --coco_path ../COCODIR \
    --with_box_refine \
    ${PY_ARGS}
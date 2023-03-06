#!/usr/bin/env bash
model=$1
para=$2
current_path=`pwd`
python test.py --without_wandb \
--resume_model_path=$current_path/model/$model/last_$model.pkl $para

python test.py --without_wandb \
--resume_model_path=$current_path/model/$model/best_$model.pkl $para


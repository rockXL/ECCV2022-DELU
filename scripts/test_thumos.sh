#!/usr/bin/env bash
cd ..

python test.py \
--model_name delu_thumos \
--dataset_name Thumos14reduced \
--path_dataset /dev/THUMOS14/Thumos14reduced \
--without_wandb
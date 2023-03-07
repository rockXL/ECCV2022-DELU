#!/usr/bin/env bash
current_path=`pwd`
num_similar=$1
batch_size=$2

python main.py \
--model_name DELU \
--seed 0 \
--alpha_edl 0 \
--alpha_uct_guide 0 \
--amplitude 0.7 \
--interval 50 \
--max_seqlen 500 \
--lr 0.00005 \
--k 7 \
--num_similar $num_similar \
--batch_size $batch_size \
--dataset_name Thumos14reduced \
--path_dataset /dev/THUMOS14/Thumos14reduced \
--save_model_path $current_path \
--model_name co2_${num_similar}_${batch_size} \
--num_class 20 \
--use_model DELU \
--max_iter 5000 \
--dataset SampleDataset \
--weight_decay 0.001 \
--AWM BWA_fusion_dropout_feat_v2

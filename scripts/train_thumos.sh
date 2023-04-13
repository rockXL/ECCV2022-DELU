#!/usr/bin/env bash
current_path=`pwd`

python main.py \
--model_name DELU \
--seed 0 \
--alpha_edl 1.3 \
--alpha_uct_guide 0.4 \
--amplitude 0.7 \
--alpha2 0.4 \
--interval 50 \
--max_seqlen 320 \
--lr 0.00005 \
--k 7 \
--num_similar 3 \
--batch_size 13 \
--do_video_concat_aug False \
--use_multi_speed_feature False \
--dataset_name Thumos14reduced \
--path_dataset /dev/THUMOS14/Thumos14reduced \
--save_model_path $current_path \
--model_name default \
--num_class 20 \
--use_model DELU \
--max_iter 5000 \
--dataset SampleDataset \
--weight_decay 0.001 \
--AWM BWA_fusion_dropout_feat_v2

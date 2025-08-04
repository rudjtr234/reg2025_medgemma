#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --num_processes=4 \
  fintuning_medgemma_v4.py \
  --model_path /your_path_directory/medgemma-4b-it \
  --train_json /your_path_directory/medgemma_tile_final_v0.1.2.json \
  --output_dir /your_path_directory/fintuning_model \
  --epochs 3 \
  --lr 3e-5 \
  --rank 8 \
  --batch_size 16 \
  --grad_accum 2 \
  --save_steps 1000 \
  --eval_steps 500 \
  --logging_steps 50 \
  --num_workers 4 \
  --image_size 896

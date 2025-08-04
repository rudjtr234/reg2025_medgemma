#!/bin/bash

accelerate launch fintuning_medgemma_v4.py \
  --model_path /home/mts/models/medgemma-4b-it \
  --train_json /home/mts/data/train.json \
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

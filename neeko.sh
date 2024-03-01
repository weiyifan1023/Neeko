#!/bin/bash

current_time=$(date +"%Y%m%d%H%M%S")

python train.py \
    --model_name_or_path /path/to/Llama-2-7b-hf \
    --data_path /path/to/datasets/character-llm-data/prompted/shuffle.jsonl \
    --embds_dir /path/to/role_embds \
    --do_train \
    --finetuning_type moelora \
    --output_dir ./ckpt/neeko/$current_time/ \
    --max_source_length 4096 \
    --overwrite_cache \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate 2e-4 \
    --num_train_epochs 1.0 \
    --plot_loss \
    --lora_rank 32 \
    --num_moe 8 \
    --gating Dense \
    --fp16 \
    --remove_unused_columns False \
    --dataset character-llm

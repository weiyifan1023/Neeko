#!/bin/bash

# 获取当前时间并格式化为年月日时分秒
current_time=$(date +"%Y%m%d%H%M%S")

# 定义需要执行的指令列表
commands=(
"python train.py \\
--model_name_or_path /home/source/PTM/Llama-2-7b-hf \\
--do_train \\
--finetuning_type lora \\
--output_dir ./ckpt/lora/$current_time/ \\
--max_source_length 2048 \\
--overwrite_cache \\
--per_device_train_batch_size 2 \\
--gradient_accumulation_steps 8 \\
--lr_scheduler_type cosine \\
--logging_steps 10 \\
--save_steps 1000 \\
--learning_rate 2e-4 \\
--num_train_epochs 2.0 \\
--plot_loss \\
--lora_rank 8 \\
--fp16 \\
--dataset character-llm"

)

# 执行指令
for cmd in "${commands[@]}"; do
    echo "Executing:"
    echo "$cmd"
    eval "$cmd"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "Command failed with exit code $exit_code, skipping remaining commands."
        break
    fi
done

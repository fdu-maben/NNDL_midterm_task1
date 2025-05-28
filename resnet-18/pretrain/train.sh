#!/usr/bin/env bash

# hyperparameter_search.sh
# 包含是否使用预训练模型的超参数搜索示例

# 定义超参数范围
learning_rates=(0.001 0.0005 0.0001)
batch_sizes=(32 64)
num_epochs_list=(100)
use_pretrained_options=(false)

for use_pre in "${use_pretrained_options[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for bs in "${batch_sizes[@]}"; do
      for ne in "${num_epochs_list[@]}"; do
        # 构造日志与模型保存目录
        tag="pre_${use_pre}_lr${lr}_bs${bs}_ne${ne}"
        log_dir="logs/${tag}"
        output_path="models/${tag}.pth"
        mkdir -p "$log_dir"
        mkdir -p "$(dirname "$output_path")"

        # 如果 use_pre=true，就加上 --use-pretrained 标志
        if [ "$use_pre" = "true" ]; then
          pre_flag="--use-pretrained"
        else
          pre_flag=""
        fi

        # 运行训练脚本
        python pretrain_train.py \
          --learning-rate "$lr" \
          --batch-size "$bs" \
          --num-epochs "$ne" \
          --log-dir "$log_dir" \
          --output-path "$output_path" \
          $pre_flag
      done
    done
  done
done

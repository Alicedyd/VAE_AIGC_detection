#!/bin/bash

###############################
### 用户可配置参数区域 BEGIN ###
###############################

# 硬件设置
GPU_IDS=3               # 可用GPU编号，多卡用逗号分隔
BATCH_SIZE=32           # 批次大小，根据显存调整

# 训练参数
EPOCHS=1                # 训练轮次
LR=0.0005               # 初始学习率
OPTIMIZER="adam"        # 优化器选择

# 路径配置
REAL_DATA="/root/autodl-tmp/AIGC_data/MSCOCO"          # 真实数据集路径
FAKE_DATA="/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-xl-base-1.0"  # 生成数据路径

# 模型配置
ARCH="DINOv2-LoRA:dinov2_vitl14"  # 模型架构
LORA_RANK=8             # LoRA秩
LORA_ALPHA=1            # LoRA alpha值

USE_VAE=true            # 是否启用VAE
VAE_PREPROCESS="None"   # VAE预处理方式

# 实验配置
EXP_NAME="coco_dino_vae_single_none_contrastive"  # 实验名称
USE_CONTRASTIVE=true    # 启用对比学习

###############################
### 用户可配置参数区域 END   ###
###############################

# 自动推导参数
OPT_FLAGS=""
$USE_VAE      && OPT_FLAGS+=" --vae"
$USE_CONTRASTIVE && OPT_FLAGS+=" --contrastive"

# 执行训练命令
python train.py \
    --name="$EXP_NAME" \
    --real_list_path="$REAL_DATA" \
    --fake_list_path="$FAKE_DATA" \
    --data_mode=mscoco \
    --arch="$ARCH" \
    --lora_rank="$LORA_RANK" \
    --lora_alpha="$LORA_ALPHA" \
    --pre_vae="$VAE_PREPROCESS" \
    --fix_backbone \
    --batch_size="$BATCH_SIZE" \
    --lr="$LR" \
    --optim="$OPTIMIZER" \
    --niter="$EPOCHS" \
    --gpu_ids="$GPU_IDS" \
    $OPT_FLAGS
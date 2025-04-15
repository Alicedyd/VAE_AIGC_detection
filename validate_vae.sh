#!/bin/bash

# 模型相关设置
DATA_MODE="ours"
ARCH="DINOv2-LoRA:dinov2_vitl14"
CKPT="./checkpoints/coco_dino_vae_single/model_iters_2000.pth"
RESULT_FOLDER="./result/dino_vae_single/"
LORA_RANK=8
LORA_ALPHA=1
JPEG_QUALITY=95
GPU_ID=3

# DRCT-2M 数据集设置
# 数据集根目录
BASE_NAME="DRCT-2M"
BASE_PATH="/root/autodl-tmp/AIGC_data/DRCT-2M"

# 数据集子目录和对应的 key
DATASETS=( \
    "controlnet-canny-sdxl-1.0:cn-sdxl" \
    "lcm-lora-sdv1-5:lcm-sd15" \
    "lcm-lora-sdxl:lcm-sdxl" \
    "ldm-text2im-large-256:ldm-t2i" \
    "sd-controlnet-canny:sd-cn" \
    "sd-turbo:sd-turbo" \
    "sd21-controlnet-canny:sd21-cn" \
    "sdxl-turbo:sdxl-turbo" \
    "stable-diffusion-2-1:sd21" \
    "stable-diffusion-2-inpainting:sd21-inpainting" \
    "stable-diffusion-inpainting:sd-inpainting" \
    "stable-diffusion-v1-4:sd14" \
    "stable-diffusion-v1-5:sd15" \
    "stable-diffusion-xl-base-1.0:sdxl" \
    "stable-diffusion-xl-refiner-1.0:sdxl-refiner" \
    "stable-diffusion-xl-1.0-inpainting-0.1:sdxl-inpainting" \
) 

# # GenImage 数据集设置
# BASE_NAME="GenImage"
# BASE_PATH="/root/autodl-tmp/AIGC_data/GenImage"

# # 数据集子目录和对应的 key
# DATASETS=( \
#     "ADM:ADM" \
#     "BigGAN:BigGAN" \
#     "glide:glide" \
#     "Midjourney:Midjourney" \
#     "stable_diffusion_v_1_4:sd14" \
#     "stable_diffusion_v_1_5:sd15" \
#     "VQDM:VQDM" \
#     "wukong:wukong" \
# ) 

# 初始化路径和key字符串
REAL_PATHS=()
FAKE_PATHS=()
KEYS=()

for ds in "${DATASETS[@]}"; do
    IFS=":" read -r NAME KEY <<< "$ds"
    REAL_PATHS+=("/root/autodl-tmp/AIGC_data/MSCOCO/val2017")
    FAKE_PATHS+=("$BASE_PATH/$NAME/val2017")
    # REAL_PATHS+=("$BASE_PATH/$NAME/val/nature")
    # FAKE_PATHS+=("$BASE_PATH/$NAME/val/ai")
    KEYS+=("$BASE_NAME/$KEY")
done

# 拼接成逗号分隔的字符串
REAL_PATH=$(IFS=, ; echo "${REAL_PATHS[*]}")
FAKE_PATH=$(IFS=, ; echo "${FAKE_PATHS[*]}")
KEY=$(IFS=, ; echo "${KEYS[*]}")

# Chameleon 数据集设置
# REAL_PATH="/root/autodl-tmp/AIGC_data/Chameleon/test/0_real"
# FAKE_PATH="/root/autodl-tmp/AIGC_data/Chameleon/test/1_fake"
# KEY="Chameleon"

# 执行 Python 脚本
python validate.py \
    --data_mode "$DATA_MODE" \
    --arch="$ARCH" \
    --real_path "$REAL_PATH" \
    --fake_path "$FAKE_PATH" \
    --key="$KEY" \
    --ckpt="$CKPT" \
    --result_folder="$RESULT_FOLDER" \
    --lora_rank="$LORA_RANK" \
    --lora_alpha="$LORA_ALPHA" \
    --gpu_id "$GPU_ID"

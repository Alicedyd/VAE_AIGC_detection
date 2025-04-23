#!/bin/bash

# 模型相关设置
DATA_MODE="ours"
ARCH="DINOv2-LoRA:dinov2_vitl14"
CKPT="./checkpoints/mult_double_resize_pure_part/model_epoch_0.pth"
RESULT_FOLDER="./result/multi_double_resize_pure_part"
LORA_RANK=8
LORA_ALPHA=1
JPEG_QUALITY=95
GPU_ID=0

SAVE_BAD_CASE=false

OPT_FLAGS=""
$SAVE_BAD_CASE && OPT_FLAGS+=" --save_bad_case"

# 初始化路径和key字符串
REAL_PATHS=()
FAKE_PATHS=()
KEYS=()

# DRCT-2M 数据集设置
# 数据集根目录
BASE_NAME="DRCT-2M"
BASE_PATH="/root/autodl-tmp/AIGC_data/DRCT-2M"

# 数据集子目录和对应的 key
# DATASETS=( \
#     "controlnet-canny-sdxl-1.0:cn-sdxl" \
#     "lcm-lora-sdv1-5:lcm-sd15" \
#     "lcm-lora-sdxl:lcm-sdxl" \
#     "ldm-text2im-large-256:ldm-t2i" \
#     "sd-controlnet-canny:sd-cn" \
#     "sd-turbo:sd-turbo" \
#     "sd21-controlnet-canny:sd21-cn" \
#     "sdxl-turbo:sdxl-turbo" \
#     "stable-diffusion-2-1:sd21" \
#     "stable-diffusion-2-inpainting:sd21-inpainting" \
#     "stable-diffusion-inpainting:sd-inpainting" \
#     "stable-diffusion-v1-4:sd14" \
#     "stable-diffusion-v1-5:sd15" \
#     "stable-diffusion-xl-base-1.0:sdxl" \
#     "stable-diffusion-xl-refiner-1.0:sdxl-refiner" \
#     "stable-diffusion-xl-1.0-inpainting-0.1:sdxl-inpainting" \
# ) 

DATASETS=( \
    "ldm-text2im-large-256:ldm-t2i" \
    "stable-diffusion-v1-4:sd14" \
    "stable-diffusion-v1-5:sd15" \
    "stable-diffusion-2-1:sd21" \
    "stable-diffusion-xl-base-1.0:sdxl" \
    "stable-diffusion-xl-refiner-1.0:sdxl-refiner" \
    "sd-turbo:sd-turbo" \
    "sdxl-turbo:sdxl-turbo" \
    "lcm-lora-sdv1-5:lcm-sd15" \
    "lcm-lora-sdxl:lcm-sdxl" \
    "sd-controlnet-canny:sd-cn" \
    "sd21-controlnet-canny:sd21-cn" \
    "controlnet-canny-sdxl-1.0:sdxl-cn" \
    "stable-diffusion-inpainting:sd-inpainting" \
    "stable-diffusion-2-inpainting:sd21-inpainting" \
    "stable-diffusion-xl-1.0-inpainting-0.1:sdxl-inpainting" \
)

for ds in "${DATASETS[@]}"; do
    IFS=":" read -r NAME KEY <<< "$ds"
    REAL_PATHS+=("/root/autodl-tmp/AIGC_data/MSCOCO/val2017")
    FAKE_PATHS+=("$BASE_PATH/$NAME/val2017")
    KEYS+=("$BASE_NAME/$KEY")
done

# GenImage 数据集设置
BASE_NAME="GenImage"
BASE_PATH="/root/autodl-tmp/AIGC_data/GenImage"

# 数据集子目录和对应的 key
DATASETS=( \
    "ADM:ADM" \
    "BigGAN:BigGAN" \
    "glide:glide" \
    "Midjourney:Midjourney" \
    "stable_diffusion_v_1_4:sd14" \
    "stable_diffusion_v_1_5:sd15" \
    "VQDM:VQDM" \
    "wukong:wukong" \
) 

for ds in "${DATASETS[@]}"; do
    IFS=":" read -r NAME KEY <<< "$ds"
    REAL_PATHS+=("$BASE_PATH/$NAME/val/nature")
    FAKE_PATHS+=("$BASE_PATH/$NAME/val/ai")
    KEYS+=("$BASE_NAME/$KEY")
done

# Chameleon 数据集
REAL_PATHS+=("/root/autodl-tmp/AIGC_data/Chameleon/test/0_real")
FAKE_PATHS+=("/root/autodl-tmp/AIGC_data/Chameleon/test/1_fake")
KEYS+=("Chameleon")

# 添加新的Eval_GEN路径
BASE_NAME="Eval_GEN"
EVAL_GEN_MODELS=( \
    "Flux" \
    "GoT" \
    "Infinity" \
    "OmniGen" \
    "NOVA" \
    "sd14" \
    "sdxl" \
)

for model in "${EVAL_GEN_MODELS[@]}"; do
    REAL_PATHS+=("/root/autodl-tmp/AIGC_data/MSCOCO/val2017")
    FAKE_PATHS+=("/root/autodl-tmp/AIGC_data/Eval_GEN/$model")
    KEYS+=("$BASE_NAME/$model")
done

# 添加GPT-ImgEval路径
REAL_PATHS+=("/root/autodl-tmp/AIGC_data/MSCOCO/val2017")
FAKE_PATHS+=("/root/autodl-tmp/AIGC_data/GPT-ImgEval")
KEYS+=("GPT-ImgEval")

# 拼接成逗号分隔的字符串
REAL_PATH=$(IFS=, ; echo "${REAL_PATHS[*]}")
FAKE_PATH=$(IFS=, ; echo "${FAKE_PATHS[*]}")
KEY=$(IFS=, ; echo "${KEYS[*]}")

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
    --gpu_id "$GPU_ID" \
    $OPT_FLAGS
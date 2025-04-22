EXP_NAME=mult_double_resize_pure_part
REAL_LIST="/root/autodl-tmp/AIGC_data/MSCOCO"
FAKE_LIST="/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-xl-base-1.0"
DATA_MODE=mscoco

ARCH=DINOv2-LoRA:dinov2_vitl14
LORA_RANK=8
LORA_ALPHA=1

VAE="XL,MSE,EMA"
USE_RESIZE=true
# RESIZE_FACTOR=0.25,0.5,0.75
# UPPER_RESIZE_FACTOR=1.5,2,2.5
RESIZE_FACTOR=0.2,0.4,0.6,0.8,1.0
UPPER_RESIZE_FACTOR=1.0,1.5,2.0,2.5,3.0

BATCH_SIZE=32
LR=1e-5
OPTIM=adam
NITER=1
ACCUMU_ITER=64

GPU_IDS=1

USE_CONTRASTIVE=true

OPT_FLAGS=""
$USE_CONTRASTIVE && OPT_FLAGS+=" --contrastive"
$USE_RESIZE && OPT_FLAGS+=" --resize_factors $RESIZE_FACTOR --upper_resize_factors $UPPER_RESIZE_FACTOR" || OPT_FLAGS+=""

python train.py \
    --name $EXP_NAME \
    --real_list_path $REAL_LIST \
    --fake_list_path $FAKE_LIST \
    --data_mode $DATA_MODE \
    --arch $ARCH \
    --fix_backbone \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --accumulation_steps $ACCUMU_ITER \
    --optim $OPTIM \
    --niter $NITER \
    --gpu_ids $GPU_IDS \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --vae_models $VAE \
    $OPT_FLAGS

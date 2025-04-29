REAL_LIST="/root/autodl-tmp/AIGC_data/MSCOCO/train2017"
FAKE_LIST="/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-xl-base-1.0"
DATA_MODE=mscoco

ARCH=Imagenet:resnet50
LORA_RANK=8
LORA_ALPHA=1

VAE="/root/autodl-tmp/AIGC_data/MSCOCO_XL/train2017"
USE_RESIZE=true
DOWN_RESIZE_FACTOR=0.2,0.4,0.6,0.8,1.0
UPPER_RESIZE_FACTOR=1.0,1.25,1.5,1.75,2.0

P_BLEND=0
RATIO_BLEND=0

P_SHUFFLE=0

RANDOM_MASK=false

ALIGN_JPEG=true

USE_JPEG_FACTOR=true
# JPEG_FACTOR=95,100
# JPEG_FACTOR=85,100
JPEG_FACTOR=75,100

BATCH_SIZE=32
LR=1e-4
OPTIM=adam
NITER=50
ACCUMU_ITER=64

GPU_IDS=3

USE_CONTRASTIVE=true
USE_TOKEN_CONTRASTIVE=false

OPT_FLAGS=""
$RANDOM_MASK && OPT_FLAGS+=" --random_mask"
$ALIGN_JPEG && OPT_FLAGS+=" --jpeg_aligned"
$USE_JPEG_FACTOR && OPT_FLAGS+=" --jpeg_quality ${JPEG_FACTOR}"
$USE_CONTRASTIVE && OPT_FLAGS+=" --contrastive "
$USE_TOKEN_CONTRASTIVE && OPT_FLAGS+="--token_contrastive "
$USE_RESIZE && OPT_FLAGS+=" --down_resize_factors $DOWN_RESIZE_FACTOR --upper_resize_factors $UPPER_RESIZE_FACTOR" || OPT_FLAGS+=""

EXP_NAME="${ARCH}_lora${LORA_RANK}_VAE_sdxl_resize_lr${LR}_bs${BATCH_SIZE}_acc${ACCUMU_ITER}_${NITER}epoch_PBLEND_${P_BLEND}RBLEND_${RATIO_BLEND}_PSHUFFLE_${P_SHUFFLE}"
$USE_CONTRASTIVE && EXP_NAME+="_contra"
$USE_TOKEN_CONTRASTIVE && EXP_NAME+="_token-contra"
$USE_JPEG_FACTOR && EXP_NAME+="_JPEG_factor_${JPEG_FACTOR}"
$RANDOM_MASK && EXP_NAME+="_random_mask"
# EXP_NAME="debug"

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
    --p_blend $P_BLEND \
    --ratio_blend $RATIO_BLEND \
    --p_shuffle $P_SHUFFLE \
    $OPT_FLAGS

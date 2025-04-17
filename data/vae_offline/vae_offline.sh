INPUT=/root/autodl-tmp/AIGC_data/GenImage/ADM/train/nature/
OUTPUT=/root/autodl-tmp/AIGC_data/ImageNet_XL/train


# INPUT=/root/autodl-tmp/AIGC_data/MSCOCO/train2017
# OUTPUT=/root/autodl-tmp/AIGC_data/MSCOCO_EMA/train2017

MODEL=stabilityai/sdxl-vae

python vae_offline.py \
	--input $INPUT \
	--output $OUTPUT \
	--model $MODEL
 
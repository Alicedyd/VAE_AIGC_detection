# INPUT=/root/autodl-tmp/AIGC_data/MSCOCO/val2017
INPUT=/root/autodl-tmp/AIGC_data/MSCOCO/train2017

# OUTPUT=/root/autodl-tmp/AIGC_data/MSCOCO_XL/val2017
OUTPUT=/root/autodl-tmp/AIGC_data/MSCOCO_MSE/train2017

MODEL=stabilityai/sd-vae-ft-mse

python vae_offline.py \
	--input $INPUT \
	--output $OUTPUT \
	--model $MODEL
 
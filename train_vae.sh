python train.py \
    --name drct_detection_dino_vae_multi \
    --real_list_path /root/autodl-tmp/AIGC_data/MSCOCO/ \
    --data_mode=mscoco \
    --arch=DINOv2-LoRA:dinov2_vitl14 \
    --fix_backbone \
    --batch_size=32 \
    --lr=0.0005 \
    --optim=adam \
    --niter=1 \
    --gpu_ids='0' \
    --lora_rank=8 \
    --lora_alpha=1

# --fake_list_path /root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-xl-base-1.0/ \

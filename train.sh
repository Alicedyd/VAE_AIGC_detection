python train.py \
    --name genimage_detection_clip \
    --wang2020_data_path=/root/autodl-tmp/AIGC_data/GenImage \
    --data_mode=multi_wang2020 \
    --arch=CLIP-LoRA:ViT-L/14 \
    --fix_backbone \
    --batch_size=32 \
    --lr=0.0005 \
    --optim=adam \
    --niter=1 \
    --gpu_ids='2' \
    --lora_rank=8 \
    --lora_alpha=1 



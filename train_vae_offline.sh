python train.py \
    --name debug \
    --real_list_path /root/autodl-tmp/AIGC_data/MSCOCO\
    --fake_list_path /root/autodl-tmp/AIGC_data/MSCOCO_XL,/root/autodl-tmp/AIGC_data/MSCOCO_EMA,/root/autodl-tmp/AIGC_data/MSCOCO_MSE \
    --data_mode=mscoco \
    --arch=DINOv2-LoRA:dinov2_vitl14 \
    --fix_backbone \
    --batch_size=16 \
    --lr=0.0005 \
    --optim=adam \
    --niter=1 \
    --gpu_ids='2' \
    --lora_rank=8 \
    --lora_alpha=1 \
    --contrastive

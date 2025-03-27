python train.py \
    --name genimage_detection_clip \
    --wang2020_data_path=/data2/jingyi/UniversalFakeDetect-main/datasets \
    --data_mode=multi_wang2020 \
    --arch=CLIP-LoRA:ViT-L/14 \
    --fix_backbone \
    --batch_size=64 \
    --lr=0.001 \
    --optim=adam \
    --niter=1 \
    --gpu_ids='5' \
    --lora_rank=8 \
    --lora_alpha=16 



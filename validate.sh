python validate.py \
    --data_mode=ours \
    --arch=DINOv2-LoRA:dinov2_vitl14 \
    --real_path=/data2/jingyi/dataset/Chameleon/test/0_real \
    --fake_path=/data2/jingyi/dataset/Chameleon/test/1_fake \
    --key='chameleon' \
    --ckpt='./checkpoints/genimage_detection/model_iters_5000.pth' 
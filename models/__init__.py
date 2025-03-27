from .clip_models import CLIPModel
from .imagenet_models import ImagenetModel
from .dinov2_models import DINOv2Model
from .clip_models_lora import CLIPModelWithLoRA
from .dinov2_models_lora import DINOv2ModelWithLoRA

VALID_NAMES = [
    'Imagenet:resnet18',
    'Imagenet:resnet34',
    'Imagenet:resnet50',
    'Imagenet:resnet101',
    'Imagenet:resnet152',
    'Imagenet:vgg11',
    'Imagenet:vgg19',
    'Imagenet:swin-b',
    'Imagenet:swin-s',
    'Imagenet:swin-t',
    'Imagenet:vit_b_16',
    'Imagenet:vit_b_32',
    'Imagenet:vit_l_16',
    'Imagenet:vit_l_32',

    'CLIP:RN50', 
    'CLIP:RN101', 
    'CLIP:RN50x4', 
    'CLIP:RN50x16', 
    'CLIP:RN50x64', 
    'CLIP:ViT-B/32', 
    'CLIP:ViT-B/16', 
    'CLIP:ViT-L/14', 
    'CLIP:ViT-L/14@336px',
    
    'DINOv2:dinov2_vits14',
    'DINOv2:dinov2_vitb14',
    'DINOv2:dinov2_vitl14',
    'DINOv2:dinov2_vitg14',
    
    'CLIP-LoRA:RN50',
    'CLIP-LoRA:RN101',
    'CLIP-LoRA:ViT-B/32', 
    'CLIP-LoRA:ViT-B/16', 
    'CLIP-LoRA:ViT-L/14',
    
    'DINOv2-LoRA:dinov2_vits14',
    'DINOv2-LoRA:dinov2_vitb14',
    'DINOv2-LoRA:dinov2_vitl14',
    'DINOv2-LoRA:dinov2_vitg14',
]





def get_model(name, lora_rank=8, lora_alpha=1.0, lora_targets=None):
    assert name in VALID_NAMES
    if name.startswith("Imagenet:"):
        return ImagenetModel(name[9:]) 
    elif name.startswith("CLIP:"):
        return CLIPModel(name[5:])  
    elif name.startswith("CLIP-LoRA:"):
        return CLIPModelWithLoRA(name[10:], lora_rank=lora_rank, lora_alpha=lora_alpha, lora_targets=lora_targets)
    elif name.startswith("DINOv2:"):
        return DINOv2Model(name[7:])
    elif name.startswith("DINOv2-LoRA:"):
        return DINOv2ModelWithLoRA(name[12:], lora_rank=lora_rank, lora_alpha=lora_alpha, lora_targets=lora_targets)
    else:
        assert False 

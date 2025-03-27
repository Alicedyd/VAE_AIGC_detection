from .clip import clip 
from PIL import Image
import torch.nn as nn
from .lora import apply_lora_to_linear_layers, get_lora_params

CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768
}

class CLIPModelWithLoRA(nn.Module):
    def __init__(self, name, num_classes=1, lora_rank=4, lora_alpha=1.0, lora_targets=None):
        super(CLIPModelWithLoRA, self).__init__()

        
        self.model, self.preprocess = clip.load(name, device="cpu")
        self.fc = nn.Linear(CHANNELS[name], num_classes)
        
        
        if lora_targets is None:
            
            if 'ViT' in name:  
                lora_targets = ['attn.in_proj', 'attn.out_proj', 'mlp.c_fc', 'mlp.c_proj']
            else:  
                lora_targets = ['attnpool']
        
        
        self.model.visual = apply_lora_to_linear_layers(
            self.model.visual, 
            rank=lora_rank, 
            alpha=lora_alpha,
            target_modules=lora_targets
        )
    
    def get_trainable_params(self):
        
        lora_params = get_lora_params(self.model.visual)
        fc_params = self.fc.parameters()
        return list(lora_params) + list(fc_params)

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x) 
        if return_feature:
            return features
        return self.fc(features)
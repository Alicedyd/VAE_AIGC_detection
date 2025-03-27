import os
import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.transforms import Normalize

from functools import partial
try:
    from dinov2.models.vision_transformer import (
        vit_small, vit_base, vit_large, vit_giant2,
        DinoVisionTransformer
    )
    HAS_OFFICIAL_DINOV2 = True
except ImportError:
    HAS_OFFICIAL_DINOV2 = False
    print("Warning: Official DINOv2 package not found. Installing or providing DINOv2 code is recommended.")
    print("You can install it with: pip install dinov2")
    print("Or clone from: https://github.com/facebookresearch/dinov2")
    
    try:
        from dinov2.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
        HAS_OFFICIAL_DINOV2 = True
    except ImportError:
        pass

DINOV2_MODELS = {
    "dinov2_vits14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",
    "dinov2_vitb14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",
    "dinov2_vitl14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",
    "dinov2_vitg14": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth",
}

CHANNELS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}

MODEL_FACTORY = {
    "dinov2_vits14": vit_small if HAS_OFFICIAL_DINOV2 else None,
    "dinov2_vitb14": vit_base if HAS_OFFICIAL_DINOV2 else None,
    "dinov2_vitl14": vit_large if HAS_OFFICIAL_DINOV2 else None,
    "dinov2_vitg14": vit_giant2 if HAS_OFFICIAL_DINOV2 else None,
}

class DINOv2Model(nn.Module):

    def __init__(self, name, num_classes=1, local_files_only=False, model_dir=None):
        super(DINOv2Model, self).__init__()
        
        if name not in CHANNELS:
            raise ValueError(f"Unknown DINOv2 model: {name}. avaliable models: {list(CHANNELS.keys())}")
        
        self.model_type = name
        
        if local_files_only and model_dir:
            model_path = os.path.join(model_dir, f"{name}_pretrain.pth")
            if not os.path.exists(model_path):
                raise ValueError(f"local file does not exist: {model_path}")
            
            print(f"load DINOv2 from local file: {model_path}")
            self._create_model()
            self._load_local_weights(model_path)
        else:
            try:
                import torch.hub
                print(f"load DINOv2 from hub: {name}")
                self.model = torch.hub.load('facebookresearch/dinov2', name)
            except Exception as e:
                print(f"Can not load DINOv2 from hub: {e}")
                
                self._create_model()
                self._load_url_weights(DINOV2_MODELS[name])
        
        self.fc = nn.Linear(CHANNELS[name], num_classes)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        self.normalize = Normalize(mean=self.mean, std=self.std)

    def _create_model(self):
        if not HAS_OFFICIAL_DINOV2:
            raise ImportError(
                "doen't have official dinov2,"
                "using pip install dinov2，or clone code from https://github.com/facebookresearch/dinov2"
            )
        
        model_fn = MODEL_FACTORY[self.model_type]
        if model_fn is None:
            raise ValueError(f"unsupport model type: {self.model_type}")
        
        self.model = model_fn(patch_size=14)
    
    def _load_local_weights(self, model_path):
        state_dict = torch.load(model_path, map_location='cpu')
        if 'model' in state_dict:
            state_dict = state_dict['model']
        self._load_state_dict(state_dict)
    
    def _load_url_weights(self, url):
        state_dict = load_state_dict_from_url(url, map_location='cpu', progress=True)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        self._load_state_dict(state_dict)
    
    def _load_state_dict(self, state_dict):
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys:
            print(f"warning: missing keys when loading model weights: {missing_keys}")
        if unexpected_keys:
            print(f"warning: unexpected keys when loading model weights: {unexpected_keys}")
    
    def forward(self, x, return_feature=False):

        if hasattr(self.model, 'forward_features'):
            features_dict = self.model.forward_features(x)
            features = features_dict['x_norm_clstoken']
        else:

            features = self.model(x)       
      
            if isinstance(features, dict):
                features = features.get('x_norm_clstoken', features.get('last_hidden_state', None)[:, 0])
            
        if return_feature:
            return features
        
        
        return self.fc(features)
    
    def get_preprocessing_transforms(self):
        from torchvision import transforms
        
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
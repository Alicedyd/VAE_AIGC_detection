from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .vision_transformer import vit_b_16, vit_b_32, vit_l_16, vit_l_32

from torchvision import transforms
from PIL import Image
import torch 
import torch.nn as nn 


model_dict = {
    'resnet18': resnet18,
    'resnet34': resnet34,
    'resnet50': resnet50,
    'resnet101': resnet101,
    'resnet152': resnet152,
    'vit_b_16': vit_b_16,
    'vit_b_32': vit_b_32,
    'vit_l_16': vit_l_16,
    'vit_l_32': vit_l_32
}


CHANNELS = {
    "resnet18" : 512,
    "resnet50" : 2048,
    "vit_b_16" : 768,
}



class ImagenetModel(nn.Module):
    def __init__(self, name, num_classes=1):
        super(ImagenetModel, self).__init__()

        self.model = model_dict[name](pretrained=True)
        self.fc = nn.Linear(CHANNELS[name], num_classes) #manually define a fc layer here
        

    def forward(self, x, return_feature=False):
        feature = self.model(x)["penultimate"]

        if return_feature:
            return feature, self.fc(feature)
        
        return self.fc(feature)

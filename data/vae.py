import torch
import torch.nn as nn
from diffusers import AutoencoderKL
from torchvision import transforms
from PIL import Image
import numpy as np


class DoNothing(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

    def __call__(self, x):
        return self.forward(x)

# class VAETransform(nn.Module):
#     """
#     使用SDXL的VAE对图像进行编码和解码的变换
#     """
#     def __init__(self, gpu_id, vae_model_path="stabilityai/sdxl-vae"):
#         super().__init__()
#         self.vae = None
#         self.vae_model_path = vae_model_path
#         self.gpu_id = gpu_id
        
#     def _load_vae(self):
#         """延迟加载VAE模型，只在第一次使用时加载"""
#         if self.vae is None:
#             print(f"Loading VAE model from {self.vae_model_path}")
#             self.vae = AutoencoderKL.from_pretrained(
#                 self.vae_model_path, 
#                 torch_dtype=torch.float16
#             )
#             self.vae.cuda(self.gpu_id)
#             self.vae.eval()

#     def forward(self, x):
#         """
#         使用VAE编码和解码图像张量
        
#         Args:
#             x (torch.Tensor): 输入图像张量 [C, H, W]，范围[0, 1]
            
#         Returns:
#             torch.Tensor: 经过VAE编码解码后的图像张量
#         """
#         self._load_vae()
        
#         # 添加batch维度
#         x_batch = x.unsqueeze(0).cuda(self.gpu_id)
        
#         # VAE期望输入范围为[-1, 1]
#         x_batch = 2 * x_batch - 1
        
#         with torch.no_grad():
#             # 编码
#             latents = self.vae.encode(x_batch).latent_dist.sample()
#             latents = latents * self.vae.config.scaling_factor
            
#             # 解码
#             decoded = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        
#         # 转换回[0, 1]范围并移除batch维度
#         decoded = (decoded + 1) / 2
#         decoded = decoded.squeeze(0).cpu()
        
#         return decoded

#     def __call__(self, x):
#         return self.forward(x)

class SingletonVAE:
    """单例模式的VAE模型，确保全局只有一个VAE实例，并注意类型一致性"""
    _instance = None
    
    @classmethod
    def get_instance(cls, gpu_id, vae_model_path="stabilityai/sdxl-vae"):
        if cls._instance is None:
            print(f"Loading VAE model from {vae_model_path}")
            vae = AutoencoderKL.from_pretrained(
                vae_model_path, 
                torch_dtype=torch.float32
            )
            vae.cuda(gpu_id)
            vae.eval()
            cls._instance = vae
        return cls._instance


class VAETransform:
    """
    使用SDXL的VAE对图像进行编码和解码的变换
    接收PIL Image输入，返回PIL Image输出
    """
    def __init__(self, gpu_id, vae_model_path="stabilityai/sdxl-vae"):
        self.vae = None
        self.vae_model_path = vae_model_path
        self.gpu_id = gpu_id
        # 用于将PIL Image转换为tensor的变换
        self.to_tensor = transforms.ToTensor()
        # 用于将tensor转换回PIL Image的变换
        self.to_pil = transforms.ToPILImage()

    def _set_gpu_id(self, gpu_id):
        self.gpu_id = gpu_id

    def __call__(self, img):
        """
        使用VAE编码和解码图像
        
        Args:
            img (PIL.Image): 输入PIL图像
            
        Returns:
            PIL.Image: 经过VAE编码解码后的PIL图像
        """
        self.vae = SingletonVAE.get_instance(self.gpu_id, self.vae_model_path)
        
        # 转换PIL图像为tensor [C, H, W]
        x = self.to_tensor(img)
        
        # 添加batch维度并移至设备
        x_batch = x.unsqueeze(0).cuda(self.gpu_id)
        
        # VAE期望输入范围为[-1, 1]
        x_batch = 2 * x_batch - 1

        # 确保输入tensor的类型和VAE模型匹配
        model_dtype = next(self.vae.parameters()).dtype
        x_batch = x_batch.to(dtype=model_dtype)
        
        with torch.no_grad():
            # 编码
            latents = self.vae.encode(x_batch).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
            
            # 解码
            decoded = self.vae.decode(latents / self.vae.config.scaling_factor).sample
        
        # 转换回[0, 1]范围并移除batch维度
        decoded = (decoded + 1) / 2
        decoded = decoded.squeeze(0).cpu()
        
        # 转换回PIL图像
        pil_img = self.to_pil(decoded)
        
        return pil_img
    
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
        
class VAERebuilder:
    """VAE重建器，用于批量处理图像文件夹"""
     
    def __init__(self, gpu_id, vae_model_path="stabilityai/sdxl-vae"):
        """
        初始化VAE重建器
        
        Args:
            vae_model_path: VAE模型路径或名称
            use_fp16: 是否使用半精度浮点数
            device: 指定设备，None时自动选择
        """
        self.vae_model_path = vae_model_path
        self.gpu_id = gpu_id
        self.vae = None
        
    def _load_vae(self):
        """加载VAE模型，仅首次使用时加载"""
        if self.vae is None:
            print(f"正在加载VAE模型 '{self.vae_model_path}'...")
            
            self.vae = AutoencoderKL.from_pretrained(
                self.vae_model_path,
                torch_dtype=torch.float32
            )
            self.vae.cuda(self.gpu_id)
            self.vae.eval()

            self.device = self.vae.device
    
    def __call__(self, batch_img, output_path=None):
        self._load_vae()
        
        try:
            results = []
            for img in batch_img:

                # 转换为tensor
                x = torch.from_numpy(np.array(img)).float() / 255.0
                x = x.permute(2, 0, 1).unsqueeze(0)
                
                # 保证数据类型匹配
                model_dtype = next(self.vae.parameters()).dtype
                x = x.to(device=self.device, dtype=model_dtype)
                
                # VAE处理
                with torch.no_grad():
                    # 转换到 [-1, 1] 范围
                    x = 2.0 * x - 1.0
                    
                    # 编码
                    latents = self.vae.encode(x).latent_dist.sample()
                    latents = latents * self.vae.config.scaling_factor
                    
                    # 解码
                    decoded = self.vae.decode(latents / self.vae.config.scaling_factor).sample
                    
                    # 转换回 [0, 1] 范围
                    decoded = (decoded + 1.0) / 2.0
                
                # 转换回PIL图像
                decoded = decoded.to(dtype=torch.float32).squeeze(0).permute(1, 2, 0).cpu().numpy()
                decoded = (decoded * 255).clip(0, 255).astype(np.uint8)
                rebuilt_img = Image.fromarray(decoded)

                results.append(rebuilt_img)

            return results
                
        except Exception as e:
            print(f"处理图像出错: {e}")
import os
import time
import torch
import argparse
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from diffusers import AutoencoderKL
import numpy as np


class VAERebuilder:
    """VAE重建器，用于批量处理图像文件夹"""
    
    def __init__(self, vae_model_path="stabilityai/sdxl-vae", use_fp16=True, device=None):
        """
        初始化VAE重建器
        
        Args:
            vae_model_path: VAE模型路径或名称
            use_fp16: 是否使用半精度浮点数
            device: 指定设备，None时自动选择
        """
        self.vae_model_path = vae_model_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.vae = None
        
    def _load_vae(self):
        """加载VAE模型，仅首次使用时加载"""
        if self.vae is None:
            print(f"正在加载VAE模型 '{self.vae_model_path}'...")
            start_time = time.time()
            
            self.vae = AutoencoderKL.from_pretrained(
                self.vae_model_path,
                torch_dtype=torch.float16 if self.use_fp16 else torch.float32
            )
            self.vae.to(self.device)
            self.vae.eval()
            
            load_time = time.time() - start_time
            print(f"VAE模型加载完成！耗时 {load_time:.2f} 秒")
    
    def rebuild_image(self, input_path, output_path=None):
        """
        对单张图像进行VAE重建
        
        Args:
            input_path: 输入图像路径
            output_path: 输出图像路径，None时返回PIL图像对象
            
        Returns:
            如果output_path为None，返回重建后的PIL图像对象
        """
        self._load_vae()
        
        try:
            # 加载图像
            img = Image.open(input_path).convert("RGB")
            
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
            
            # 保存或返回
            if output_path:
                output_dir = os.path.dirname(output_path)
                if output_dir:  # 如果输出路径包含目录
                    os.makedirs(output_dir, exist_ok=True)
                rebuilt_img.save(output_path)
                return True
            else:
                return rebuilt_img
                
        except Exception as e:
            print(f"处理图像出错 {input_path}: {e}")
            return False if output_path else None


def find_images(folder, extensions, recursive=True):
    """
    在文件夹中查找指定扩展名的图像文件
    
    Args:
        folder: 文件夹路径
        extensions: 扩展名列表
        recursive: 是否递归查找子文件夹
        
    Returns:
        图像文件路径列表
    """
    folder = os.path.abspath(folder)  # 转换为绝对路径
    print(f"查找图像文件：{folder}")
    print(f"支持的扩展名：{extensions}")
    
    if not os.path.exists(folder):
        print(f"错误: 文件夹 '{folder}' 不存在!")
        return []
    
    if not os.path.isdir(folder):
        print(f"错误: '{folder}' 不是文件夹!")
        return []
    
    # 准备用于匹配的扩展名（包括大小写）
    all_extensions = []
    for ext in extensions:
        ext = ext.lower().strip('.')
        all_extensions.append(ext.lower())
        all_extensions.append(ext.upper())
    
    # 方法1: 使用os.walk遍历文件夹
    image_paths = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            # 检查文件扩展名
            file_ext = os.path.splitext(file)[1].strip('.')
            if file_ext.lower() in all_extensions:
                image_paths.append(os.path.join(root, file))
        
        # 如果不递归，则跳出循环
        if not recursive:
            break
    
    print(f"找到图像文件数量: {len(image_paths)}")
    
    # 如果没有找到图像，打印更详细的信息
    if len(image_paths) == 0:
        print("调试信息:")
        print(f"  文件夹内容: {os.listdir(folder)}")
        print(f"  扩展名检查: {all_extensions}")
        
        # 尝试找出问题
        for root, dirs, files in os.walk(folder):
            if files:
                print(f"  '{root}' 中有文件: {files[:5]}" + ("..." if len(files) > 5 else ""))
                for file in files[:3]:
                    file_ext = os.path.splitext(file)[1]
                    print(f"    文件 '{file}' 扩展名: '{file_ext}'")
            if not recursive:
                break
    
    return image_paths


def rebuild_folder(input_folder, output_folder, vae_model="stabilityai/sdxl-vae", 
                  use_fp16=True, extensions=["jpg", "jpeg", "png", "bmp"],
                  num_threads=4, device=None, recursive=True, skip_existing=True):
    """
    对整个文件夹中的图像进行VAE重建
    
    Args:
        input_folder: 输入文件夹路径
        output_folder: 输出文件夹路径
        vae_model: VAE模型名称或路径
        use_fp16: 是否使用半精度浮点数
        extensions: 要处理的图像扩展名列表
        num_threads: 处理线程数
        device: 计算设备，None时自动选择
        recursive: 是否递归处理子文件夹
        skip_existing: 是否跳过已存在的输出文件
        
    Returns:
        处理成功的图像数量
    """
    # 确保输入输出路径是绝对路径
    input_folder = os.path.abspath(input_folder)
    output_folder = os.path.abspath(output_folder)
    
    print(f"输入文件夹: {input_folder}")
    print(f"输出文件夹: {output_folder}")
    
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    
    # 查找所有要处理的图像
    image_paths = find_images(input_folder, extensions, recursive)
    
    if not image_paths:
        print("没有找到需要处理的图像文件，任务终止。")
        return 0
    
    # 创建VAE实例
    rebuilder = VAERebuilder(vae_model_path=vae_model, use_fp16=use_fp16, device=device)
    
    # 计算输出路径并筛选已处理文件
    tasks = []
    input_folder_len = len(input_folder) + 1  # 加1是为了去掉路径分隔符
    
    for path in image_paths:
        # 计算相对路径
        rel_path = path[input_folder_len:] if path.startswith(input_folder) else os.path.basename(path)
        out_path = os.path.join(output_folder, rel_path.split('.')[0] + ".png")
        
        # 如果需要跳过已存在的文件
        if skip_existing and os.path.exists(out_path):
            continue
            
        tasks.append((path, out_path))
    
    print(f"将处理 {len(tasks)} 个图像，跳过 {len(image_paths) - len(tasks)} 个已处理图像")
    if len(tasks) == 0:
        print("没有需要处理的图像，任务完成！")
        return 0
    
    # 定义线程工作函数
    def process_image(args):
        in_path, out_path = args
        return rebuilder.rebuild_image(in_path, out_path)
    
    # 使用线程池处理任务
    start_time = time.time()
    success_count = 0
    
    # 调整线程数，不超过任务数
    num_threads = min(num_threads, len(tasks))
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 使用tqdm显示进度
        for result in tqdm(executor.map(process_image, tasks), total=len(tasks), desc="处理图像"):
            if result:
                success_count += 1
    
    # 计算统计信息
    elapsed_time = time.time() - start_time
    failed_count = len(tasks) - success_count
    
    print(f"\n处理完成!")
    print(f"总图像数: {len(tasks)}")
    print(f"成功处理: {success_count}")
    print(f"处理失败: {failed_count}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    if success_count > 0:
        print(f"平均速度: {success_count / elapsed_time:.2f} 图像/秒")
    
    return success_count


# 如果作为脚本直接运行
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='对文件夹中的图像进行VAE重建')
    parser.add_argument('--input', type=str, required=True, help='输入图像文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出图像文件夹路径')
    parser.add_argument('--model', type=str, default='stabilityai/sdxl-vae', help='VAE模型名称或路径')
    parser.add_argument('--fp16', action='store_true', help='使用半精度浮点数(FP16)')
    parser.add_argument('--threads', type=int, default=4, help='处理线程数')
    parser.add_argument('--extensions', type=str, default='jpg,jpeg,png', help='要处理的图像扩展名,以逗号分隔')
    parser.add_argument('--no-recursive', action='store_true', help='不递归处理子文件夹')
    parser.add_argument('--force', action='store_true', help='强制重新处理已存在的文件')
    
    args = parser.parse_args()
    
    # 处理扩展名
    extensions = [ext.strip() for ext in args.extensions.split(',')]
    
    # 调用函数
    rebuild_folder(
        input_folder=args.input,
        output_folder=args.output,
        vae_model=args.model,
        use_fp16=args.fp16,
        extensions=extensions,
        num_threads=args.threads,
        recursive=not args.no_recursive,
        skip_existing=not args.force
    )
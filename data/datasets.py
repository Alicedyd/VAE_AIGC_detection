import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import random
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from PIL import ImageDraw
from scipy.ndimage.filters import gaussian_filter
import pickle
import os 
from skimage.io import imread
from copy import deepcopy
import torch

from tqdm import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform

ImageFile.LOAD_TRUNCATED_IMAGES = True


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

class MedianBlur:
    """Apply median blur to image.
    
    Args:
        kernel_size (int): Size of the median filter. Must be odd and positive.
    """
    
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1  # Ensure kernel size is odd
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be blurred.
            
        Returns:
            PIL Image: Blurred image.
        """
        # Convert PIL to numpy array
        img_np = np.array(img)
        
        # Apply median blur
        blurred = cv2.medianBlur(img_np, self.kernel_size)
        
        # Convert back to PIL Image
        return Image.fromarray(blurred)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(kernel_size={self.kernel_size})'


class RandomErasing:
    """Randomly erase rectangular regions from an image.
    
    Args:
        p (float): Probability of applying the transform. Default: 0.5.
        scale (tuple): Range of area of the erased region relative to the image area.
                      Default: (0.02, 0.33).
        ratio (tuple): Range of aspect ratio of the erased region. Default: (0.3, 3.3).
        value (int, tuple): Value used to fill the erased region. Default: 0.
    """
    
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be erased.
            
        Returns:
            PIL Image: Erased image.
        """
        if random.random() > self.p:
            return img
        
        # Convert PIL to numpy array
        img_np = np.array(img)
        height, width = img_np.shape[:2]
        
        # Calculate area of image
        img_area = height * width
        
        # Get random area to erase
        for _ in range(10):  # Try 10 times to find valid parameters
            erase_area_ratio = random.uniform(self.scale[0], self.scale[1])
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            # Calculate target area
            erase_area = int(img_area * erase_area_ratio)
            
            # Calculate target width and height
            h = int(np.sqrt(erase_area / aspect_ratio))
            w = int(np.sqrt(erase_area * aspect_ratio))
            
            # Ensure width and height are within image bounds
            if w < width and h < height:
                # Random position
                x = random.randint(0, width - w)
                y = random.randint(0, height - h)
                
                # Erase the region
                if isinstance(self.value, (int, float)):
                    # Single value for all channels
                    if len(img_np.shape) == 3:  # Color image
                        img_np[y:y+h, x:x+w, :] = self.value
                    else:  # Grayscale image
                        img_np[y:y+h, x:x+w] = self.value
                else:
                    # Different values for each channel
                    if len(img_np.shape) == 3:  # Color image
                        for i, val in enumerate(self.value):
                            if i < img_np.shape[2]:
                                img_np[y:y+h, x:x+w, i] = val
                    else:  # Grayscale image
                        img_np[y:y+h, x:x+w] = self.value[0]
                
                break
        
        # Convert back to PIL Image
        return Image.fromarray(img_np)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p}, scale={self.scale}, ratio={self.ratio}, value={self.value})'
    
    
class RandomSharpen:
    """Apply random sharpening to image.
    
    Args:
        p (float): Probability of applying the transform. Default: 0.1.
        factor (float or tuple): Sharpening factor. If tuple (min, max), the factor will 
                          be randomly chosen between min and max. Default: (1.0, 3.0).
    """
    
    def __init__(self, p=0.1, factor=(1.0, 3.0)):
        self.p = p
        self.factor = factor
        if isinstance(factor, (list, tuple)):
            assert len(factor) == 2, "factor should be a tuple of (min, max)"
            self.min_factor, self.max_factor = factor
        else:
            self.min_factor = self.max_factor = factor
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be sharpened.
            
        Returns:
            PIL Image: Sharpened image.
        """
        if random.random() > self.p:
            return img
        
        # Convert PIL to numpy array
        img_np = np.array(img).astype(np.float32)
        
        # Create blurred version for unsharp mask
        blurred = cv2.GaussianBlur(img_np, (0, 0), 3.0)
        
        # Determine sharpening factor for this application
        factor = random.uniform(self.min_factor, self.max_factor)
        
        # Apply unsharp mask formula: original + factor * (original - blurred)
        sharpened = img_np + factor * (img_np - blurred)
        
        # Clip values to valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        return Image.fromarray(sharpened)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p}, factor=({self.min_factor}, {self.max_factor}))'
    
class RandomPepperNoise:
    """Apply pepper noise to an image.
    
    Args:
        p (float): Probability of applying the transform. Default: 0.1.
        noise_ratio (float or tuple): Ratio of pixels to be replaced with noise. 
                             If tuple (min, max), the ratio will be randomly chosen 
                             between min and max. Default: (0.01, 0.05).
    """
    
    def __init__(self, p=0.1, noise_ratio=(0.01, 0.05)):
        self.p = p
        self.noise_ratio = noise_ratio
        if isinstance(noise_ratio, (list, tuple)):
            assert len(noise_ratio) == 2, "noise_ratio should be a tuple of (min, max)"
            self.min_ratio, self.max_ratio = noise_ratio
        else:
            self.min_ratio = self.max_ratio = noise_ratio
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be applied with pepper noise.
            
        Returns:
            PIL Image: Image with pepper noise.
        """
        if random.random() > self.p:
            return img
        
        # Convert PIL to numpy array
        img_np = np.array(img)
        height, width = img_np.shape[:2]
        
        # Determine noise ratio for this particular application
        noise_ratio = random.uniform(self.min_ratio, self.max_ratio)
        
        # Calculate number of pixels to replace
        n_pixels = int(height * width * noise_ratio)
        
        # Generate random coordinates
        y_coords = np.random.randint(0, height, n_pixels)
        x_coords = np.random.randint(0, width, n_pixels)
        
        # Apply pepper noise (black pixels)
        if len(img_np.shape) == 3:  # Color image
            img_np[y_coords, x_coords, :] = 0
        else:  # Grayscale image
            img_np[y_coords, x_coords] = 0
        
        # Convert back to PIL Image
        return Image.fromarray(img_np)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p}, noise_ratio=({self.min_ratio}, {self.max_ratio}))'
    
class MotionBlur:
    """Apply motion blur to image by applying a directional filter.
    
    Args:
        kernel_size (int): Size of the motion blur kernel. Must be odd.
        angle (float, optional): Angle of motion blur in degrees. If None, 
                                a random angle will be chosen.
    """
    
    def __init__(self, kernel_size=5, angle=None):
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1  # Ensure kernel size is odd
        self.angle = angle
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be blurred.
            
        Returns:
            PIL Image: Motion blurred image.
        """
        # Convert PIL to numpy array
        img_np = np.array(img)
        
        # Choose a random angle if none is specified
        angle = self.angle
        if angle is None:
            angle = np.random.uniform(0, 180)
        
        # Create motion blur kernel
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        
        # Calculate x, y coordinates for the line
        for i in range(self.kernel_size):
            offset = i - center
            x = int(center + np.round(offset * np.cos(angle_rad)))
            y = int(center + np.round(offset * np.sin(angle_rad)))
            
            if 0 <= x < self.kernel_size and 0 <= y < self.kernel_size:
                kernel[y, x] = 1
        
        # Normalize the kernel
        kernel = kernel / np.sum(kernel)
        
        # Apply the kernel to each channel
        if len(img_np.shape) == 3:  # Color image
            blurred = np.zeros_like(img_np)
            for c in range(img_np.shape[2]):
                blurred[:, :, c] = cv2.filter2D(img_np[:, :, c], -1, kernel)
        else:  # Grayscale image
            blurred = cv2.filter2D(img_np, -1, kernel)
            
        # Convert back to PIL Image
        return Image.fromarray(blurred)
    
class RandomPure:
    """随机将图像的一部分转换为纯色矩形"""
    def __init__(self, p=0.01, min_size=0.1, max_size=0.5):
        """
        初始化
        
        参数:
            p (float): 应用转换的概率
            min_size (float): 纯色区域最小尺寸比例(相对于图像尺寸)
            max_size (float): 纯色区域最大尺寸比例(相对于图像尺寸)
        """
        self.p = p
        self.min_size = min_size
        self.max_size = max_size
    
    def _generate_random_color(self):
        """生成随机RGB颜色"""
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    
    def _get_random_rectangle_params(self, img_width, img_height):
        """获取随机矩形的参数"""
        # 确定矩形尺寸
        width_ratio = random.uniform(self.min_size, self.max_size)
        height_ratio = random.uniform(self.min_size, self.max_size)
        
        rect_width = int(img_width * width_ratio)
        rect_height = int(img_height * height_ratio)
        
        # 随机位置(确保矩形完全在图像内)
        x1 = random.randint(0, img_width - rect_width)
        y1 = random.randint(0, img_height - rect_height)
        x2 = x1 + rect_width
        y2 = y1 + rect_height
        
        return (x1, y1, x2, y2)
    
    def __call__(self, img):
        """应用变换"""
        if random.random() < self.p:
            # 获取图像尺寸
            width, height = img.size
            
            # 生成随机颜色
            color = self._generate_random_color()
            
            # 获取随机矩形参数
            bbox = self._get_random_rectangle_params(width, height)
            
            # 创建一个与原图像相同的副本
            result = img.copy()
            draw = ImageDraw.Draw(result)
            
            # 绘制随机颜色的矩形
            draw.rectangle(bbox, fill=color)
            
            return result
        else:
            return img

class RandomGaussianNoise:
    """为PIL图像添加高斯噪声的转换"""
    def __init__(self, mean=0, std=35, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            # 将PIL图像转换为numpy数组
            img_array = np.array(img).astype(np.float32)
            
            # 生成噪声
            noise = np.random.normal(self.mean, self.std, img_array.shape)
            
            # 添加噪声
            noisy_img = img_array + noise
            
            # 裁剪到有效的像素值范围
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            
            # 转回PIL图像
            return Image.fromarray(noisy_img)
        return img
    
class RandomJPEGCompression:
    def __init__(self, quality_lower=30, quality_upper=95, p=0.3):
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(self.quality_lower, self.quality_upper)
            out = BytesIO()
            img.save(out, format='jpeg', quality=quality)
            out.seek(0)
            img = Image.open(out)
            return img
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + f'(quality_lower={self.quality_lower}, quality_upper={self.quality_upper}, p={self.p})'
    

class RandomResizedCropWithVariableSize(transforms.RandomResizedCrop):
    def __init__(self, min_size, max_size, scale=(0.08, 1.0), ratio=(1.0, 1.0), interpolation=transforms.InterpolationMode.BILINEAR):
        self.min_size = min_size
        self.max_size = max_size
        super().__init__(size=min_size, scale=scale, ratio=ratio, interpolation=interpolation)
    
    def get_random_size(self):
        """Return a random size between min_size and max_size."""
        size = random.randint(self.min_size, self.max_size)
        return size

    def __call__(self, img):
        size = img.size 
        size = tuple(int(element * 0.54) for element in size)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        ret =  F.resized_crop(img, i, j, h, w, size, self.interpolation, antialias=self.antialias)
        return ret


def create_train_transforms(size=224, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), is_crop=True):
    """创建训练数据增强转换管道"""
    
    # 根据是否裁剪选择不同的大小调整方法
    if is_crop:
        resize_func = PadRandomCrop(size)
    else:
        resize_func = transforms.Resize(size)
    
    # 构建转换列表
    transform_list = [
        # 千万不要在这里加随机JPEG压缩，加上这个会导致 real acc 大幅下降
        # RandomJPEGCompression(quality_lower=30, quality_upper=90, p=0.1), 

        # 随机水平翻转
        transforms.RandomHorizontalFlip(),
        # 随机竖直翻转
        transforms.RandomVerticalFlip(),
        
        # 添加高斯噪声
        RandomGaussianNoise(p=0.1),
        # 添加椒盐噪声
        RandomPepperNoise(p=0.1),
        
        # 随机模糊组合
        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.GaussianBlur(kernel_size=3),
                transforms.GaussianBlur(kernel_size=5),
                MedianBlur(kernel_size=3),
                MotionBlur(kernel_size=5)
            ])
        ], p=0.2),
        
        # 随机锐化
        RandomSharpen(p=0.1),
        
        # 随机应用颜色变换（亮度、对比度等）
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15)
        ], p=0.5),
        
        # 随机转为灰度图
        transforms.RandomGrayscale(p=0.2),
        
        # # 随机遮罩部分区域
        # RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),

        # 调整大小
        resize_func,

        # 转为张量
        transforms.ToTensor(),
        
        # 标准化
        transforms.Normalize(mean=mean, std=std),
    ]
    
    # 创建转换组合
    return transforms.Compose(transform_list)


def data_augment(img, opt):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random.random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random.random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return F.resize(img, opt.loadSize, interpolation=rz_dict[interp])



def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    out = [] 
    for r, d, f in os.walk(rootdir, followlinks=True):
        for file in f:
            if (file.split('.')[1].lower() in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list




class CustomBatchSampler:
    def __init__(self, opt, vae_model, transform_funcs):
        temp = 'train' if opt.data_label == 'train' else 'val'
        self.real_list = get_list(opt.real_list_path)
        self.fake_list = self.real_list.copy()
        self.vae_model = vae_model
        self.transform_funcs = transform_funcs

        self.fake_num = len(self.vae_model)

        self.vae_transform_funcs_list = []
        for vae in vae_model:
            vae_transform_funcs = transform_funcs.copy()
            vae_transform_funcs.insert(1, vae)
            self.vae_transform_funcs_list.append(vae_transform_funcs)
        
        self.batch_size =opt.batch_size
        self.gpu_id = opt.gpu_ids[0]

        self.indices = list(range(len(self.real_list)))
        
    def __iter__(self):
        return self
    
    def set_epoch_start(self):
        random.shuffle(self.real_list)
        random.shuffle(self.fake_list)
        self.iteration = 0

    def __next__(self):
        """返回一个批次的处理后数据"""
        batch_images = []
        for idx in range(self.iteration * (self.batch_size // (1 + self.fake_num)), (self.iteration + 1) * (self.batch_size // (1 + self.fake_num))):
            try:
                # 加载图像
                img_path = self.real_list[idx]
                
                # 打开并转换图像
                img = Image.open(img_path).convert("RGB")
                
                # 添加到批次
                batch_images.append(img)
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

        # 复制出所需的fake份数
        batch_fake_images_list = []
        for i in range(self.fake_num):
            batch_fake_images_list.append(batch_images.copy())

        # 处理label
        label_list = [0] * len(batch_images)
        for i in range(self.fake_num):
            label_list += [1] * len(batch_images)
        labels_tensor = torch.tensor(label_list)

        # Save random state before transformations
        random_state = torch.get_rng_state()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        for trans in self.transform_funcs:
            torch.set_rng_state(random_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            batch_images = trans(batch_images)

        real_images_tensor = torch.stack(batch_images).cuda(self.gpu_id)

        fake_images_tensor_list = []
        for i in range(self.fake_num):
            vae_transform_funcs = self.vae_transform_funcs_list[i]
            batch_fake_images = batch_fake_images_list[i]

            for trans in vae_transform_funcs:
                torch.set_rng_state(random_state)
                np.random.set_state(numpy_state)
                random.setstate(python_state)
                batch_fake_images = trans(batch_fake_images)
            
            fake_images_tensor_list.append(torch.stack(batch_fake_images))

        fake_images_tensor = torch.cat(fake_images_tensor_list).cuda(self.gpu_id)

        images_tensor = torch.cat((real_images_tensor, fake_images_tensor), dim=0)

        self.iteration += 1

        return images_tensor, labels_tensor

    
    def __len__(self):
        """返回一个epoch中的批次数量"""
        return ( len(self.real_list) // (self.batch_size) ) * (1 + self.fake_num)

# ---------- Offline utils ----------

class PadRandomCrop:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):

        w, h = img.size  # 假设输入为 [C, H, W]

        pad_h = max(0, self.size - h)
        pad_w = max(0, self.size - w)

        # 如果需要填充
        if pad_h > 0 or pad_w > 0:
            padding = (
                pad_w // 2,          # left
                pad_h // 2,          # top
                pad_w - pad_w // 2,  # right
                pad_h - pad_h // 2   # bottom
            )
            img = F.pad(img, padding, fill=0)  # 填充0或其他值，如255

        # 对单个图像应用随机裁剪
        cropped = transforms.RandomCrop(self.size)(img)
            
        return cropped

class PadCenterCrop:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):

        w, h = img.size  # 假设输入为 [C, H, W]

        pad_h = max(0, self.size - h)
        pad_w = max(0, self.size - w)

        # 如果需要填充
        if pad_h > 0 or pad_w > 0:
            padding = (
                pad_w // 2,          # left
                pad_h // 2,          # top
                pad_w - pad_w // 2,  # right
                pad_h - pad_h // 2   # bottom
            )
            img = F.pad(img, padding, fill=0)  # 填充0或其他值，如255

        # 对单个图像应用随机裁剪
        cropped = transforms.CenterCrop(self.size)(img)
            
        return cropped
    
class ComposedTransforms:
    """一个图像转换组合，可以一致地应用于多个图像"""
    def __init__(self, transforms_list):
        """
        初始化转换组合
        
        Args:
            transforms_list: 一个torchvision.transforms.Compose对象
        """
        self.transforms = transforms_list
        
    def __call__(self, images_dict):
        """
        对字典中的所有图像应用相同的转换
        
        Args:
            images_dict: 包含不同类型图像的字典
                        (例如, 'real', 'fake', 'real_resized', 'fake_resized')
                        
        Returns:
            包含转换后图像的字典
        """
        # 保存随机状态以实现一致的转换
        torch_state = torch.get_rng_state()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        
        result = {}
        
        # 对字典中的每个图像应用转换
        for key, val in images_dict.items():
            if val is None:
                result[key] = None
                continue
                
            if isinstance(val, list):
                # 处理图像列表
                transformed_imgs = []
                for i, single_img in enumerate(val):                   
                    # 为每个图像重置随机状态
                    torch.set_rng_state(torch_state)
                    np.random.set_state(numpy_state)
                    random.setstate(python_state)
                    
                    # 应用所有转换
                    transformed = self.transforms(single_img)
                    transformed_imgs.append(transformed)
                
                result[key] = transformed_imgs

            elif isinstance(val, Image.Image):
                # 处理单个PIL图像
                # 重置随机状态
                torch.set_rng_state(torch_state)
                np.random.set_state(numpy_state)
                random.setstate(python_state)
                
                # 应用所有转换
                transformed = self.transforms(val)
                result[key] = transformed
            else:
                result[key] = val
                
        return result


def shuffle_image_patches(image, patch_size=14, shuffle_order=None):
    """Shuffle patches of a tensor image.
    
    Args:
        image: Tensor image in [C, H, W] format
        patch_size: Size of patches to shuffle
        shuffle_order: Optional pre-defined shuffle order to use for consistency
        
    Returns:
        Shuffled tensor image
    """
    channels, height, width = image.shape
    num_patches_per_row = height // patch_size
    num_patches_per_col = width // patch_size
    total_patches = num_patches_per_row * num_patches_per_col
    
    # Store patches
    image_patches = []
    
    # Divide image into patches
    for i in range(num_patches_per_row):
        for j in range(num_patches_per_col):
            start_row = i * patch_size
            start_col = j * patch_size
            
            image_patch = image[:, start_row:start_row + patch_size, 
                             start_col:start_col + patch_size]
            image_patches.append(image_patch)
    
    # Use provided shuffle order or generate a new one
    if shuffle_order is None or len(shuffle_order) != total_patches:
        shuffle_order = np.arange(total_patches)
        np.random.shuffle(shuffle_order)
    
    # Rebuild shuffled image
    shuffled_image = torch.zeros_like(image)
    
    for idx, shuffle_idx in enumerate(shuffle_order):
        i = idx // num_patches_per_col
        j = idx % num_patches_per_col
        
        start_row = i * patch_size
        start_col = j * patch_size
        
        shuffled_image[:, start_row:start_row + patch_size, 
                     start_col:start_col + patch_size] = image_patches[shuffle_idx]
    
    return shuffled_image, shuffle_order


# Function to apply sequential resizing
def apply_resize(image, resize_factor, resize_method=None):
    """Apply a single resize operation to an image with specified factor and method.
    
    Args:
        image: PIL Image to resize
        resize_factor: Float scaling factor
        resize_method: PIL resampling method (defaults to random selection if None)
    
    Returns:
        Resized PIL Image
    """
    if resize_method is None:
        resampling_methods = [
            Image.NEAREST,
            Image.BOX,
            Image.BILINEAR,
            Image.HAMMING,
            Image.BICUBIC,
            Image.LANCZOS,
        ]
        resize_method = random.choice(resampling_methods)
    
    # Get current dimensions and calculate new dimensions
    w, h = image.size
    new_w, new_h = int(w * resize_factor), int(h * resize_factor)
    
    # Apply resize
    resized = image.resize((new_w, new_h), resize_method)
    
    return resized


class RealFakeDataset(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val"]
        self.opt = opt
        self.data_label = opt.data_label
        
        # Get dataset split
        temp = 'train' if opt.data_label == 'train' else 'val'
        
        # Get real image paths
        real_list = get_list(opt.real_list_path)
        real_list.sort()
        
        # Parse VAE models and resize factors
        self.vae_models = opt.vae_models.split(',')

        self.down_resize_factors = [float(f) for f in opt.down_resize_factors.split(',')]
        self.upper_resize_factors = [float(f) for f in opt.upper_resize_factors.split(',')]
        
        # Create a list of data samples, each sample is a dictionary
        self.data_list = []
        
        # Construct the complete dataset
        for real_path in tqdm(real_list, desc="Loading vae rec data..."):
            # Create a data sample with real path and corresponding fake paths
            sample = {
                'real_path': real_path,
                'fake_paths': [],
            }

            basename_real_path = os.path.basename(real_path)
            basename_real_path_without_suffix = os.path.splitext(basename_real_path)[0] 
            basename_fake_path = basename_real_path_without_suffix + '.png'
                        
            for vae_rec_dir in self.vae_models:
                vae_rec_path = os.path.join(vae_rec_dir, basename_fake_path)
                if not os.path.exists(vae_rec_path):
                    print(f"{vae_rec_path} not exists")
                else:
                    sample['fake_paths'].append(vae_rec_path)
            
            # Only add samples that have at least one valid fake path
            if len(sample['fake_paths']) > 0:
                self.data_list.append(sample)
        
        # Shuffle the data list
        random.shuffle(self.data_list)
    
        # Choose normalization stats
        stat_from = "imagenet" if self.opt.arch.lower().startswith("imagenet") else "clip"
        print("mean and std stats are from:", stat_from)
   
        transform_list = create_train_transforms(size=opt.cropSize, mean=MEAN[stat_from], std=STD[stat_from])
        
        # Create composed transforms
        self.transform = ComposedTransforms(transform_list)

        # Set patch shuffle parameters
        self.patch_size = getattr(opt, 'patch_size', 14)  # Default to 14 if not specified

    def __len__(self):
        return len(self.data_list)
        
    def __getitem__(self, idx):
        current_idx = idx
        
        try:
            # Get sample data
            sample = self.data_list[current_idx]
            
            # Create image dictionary
            img_dict = {
                'real': None,
                'fake': None,
                'real_resized': [],
                'fake_resized': [],
            }
            
            # Load real image
            real_img_path = sample['real_path']
            real_img = Image.open(real_img_path).convert("RGB")
            
            img_dict['real'] = real_img
            
            # Randomly select one fake image path
            if len(sample['fake_paths']) > 0:
                fake_path = random.choice(sample['fake_paths'])
                fake_img = Image.open(fake_path).convert("RGB")
                
                # Apply blending if needed
                if self.opt.p_blend > 0 and random.random() < self.opt.p_blend:
                    if fake_img.size != real_img.size:
                        real_img_resized = real_img.resize(fake_img.size, Image.LANCZOS)
                        fake_img = Image.blend(real_img_resized, fake_img, self.opt.ratio_blend)
                    else:
                        fake_img = Image.blend(real_img, fake_img, self.opt.ratio_blend)
                
                img_dict['fake'] = fake_img
            else:
                raise ValueError("No fake images could be loaded")
                
            
            resampling_methods = [
                Image.NEAREST,
                Image.BOX,
                Image.BILINEAR,
                Image.HAMMING,
                Image.BICUBIC,
                Image.LANCZOS,
            ]
            
            # Select random resize factors and methods for consistency
            down_resize_factor = random.choice(self.down_resize_factors)
            upper_resize_factor = random.choice(self.upper_resize_factors)
            # Select two random methods without replacement
            # Note: random.sample() already selects without replacement
            resize_methods = random.sample(resampling_methods, 2)  # Selects 2 unique methods

            for resize_factor, resize_method in zip([down_resize_factor, upper_resize_factor], resize_methods):
                # Create resized versions with the SAME resize factor and method
                real_resized = apply_resize(real_img, resize_factor, resize_method)
                fake_resized = apply_resize(fake_img, resize_factor, resize_method)
            
                img_dict['real_resized'].append(real_resized)
                img_dict['fake_resized'].append(fake_resized)
            
            # Apply transforms to all images
            transformed_dict = self.transform(img_dict)
            
            # Apply image patch shuffle with probability p_shuffle
            if hasattr(self.opt, 'p_shuffle') and self.opt.p_shuffle > 0 and random.random() < self.opt.p_shuffle:
                # Generate a single shuffle order to be used consistently for all images
                shuffle_order = None
                
                # Apply patch shuffle to all images in the dictionary using the same shuffle order
                if transformed_dict['real'] is not None:
                    transformed_dict['real'], shuffle_order = shuffle_image_patches(
                        transformed_dict['real'], self.patch_size, shuffle_order
                    )
                
                if transformed_dict['fake'] is not None:
                    transformed_dict['fake'], _ = shuffle_image_patches(
                        transformed_dict['fake'], self.patch_size, shuffle_order
                    )
                
                # Apply to resized images if they exist
                if 'real_resized' in transformed_dict and len(transformed_dict['real_resized']) > 0:
                    shuffled_real_resized = []
                    for img in transformed_dict['real_resized']:
                        shuffled_img, _ = shuffle_image_patches(img, self.patch_size, shuffle_order)
                        shuffled_real_resized.append(shuffled_img)
                    transformed_dict['real_resized'] = shuffled_real_resized
                
                if 'fake_resized' in transformed_dict and len(transformed_dict['fake_resized']) > 0:
                    shuffled_fake_resized = []
                    for img in transformed_dict['fake_resized']:
                        shuffled_img, _ = shuffle_image_patches(img, self.patch_size, shuffle_order)
                        shuffled_fake_resized.append(shuffled_img)
                    transformed_dict['fake_resized'] = shuffled_fake_resized
            
            return transformed_dict
            
        except Exception as e:
            print(f"Error processing image at index {current_idx}: {e}")
            # Try the next image
            current_idx = (current_idx + 1) % len(self.data_list)
            return self.__getitem__(current_idx)

    
def custom_collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    # If the entire batch is None, return an empty batch
    if len(batch) == 0:
        return {
            "real": torch.tensor([]),
            "real_resized": torch.tensor([]),
            "fake": torch.tensor([]),
            "fake_resized": torch.tensor([])
        }
    
    # Create lists for different image types
    real_images = []
    fake_images = []
    real_resized_images = []
    fake_resized_images = []
    
    # Extract images from batch
    for item in batch:
        # Process 'real' images
        if 'real' in item:
            if isinstance(item['real'], list):
                for real_img in item['real']:
                    if real_img is not None:
                        real_images.append(real_img)
            else:  # Single instance
                if item['real'] is not None:
                    real_images.append(item['real'])
        
        # Process 'fake' images
        if 'fake' in item:
            if isinstance(item['fake'], list):
                for fake_img in item['fake']:
                    if fake_img is not None:
                        fake_images.append(fake_img)
            else:  # Single instance
                if item['fake'] is not None:
                    fake_images.append(item['fake'])

        # Process 'real_resized' images
        if 'real_resized' in item:
            if isinstance(item['real_resized'], list):
                for real_resize_img in item['real_resized']:    
                    if real_resize_img is not None:
                        real_resized_images.append(real_resize_img)
            else:  # Single instance
                if item['real_resized'] is not None:
                    real_resized_images.append(item['real_resized'])
        
        # Process 'fake_resized' images
        if 'fake_resized' in item:
            if isinstance(item['fake_resized'], list):
                for fake_resize_img in item['fake_resized']:
                    if fake_resize_img is not None:
                        fake_resized_images.append(fake_resize_img)
            else:  # Single instance
                if item['fake_resized'] is not None:
                    fake_resized_images.append(item['fake_resized'])
    
    # Only stack if there are images
    real_images_tensor = torch.stack(real_images) if real_images else torch.tensor([])
    real_resized_images_tensor = torch.stack(real_resized_images) if real_resized_images else torch.tensor([])
    fake_images_tensor = torch.stack(fake_images) if fake_images else torch.tensor([])
    fake_resized_images_tensor = torch.stack(fake_resized_images) if fake_resized_images else torch.tensor([])
    
    return {
        "real": real_images_tensor, 
        "real_resized": real_resized_images_tensor, 
        "fake": fake_images_tensor, 
        "fake_resized": fake_resized_images_tensor
    }

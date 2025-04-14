import cv2
import torch
import numpy as np
from random import random, choice
from PIL import Image, ImageOps
from scipy.ndimage.filters import gaussian_filter
from io import BytesIO
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

def data_augment(img, opt):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


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


# ------------ batch version transforms ------------

def batch_unify_size_resize(batch_img, target_size=None):
    """将批量图像调整为统一尺寸
    
    Args:
        batch_img: 图像批次列表，每个元素为PIL Image对象
        target_size: 目标尺寸，如果为None则使用批次中最大的图像尺寸
        
    Returns:
        调整大小后的图像批次列表
    """
    batch_size = len(batch_img)
    result = []
    
    # 如果没有指定目标尺寸，则使用批次中最大的尺寸
    if target_size is None:
        max_width = 0
        max_height = 0
        for img in batch_img:
            w, h = img.size
            max_width = max(max_width, w)
            max_height = max(max_height, h)
        target_size = (max_width, max_height)
    
    for i in range(batch_size):
        # 对单个图像进行resize操作
        resized = F.resize(batch_img[i], target_size, interpolation=Image.BILINEAR)
        result.append(resized)
        
    return result

def batch_unify_size_padding(batch_img, target_size=None):
    """将批量图像通过填充调整为统一尺寸
    
    Args:
        batch_img: 图像批次列表，每个元素为PIL Image对象
        target_size: 目标尺寸，如果为None则使用批次中最大的图像尺寸
        
    Returns:
        填充后的图像批次列表
    """
    batch_size = len(batch_img)
    result = []
    
    # 如果没有指定目标尺寸，则使用批次中最大的尺寸
    if target_size is None:
        max_width = 0
        max_height = 0
        for img in batch_img:
            w, h = img.size
            max_width = max(max_width, w)
            max_height = max(max_height, h)
        target_size = (max_width, max_height)
    
    for i in range(batch_size):
        img = batch_img[i]
        w, h = img.size
        
        # 计算需要的填充量
        pad_w = max(0, target_size[0] - w)
        pad_h = max(0, target_size[1] - h)
        
        # 应用填充
        if pad_w > 0 or pad_h > 0:
            padding = (
                pad_w // 2,          # left
                pad_h // 2,          # top
                pad_w - pad_w // 2,  # right
                pad_h - pad_h // 2   # bottom
            )
            padded = ImageOps.expand(img, padding, fill=0)  # 黑色填充
        else:
            # 如果图像已经大于或等于目标尺寸，则进行中心裁剪
            padded = transforms.CenterCrop(target_size)(img)
        
        result.append(padded)
        
    return result

def batch_data_augment(batch_img, opt):
    """对批量图像进行数据增强处理"""
    batch_size = len(batch_img)
    result = []
    
    for i in range(batch_size):
        # 对单个样本进行数据增强
        augmented = data_augment(batch_img[i], opt)
        result.append(augmented)
    
    return result

def batch_to_tensor(batch_img):
    """将批量图像转换为tensor"""
    batch_size = len(batch_img)
    result = []
    
    for i in range(batch_size):
        # 转换为tensor并归一化
        img_tensor = transforms.ToTensor()(batch_img[i])
        img_tensor = transforms.Normalize(mean=MEAN["clip"], std=STD["clip"])(img_tensor)
        result.append(img_tensor)
        
    return result

class BatchRandomCrop:
    """批量随机裁剪"""
    def __init__(self, size):
        self.size = size
        
    def __call__(self, batch_img):
        batch_size = len(batch_img)
        result = []
        
        for i in range(batch_size):

            w, h = batch_img[i].size  # 假设输入为 [C, H, W]

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
                batch_img[i] = F.pad(batch_img[i], padding, fill=0)  # 填充0或其他值，如255

            # 对单个图像应用随机裁剪
            cropped = transforms.RandomCrop(self.size)(batch_img[i])
            result.append(cropped)
            
        return result

class BatchCenterCrop:
    """批量中心裁剪"""
    def __init__(self, size):
        self.size = size
        
    def __call__(self, batch_img):
        # 中心裁剪可以直接对整个批次操作
        batch_size = len(batch_img)
        result = []
        
        for i in range(batch_size):

            w, h = batch_img[i].size  # 假设输入为 [C, H, W]

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
                batch_img[i] = F.pad(batch_img[i], padding, fill=0)  # 填充0或其他值，如255

            # 对单个图像应用随机裁剪
            cropped = transforms.CenterCrop(self.size)(batch_img[i])
            result.append(cropped)
            
        return result

class BatchRandomHorizontalFlip:
    """批量随机水平翻转"""
    def __init__(self, p=0.5):
        self.p = p
        
    def __call__(self, batch_img):
        batch_size = len(batch_img)
        result = []
        
        for i in range(batch_size):
            # 对单个图像应用随机翻转
            if torch.rand(1) < self.p:
                flipped = F.hflip(batch_img[i])
            else:
                flipped = batch_img[i]
            result.append(flipped)
            
        return result

def create_transformations(opt):
    """创建批量图像变换函数列表"""
    transforms_list = []

    # 预处理 统一图像尺寸
    if opt.pre_vae == "resize":
        # 将图像通过resize调整为统一大小
        unify_size_func = lambda batch_img: batch_unify_size_resize(batch_img)
    elif opt.pre_vae == "padding":
        # 将图像通过padding调整为统一大小
        unify_size_func = lambda batch_img: batch_unify_size_padding(batch_img)
    else:
        unify_size_func = lambda batch_img: batch_img
    transforms_list.append(unify_size_func)

    # 数据增强
    transforms_list.append(lambda batch_img: batch_data_augment(batch_img, opt))

    # 裁剪
    if opt.isTrain:
        crop_func = BatchRandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = lambda batch_img: batch_img
    else:
        crop_func = BatchCenterCrop(opt.cropSize)
    transforms_list.append(crop_func)
    
    # 翻转
    if opt.isTrain and not opt.no_flip:
        flip_func = BatchRandomHorizontalFlip()
    else:
        flip_func = lambda batch_img: batch_img
    transforms_list.append(flip_func)

    transforms_list.append(lambda batch_img: batch_to_tensor(batch_img))
    
    return transforms_list

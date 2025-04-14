import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import random as rd
from random import random, choice, shuffle
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
import pickle
import os 
from skimage.io import imread
from copy import deepcopy
import torch

from .vae import VAETransform, DoNothing
import torchvision.transforms.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True


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


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, opt.loadSize, interpolation=rz_dict[interp])



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
        self.real_list = get_list( os.path.join(opt.real_list_path, f'{temp}2017') )
        self.fake_list = self.real_list.copy()
        self.vae_model = vae_model
        self.transform_funcs = transform_funcs

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
        shuffle(self.real_list)
        shuffle(self.fake_list)
        self.iteration = 0
        
    def __next__(self):
        """返回一个批次的处理后数据"""
        # # 随机选择索引
        # real_batch_indices = rd.sample(self.indices, self.batch_size // 2)
        # fake_batch_indices = rd.sample(self.indices, self.batch_size // 2)

        batch_real_images = []
        batch_fake_images = []
        batch_labels = []

        for idx in range(self.iteration * (self.batch_size // 2), (self.iteration + 1) * (self.batch_size // 2)):
            try:
                # 加载图像
                img_path = self.real_list[idx]
                label = 0
                
                # 打开并转换图像
                img = Image.open(img_path).convert("RGB")
                
                # # 应用数据变换（裁剪、翻转等）
                # for transform in self.transform_funcs:
                #     img = transform(img)
                
                # 转换为tensor并归一化
                # if isinstance(img, Image.Image):
                #     img = transforms.ToTensor()(img)
                #     img = transforms.Normalize(mean=MEAN["clip"], std=STD["clip"])(img)
                
                # 添加到批次
                batch_real_images.append(img)
                batch_labels.append(label)
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

        for idx in range(self.iteration * (self.batch_size // 2), (self.iteration + 1) * (self.batch_size // 2)):
            try:
                # 加载图像
                img_path = self.fake_list[idx]
                label = 1
                
                # 打开并转换图像
                img = Image.open(img_path).convert("RGB")
                
                # # 应用数据变换（裁剪、翻转等）
                # for transform in self.transform_funcs:
                #     img = transform(img)
                
                # 转换为tensor并归一化
                # if isinstance(img, Image.Image):
                #     img = transforms.ToTensor()(img)
                #     img = transforms.Normalize(mean=MEAN["clip"], std=STD["clip"])(img)
                
                # 添加到批次
                batch_fake_images.append(img)
                batch_labels.append(label)
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

        # 批量transform和vae

        for trans in self.transform_funcs:
            batch_real_images = trans(batch_real_images)

        vae_transform_funcs = rd.choice(self.vae_transform_funcs_list)
        for trans in vae_transform_funcs:
            batch_fake_images = trans(batch_fake_images)
     
        # 堆叠为张量
        real_images_tensor = torch.stack(batch_real_images).cuda(self.gpu_id)
        fake_images_tensor = torch.stack(batch_fake_images).cuda(self.gpu_id)
        labels_tensor = torch.tensor(batch_labels).cuda(self.gpu_id)

        images_tensor = torch.cat((real_images_tensor, fake_images_tensor), dim=0)

        self.iteration += 1
            
        return images_tensor, labels_tensor
    
    def __len__(self):
        """返回一个epoch中的批次数量"""
        return ( len(self.real_list) // (self.batch_size) ) * (1+len(self.vae_transform_funcs_list))

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


class RealFakeDataset(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val"]
        #assert opt.data_mode in ["ours", "wang2020", "ours_wang2020"]
        self.data_label  = opt.data_label
        if opt.data_mode == 'ours':
            pickle_name = "train.pickle" if opt.data_label=="train" else "val.pickle"
            real_list = get_list( os.path.join(opt.real_list_path, pickle_name) )
            fake_list = get_list( os.path.join(opt.fake_list_path, pickle_name) )
        elif opt.data_mode == 'wang2020':
            temp = 'train/progan' if opt.data_label == 'train' else 'test/progan'
            real_list = get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='0_real' )
            fake_list = get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='1_fake' )
        elif opt.data_mode == 'ours_wang2020':
            pickle_name = "train.pickle" if opt.data_label=="train" else "val.pickle"
            real_list = get_list( os.path.join(opt.real_list_path, pickle_name) )
            fake_list = get_list( os.path.join(opt.fake_list_path, pickle_name) )
            temp = 'train/progan' if opt.data_label == 'train' else 'test/progan'
            real_list += get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='0_real' )
            fake_list += get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='1_fake' )    
        elif opt.data_mode == 'multi_wang2020':
            temp = 'train/progan' if opt.data_label == 'train' else 'test/progan'
            real_list = get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='0_real' )
            fake_list = get_list( os.path.join(opt.wang2020_data_path,temp), must_contain='1_fake' )
            name = ['adm', 'biggan', 'glide', 'midjourney', 'sdv14', 'sdv15', 'wukong', 'vqdm']
            for dataset in name:
                temp = 'train/'+dataset if opt.data_label == 'train' else 'test/'+dataset
                print(temp)
                real_list += get_list( os.path.join(opt.wang2020_data_path, temp), must_contain='0_real' )
                fake_list += get_list( os.path.join(opt.wang2020_data_path, temp), must_contain='1_fake' )

            if opt.chameleon:
                temp = 'train' if opt.data_label == 'train' else 'val'
                print(f"chameleon/{temp}")
                real_chameleon_list = get_list( f'/root/autodl-tmp/AIGC_data/Chameleon/{temp}', must_contain='0_real' )
                fake_chameleon_list = get_list( f'/root/autodl-tmp/AIGC_data/Chameleon/{temp}', must_contain='1_fake' )
        elif opt.data_mode == 'mscoco':
            temp = 'train' if opt.data_label == 'train' else 'val'
            print(opt.real_list_path)
            # print(opt.fake_list_path)
            real_list = get_list( os.path.join(opt.real_list_path, f'{temp}2017') )
            # fake_list = get_list( os.path.join(opt.fake_list_path, f'{temp}2017') )
            fake_list_paths = opt.fake_list_path.split(',')
            fake_list = []
            for fake_list_path in fake_list_paths:
                print(fake_list_path)
                fake_list += get_list( os.path.join(fake_list_path, f'{temp}2017') )
        print(len(real_list))



        # setting the labels for the dataset
        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1
            
        if opt.chameleon:
            self.chameleon_labels_dict = {}
            for i in real_chameleon_list:
                self.chameleon_labels_dict[i] = 0
            for i in fake_chameleon_list:
                self.chameleon_labels_dict[i] = 1

        self.chameleon = opt.chameleon
        self.chameleon_freq = opt.chameleon_freq

        self.total_list = real_list + fake_list
        self.chameleon_list = real_chameleon_list + fake_chameleon_list if opt.chameleon else []
        shuffle(self.total_list)
        shuffle(self.chameleon_list)

        target_size = (opt.cropSize, opt.cropSize)
        if opt.isTrain:
            crop_func = PadRandomCrop(opt.cropSize)
        elif opt.no_crop:
            crop_func = transforms.Lambda(lambda img: img)
        else:
            crop_func = PadCenterCrop(opt.cropSize)

        if opt.isTrain and not opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(lambda img: img)

        stat_from = "imagenet" if opt.arch.lower().startswith("imagenet") else "clip"

        print("mean and std stats are from: ", stat_from)
        if '2b' not in opt.arch:
            print ("using Official CLIP's normalization")
            self.transform = transforms.Compose([
                transforms.Lambda(lambda img: data_augment(img, opt)),
                crop_func,
                flip_func,
                transforms.ToTensor(),
                transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
            ])
        else:
            print ("Using CLIP 2B transform")
            self.transform = None # will be initialized in trainer.py


    def __len__(self):
        return len(self.total_list)


    def __getitem__(self, idx):
        max_attempts = 5
        current_idx = idx 
        
        for _ in range(max_attempts):
            try:
                if self.chameleon and (current_idx % self.chameleon_freq) == 0:
                    img_path = self.chameleon_list[(current_idx // self.chameleon_freq) % len(self.chameleon_list)]
                    label = self.chameleon_labels_dict[img_path]
                else:
                    img_path = self.total_list[current_idx]
                    label = self.labels_dict[img_path]
                img = Image.open(img_path).convert("RGB")
                img = self.transform(img)
                return img, label
                
            except Exception as e:
                if self.chameleon and (current_idx % self.chameleon_freq) == 0:
                    print(f"加载图像出错 {self.chameleon_list[(current_idx // self.chameleon_freq) % len(self.chameleon_list)]}: {e}")
                else:
                    print(f"加载图像出错 {self.total_list[current_idx]}: {e}")
                current_idx = (current_idx + 1) % len(self.total_list)
        
        print(f"警告: 多次尝试后仍无法加载有效图像，返回空白图像")
        blank_img = torch.zeros(3, 224, 224)
        return blank_img, 1 


# def get_padding(img_size, target_size):
#     img_w, img_h = img_size
#     target_w, target_h = target_size

#     pad_w = max(target_w - img_w, 0)
#     pad_h = max(target_h - img_h, 0)

#     # 左右、上下对称 padding
#     padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
#     return padding

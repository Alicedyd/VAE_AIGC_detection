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
        shuffle(self.real_list)
        shuffle(self.fake_list)
        self.iteration = 0

    def __next__(self):
        """返回一个批次的处理后数据"""
        # # 随机选择索引
        # real_batch_indices = rd.sample(self.indices, self.batch_size // 2)
        # fake_batch_indices = rd.sample(self.indices, self.batch_size // 2)

        # batch_real_images = []
        # batch_fake_images = []
        # batch_labels = []

        # for idx in range(self.iteration * (self.batch_size // 2), (self.iteration + 1) * (self.batch_size // 2)):
        #     try:
        #         # 加载图像
        #         img_path = self.real_list[idx]
        #         label = 0
                
        #         # 打开并转换图像
        #         img = Image.open(img_path).convert("RGB")
                
        #         # # 应用数据变换（裁剪、翻转等）
        #         # for transform in self.transform_funcs:
        #         #     img = transform(img)
                
        #         # 转换为tensor并归一化
        #         # if isinstance(img, Image.Image):
        #         #     img = transforms.ToTensor()(img)
        #         #     img = transforms.Normalize(mean=MEAN["clip"], std=STD["clip"])(img)
                
        #         # 添加到批次
        #         batch_real_images.append(img)
        #         batch_labels.append(label)
                
        #     except Exception as e:
        #         print(f"Error processing image {img_path}: {e}")

        # for idx in range(self.iteration * (self.batch_size // 2), (self.iteration + 1) * (self.batch_size // 2)):
        #     try:
        #         # 加载图像
        #         img_path = self.real_list[idx]
        #         label = 1
                
        #         # 打开并转换图像
        #         img = Image.open(img_path).convert("RGB")
                
        #         # # 应用数据变换（裁剪、翻转等）
        #         # for transform in self.transform_funcs:
        #         #     img = transform(img)
                
        #         # 转换为tensor并归一化
        #         # if isinstance(img, Image.Image):
        #         #     img = transforms.ToTensor()(img)
        #         #     img = transforms.Normalize(mean=MEAN["clip"], std=STD["clip"])(img)
                
        #         # 添加到批次
        #         batch_fake_images.append(img)
        #         batch_labels.append(label)
                
        #     except Exception as e:
        #         print(f"Error processing image {img_path}: {e}")

        # # 批量transform和vae

        # # Save random state before transformations
        # random_state = torch.get_rng_state()
        # numpy_state = np.random.get_state()
        # python_state = rd.getstate()
        # for trans in self.transform_funcs:
        #     torch.set_rng_state(random_state)
        #     np.random.set_state(numpy_state)
        #     rd.setstate(python_state)
        #     batch_real_images = trans(batch_real_images)

        # vae_transform_funcs = rd.choice(self.vae_transform_funcs_list)
        # for trans in vae_transform_funcs:
        #     torch.set_rng_state(random_state)
        #     np.random.set_state(numpy_state)
        #     rd.setstate(python_state)
        #     batch_fake_images = trans(batch_fake_images)
     
        # # 堆叠为张量
        # real_images_tensor = torch.stack(batch_real_images).cuda(self.gpu_id)
        # fake_images_tensor = torch.stack(batch_fake_images).cuda(self.gpu_id)
        # labels_tensor = torch.tensor(batch_labels).cuda(self.gpu_id)

        # images_tensor = torch.cat((real_images_tensor, fake_images_tensor), dim=0)

        # self.iteration += 1
            
        # return images_tensor, labels_tensor

        batch_images = []
        for idx in range(self.iteration * (self.batch_size // (1 + self.fake_num)), (self.iteration + 1) * (self.batch_size // (1 + self.fake_num))):
            try:
                # 加载图像
                img_path = self.real_list[idx]
                
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
        python_state = rd.getstate()
        for trans in self.transform_funcs:
            torch.set_rng_state(random_state)
            np.random.set_state(numpy_state)
            rd.setstate(python_state)
            batch_images = trans(batch_images)

        real_images_tensor = torch.stack(batch_images).cuda(self.gpu_id)

        fake_images_tensor_list = []
        for i in range(self.fake_num):
            vae_transform_funcs = self.vae_transform_funcs_list[i]
            batch_fake_images = batch_fake_images_list[i]

            for trans in vae_transform_funcs:
                torch.set_rng_state(random_state)
                np.random.set_state(numpy_state)
                rd.setstate(python_state)
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
        elif opt.data_mode == 'mscoco':
            temp = 'train' if opt.data_label == 'train' else 'val'
            real_list = get_list(os.path.join(opt.real_list_path, f'{temp}2017'))
            real_list.sort()

            # 多个子域的 fake_list
            fake_list_paths = opt.fake_list_path.split(',')
            fake_list = []
            for fake_list_path in fake_list_paths:
                path_list = get_list(os.path.join(fake_list_path, f'{temp}2017'))
                path_list.sort()
                assert len(path_list) == len(real_list), "Mismatch in real and fake list length"
                fake_list.append(path_list)

            # 一致打乱
            indices = list(range(len(real_list)))
            rd.shuffle(indices)

            self.real_list = [real_list[i] for i in indices]
            self.fake_list = []
            for i in range(len(fake_list)):  # 对每个子域
                self.fake_list.append([fake_list[i][j] for j in indices])

            self.num_fake = len(self.fake_list)

        print(len(real_list))



        # setting the labels for the dataset
        # self.labels_dict = {}
        # for i in real_list:
        #     self.labels_dict[i] = 0
        # for i in fake_list:
        #     self.labels_dict[i] = 1

        # self.total_list = real_list + fake_list
        # shuffle(self.total_list)

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
        # return len(self.total_list)
        return len(self.real_list)


    def __getitem__(self, idx):
        max_attempts = 5
        current_idx = idx 
        
        for _ in range(max_attempts):
            try:
                # if self.chameleon and (current_idx % self.chameleon_freq) == 0:
                #     img_path = self.chameleon_list[(current_idx // self.chameleon_freq) % len(self.chameleon_list)]
                #     label = self.chameleon_labels_dict[img_path]
                # else:
                #     img_path = self.total_list[current_idx]
                #     label = self.labels_dict[img_path]
                # img = Image.open(img_path).convert("RGB")
                # img = self.transform(img)
                # return img, label

                # real_img_path = self.real_list[current_idx]
                # real_label = 0

                # current_fake_list = rd.choice(self.fake_list)
                # fake_img_path = current_fake_list[current_idx]
                # fake_label = 1

                # real_img = Image.open(real_img_path).convert("RGB")
                # fake_img = Image.open(fake_img_path).convert("RGB")

                # # Save random state before transformations
                # random_state = torch.get_rng_state()
                # numpy_state = np.random.get_state()
                # python_state = rd.getstate()

                # real_img = self.transform(real_img)

                # # Restore random state before transforming second image
                # torch.set_rng_state(random_state)
                # np.random.set_state(numpy_state)
                # rd.setstate(python_state)

                # fake_img = self.transform(fake_img)

                # return real_img, fake_img, real_label, fake_label
                # Get real image
                real_img_path = self.real_list[current_idx]
                real_img = Image.open(real_img_path).convert("RGB")
                
                # Store random state for consistent transformations
                random_state = torch.get_rng_state()
                numpy_state = np.random.get_state()
                python_state = rd.getstate()
                
                # Apply transforms to real image
                real_img = self.transform(real_img)
                
                # Get fake images from all domains
                fake_imgs = []
                for idx in range(self.num_fake):
                    # Get fake image from current domain
                    fake_img_path = self.fake_list[idx][current_idx]
                    fake_img = Image.open(fake_img_path).convert("RGB")
                    
                    # Restore random state for consistent transformation
                    torch.set_rng_state(random_state)
                    np.random.set_state(numpy_state)
                    rd.setstate(python_state)
                    
                    # Apply transforms to fake image
                    fake_img = self.transform(fake_img)
                    fake_imgs.append(fake_img)
                
                # Return real image, list of fake images, and labels
                return real_img, fake_imgs, 0, [1] * self.num_fake
                
            except Exception as e:
                print(f"加载图像出错 {self.real_list[current_idx]}: {e}")
                current_idx = (current_idx + 1) % len(self.real_list)
        
        print(f"警告: 多次尝试后仍无法加载有效图像，返回空白图像")
        blank_img = torch.zeros(3, 224, 224)
        return blank_img, 1 

# for customized batch
def custom_collate_fn(batch):
    # batch 是一个 list，每个元素是 __getitem__ 的返回值
    # 解包
    real_imgs, fake_imgs_list, labels, domain_labels_list = zip(*batch)

    # real_imgs 是 list[tensor] → stack 成 batch
    real_imgs = torch.stack(real_imgs, dim=0)  # (B, C, H, W)

    # fake_imgs_list 是 list[list[tensor]]
    # 转置后变为：num_fake_domains 个 list，每个 list 里是 B 个 tensor
    fake_imgs_list = list(zip(*fake_imgs_list))
    fake_imgs = [torch.stack(fake_imgs_per_domain, dim=0) for fake_imgs_per_domain in fake_imgs_list]
    # fake_imgs: list[tensor]，每个 tensor 形状是 (B, C, H, W)

    # labels 是 tuple[int] → 转成 tensor
    labels = torch.tensor(labels)

    # domain_labels_list 是 list[list[int]]
    domain_labels_list = list(zip(*domain_labels_list))
    domain_labels = [torch.tensor(domain_label_per_domain) for domain_label_per_domain in domain_labels_list]
    # domain_labels: list[tensor]，每个 tensor 是 (B,)

    return real_imgs, fake_imgs, labels, domain_labels


# def get_padding(img_size, target_size):
#     img_w, img_h = img_size
#     target_w, target_h = target_size

#     pad_w = max(target_w - img_w, 0)
#     pad_h = max(target_h - img_h, 0)

#     # 左右、上下对称 padding
#     padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
#     return padding

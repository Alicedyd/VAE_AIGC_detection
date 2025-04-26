import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from models import get_model

from tqdm import tqdm
from torch.utils.data import Dataset
from scipy.ndimage.filters import gaussian_filter

import pickle
import os
from io import BytesIO
from PIL import Image 
import random


TSNE_DATASET_PATHS = [
    {'real_path': '/root/autodl-tmp/AIGC_data/MSCOCO/val2017', 'fake_path': '', 'data_mode': 'ours', 'key': 'MSCOCO'},

    {'real_path': '', 'fake_path': '/root/autodl-tmp/AIGC_data/MSCOCO_XL/val2017', 'data_mode': 'ours', 'key': 'XL-VAE'},

    {'real_path': '', 'fake_path': '/root/autodl-tmp/AIGC_data/flux-dev/1.0/val2017', 'data_mode': 'ours', 'key': 'FLUX'}

    # {'real_path': '', 'fake_path': '/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-xl-base-1.0/val2017', 'data_mode': 'ours', 'key': 'XL-FAKE'},

    # {'real_path': '', 'fake_path': '/root/autodl-tmp/AIGC_data/DRCT-2M/stable-diffusion-xl-1.0-inpainting-0.1//val2017', 'data_mode': 'ours', 'key': 'XL-Inpainting'},

    # {'real_path': '', 'fake_path': '/root/autodl-tmp/AIGC_data/MSCOCO_MSE/val2017', 'data_mode': 'ours', 'key': 'MSE'},

    # {'real_path': '', 'fake_path': '/root/autodl-tmp/AIGC_data/MSCOCO_EMA/val2017', 'data_mode': 'ours', 'key': 'EMA'},

    # {'real_path': '', 'fake_path': '/root/autodl-tmp/AIGC_data/GenImage/Midjourney/val/ai', 'data_mode': 'ours', 'key': 'GenImage/Midjourney'}, 
    
    # {'real_path': '', 'fake_path': '/root/autodl-tmp/AIGC_data/GenImage/stable_diffusion_v_1_4/val/ai', 'data_mode': 'ours', 'key': 'GenImage/sd14'}, 
    
    # {'real_path': '', 'fake_path': '/root/autodl-tmp/AIGC_data/GenImage/stable_diffusion_v_1_5/val/ai', 'data_mode': 'ours', 'key': 'GenImage/sd15'}, 

    # {'real_path': '', 'fake_path': '/root/autodl-tmp/AIGC_data/GenImage/ADM/val/ai', 'data_mode': 'ours', 'key': 'GenImage/ADM'}, 

    # {'real_path': '', 'fake_path': '/root/autodl-tmp/AIGC_data/GenImage/glide/val/ai', 'data_mode': 'ours', 'key': 'GenImage/glide'}, 

    # {'real_path': '', 'fake_path': '/root/autodl-tmp/AIGC_data/GenImage/wukong/val/ai', 'data_mode': 'ours', 'key': 'GenImage/wukong'}, 

    # {'real_path': '', 'fake_path': '/root/autodl-tmp/AIGC_data/GenImage/VQDM/val/ai', 'data_mode': 'ours', 'key': 'GenImage/VQDM'}, 
    
    # {'real_path': '', 'fake_path': '/root/autodl-tmp/AIGC_data/GenImage/BigGAN/val/ai', 'data_mode': 'ours', 'key': 'GenImage/BigGAN'}, 
]

MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}

def png2jpg(img, quality):
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality) # ranging from 0-95, 75 is default
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return Image.fromarray(img)


def gaussian_blur(img, sigma):
    img = np.array(img)

    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)

    return Image.fromarray(img)

def recursively_read(rootdir, must_contain, exts=["PNG", "png", "jpg", "JPEG", "jpeg", "bmp"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain='', eval_gen=False):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
        if eval_gen:
            image_list = [item for item in image_list if '0' in os.path.basename(item)]
    return image_list

class RealFakeDataset(Dataset):
    def __init__(self,  real_path, 
                        fake_path, 
                        data_mode, 
                        max_sample,
                        arch,
                        jpeg_quality=None,
                        gaussian_sigma=None,
                        resolution_thres=None,
                        key=None):

        # assert data_mode in ["wang2020", "ours"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        self.resolution_thres = resolution_thres

        self.key = key
        
        # = = = = = = data path = = = = = = = = = # 
        if type(real_path) == str and type(fake_path) == str:
            real_list, fake_list = self.read_path(real_path, fake_path, data_mode, max_sample)
        else:
            real_list = []
            fake_list = []
            for real_p, fake_p in zip(real_path, fake_path):
                real_l, fake_l = self.read_path(real_p, fake_p, data_mode, max_sample)
                real_list += real_l
                fake_list += fake_l

        self.total_list = real_list + fake_list


        # = = = = = =  label = = = = = = = = = # 

        self.labels_dict = {}
        for i in real_list:
            self.labels_dict[i] = 0
        for i in fake_list:
            self.labels_dict[i] = 1

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize( mean=MEAN[stat_from], std=STD[stat_from] ),
        ])


    def read_path(self, real_path, fake_path, data_mode, max_sample):

        is_eval_gen = self.key is not None and 'Eval_GEN' in self.key

        if data_mode == 'wang2020':
            real_list = get_list(real_path, must_contain='0_real')
            fake_list = get_list(fake_path, must_contain='1_fake')
        else:
            real_list = get_list(real_path)
            fake_list = []
            fake_path_list = fake_path.split(",")
            for fake_path in fake_path_list:
                fake_list += get_list(fake_path, eval_gen=is_eval_gen)


        if max_sample is not None:
            if (max_sample <= len(real_list)):
                random.shuffle(real_list)
                real_list = real_list[0:max_sample]

            if max_sample <= len(fake_list):
                random.shuffle(fake_list)
                fake_list = fake_list[0:max_sample]


        # assert len(real_list) == len(fake_list)  

        return real_list, fake_list



    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        
        img_path = self.total_list[idx]

        label = self.labels_dict[img_path]
        img = Image.open(img_path).convert("RGB")

        if self.gaussian_sigma is not None:
            img = gaussian_blur(img, self.gaussian_sigma) 
        if self.jpeg_quality is not None:
            img = png2jpg(img, self.jpeg_quality)

        img = self.transform(img)
        return img, label

ARCH="DINOv2-LoRA:dinov2_vitl14"

# 1. 加载预训练模型
model = get_model(ARCH)
state_dict = torch.load("checkpoints/dinov2_lora8_VAE_sdxl_flux_resize_lr1e-5_bs32_acc64_2epoch_PBLEND_0.25RBLEND_0.25/model_iters_30000.pth", map_location='cpu')['model']
#model.fc.load_state_dict(state_dict)
model.load_state_dict(state_dict)


# 4. 为每个数据集创建DataLoader
datasets = {}
dataloaders = {}

for dataset_path in TSNE_DATASET_PATHS:
    dataset = RealFakeDataset(  
        dataset_path['real_path'], 
        dataset_path['fake_path'], 
        dataset_path['data_mode'], 
        500, 
        ARCH,
    )
    datasets[dataset_path['key']] = dataset

for name, dataset in datasets.items():
    dataloaders[name] = DataLoader(dataset, batch_size=128, shuffle=False)

# 5. 提取每个数据集的特征
features = {}
labels = {}
dataset_indicators = {}  # 用于标记每个特征属于哪个数据集

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for dataset_name, dataloader in dataloaders.items():
    print(f"extracting feature from {dataset_name}")

    features[dataset_name] = []
    labels[dataset_name] = []
    
    with torch.no_grad():
        for inputs, batch_labels in tqdm(dataloader):
            inputs = inputs.to(device)
            # 提取特征
            batch_features, _ = model(inputs, return_feature=True)
            # 将特征转换为向量
            batch_features = batch_features.view(batch_features.size(0), -1)
            
            features[dataset_name].append(batch_features.cpu().numpy())
            labels[dataset_name].append(batch_labels.numpy())
    
    # 合并批次
    features[dataset_name] = np.concatenate(features[dataset_name], axis=0)
    labels[dataset_name] = np.concatenate(labels[dataset_name], axis=0)

from sklearn.decomposition import PCA

# 1. 将所有数据集的特征合并
all_features = []
all_labels = []
all_dataset_indicators = []

for idx, (dataset_name, dataset_features) in enumerate(features.items()):
    all_features.append(dataset_features)
    all_labels.append(labels[dataset_name])
    # 创建数据集指示器
    dataset_indicator = np.full(dataset_features.shape[0], idx)
    all_dataset_indicators.append(dataset_indicator)

all_features = np.concatenate(all_features, axis=0)
all_labels = np.concatenate(all_labels, axis=0)
all_dataset_indicators = np.concatenate(all_dataset_indicators, axis=0)

# 2. 可选：使用PCA预降维
n_components_pca = 50  # 可以调整
pca = PCA(n_components=n_components_pca)
features_pca = pca.fit_transform(all_features)
print(f"Cumulative explained variance with {n_components_pca} PCA components: {np.sum(pca.explained_variance_ratio_):.4f}")

# 3. 应用t-SNE
n_components_tsne = 2  # 可以是2或3，取决于您想要的可视化维度
perplexity = 30  # 可以调整，通常在5-50之间
tsne = TSNE(n_components=n_components_tsne, perplexity=perplexity, random_state=42, verbose=1)
features_tsne = tsne.fit_transform(features_pca)  # 也可直接用all_features，但处理较慢

# 1. 创建数据框
df_tsne = pd.DataFrame()
df_tsne['x'] = features_tsne[:, 0]
df_tsne['y'] = features_tsne[:, 1]
if n_components_tsne == 3:
    df_tsne['z'] = features_tsne[:, 2]
df_tsne['label'] = all_labels
df_tsne['dataset'] = all_dataset_indicators

# 2. 绘制散点图
plt.figure(figsize=(12, 10))

# 根据数据集划分颜色
dataset_names = list(features.keys())
palette = sns.color_palette("hls", len(dataset_names))

# 绘制2D散点图，按数据集着色
scatter = sns.scatterplot(
    x='x', y='y',
    hue='dataset',
    palette=palette,
    hue_order=range(len(dataset_names)),
    data=df_tsne,
    legend="full",
    alpha=0.7
)

# 添加图例
handles, labels = scatter.get_legend_handles_labels()
plt.legend(handles, dataset_names, title="Datasets")

plt.title(f't-SNE Visualization of Features Across Multiple Datasets (perplexity={perplexity})')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.savefig('tsne_datasets_comparison.png', dpi=300)
plt.show()

# 3. 额外: 创建按类别着色的图表
plt.figure(figsize=(12, 10))

# 获取唯一类别
unique_labels = np.unique(all_labels)
palette = sns.color_palette("hls", len(unique_labels))

# 绘制2D散点图，按类别着色
scatter = sns.scatterplot(
    x='x', y='y',
    hue='label',
    palette=palette,
    data=df_tsne,
    legend="full",
    alpha=0.7
)

plt.title(f't-SNE Visualization of Features Across Multiple Datasets (by Class)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.tight_layout()
plt.savefig('./tsne.png', dpi=300)
plt.show()
import argparse
from ast import arg
import os
import csv
import torch
import torchvision.transforms as transforms
import torch.utils.data
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from torch.utils.data import Dataset
import sys
from models import get_model
from PIL import Image 
import pickle
from tqdm import tqdm
from io import BytesIO
from copy import deepcopy
from dataset_paths import DATASET_PATHS
import random
import shutil
from scipy.ndimage.filters import gaussian_filter

import torchvision.transforms.functional as TF
from random import choice

import csv

SEED = 0
def set_seed():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}





def find_best_threshold(y_true, y_pred):
    "We assume first half is real 0, and the second half is fake 1"

    N = y_true.shape[0]

    if y_pred[0:N//2].max() <= y_pred[N//2:N].min(): # perfectly separable case
        return (y_pred[0:N//2].max() + y_pred[N//2:N].min()) / 2 

    best_acc = 0 
    best_thres = 0 
    for thres in y_pred:
        temp = deepcopy(y_pred)
        temp[temp>=thres] = 1 
        temp[temp<thres] = 0 

        acc = (temp == y_true).sum() / N  
        if acc >= best_acc:
            best_thres = thres
            best_acc = acc 
    
    return best_thres
        

 
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



def calculate_acc(y_true, y_pred, thres):
    r_acc = accuracy_score(y_true[y_true==0], y_pred[y_true==0] > thres)
    f_acc = accuracy_score(y_true[y_true==1], y_pred[y_true==1] > thres)
    acc = accuracy_score(y_true, y_pred > thres)
    return r_acc, f_acc, acc    


def validate(model, loader, find_thres=False, gpu_id=None, save_incorrect=False, save_dir=None):

    with torch.no_grad():
        y_true, y_pred = [], []
        print ("Length of dataset: %d" %(len(loader)))

        # Get image paths from dataset
        if save_incorrect:
            paths = loader.dataset.total_list

        img_idx = 0
        incorrect_fake_indices = []

        for img, label in loader:
            if gpu_id is not None:
                in_tens = img.cuda(gpu_id)
            else:
                in_tens = img.cuda()

            batch_preds = model(in_tens, return_feature=False).sigmoid().flatten().tolist()
            batch_size = len(batch_preds)
            
            # Check for incorrect fake predictions in this batch
            if save_incorrect:
                for i in range(batch_size):
                    if label[i] == 1 and batch_preds[i] < 0.5:  # Fake classified as real
                        incorrect_fake_indices.append(img_idx + i)

            img_idx += batch_size

            y_pred.extend(batch_preds)
            y_true.extend(label.flatten().tolist())

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    # ================== save this if you want to plot the curves =========== # 
    # torch.save( torch.stack( [torch.tensor(y_true), torch.tensor(y_pred)] ),  'baseline_predication_for_pr_roc_curve.pth' )
    # exit()
    # =================================================================== #
    
    # Get AP 
    ap = average_precision_score(y_true, y_pred)

    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)

    # Save misclassified fake images
    if save_incorrect and len(incorrect_fake_indices) > 0:
        # print(f"Saving {len(incorrect_fake_indices)} misclassified fake images to {save_dir}")

        os.makedirs(save_dir, exist_ok=True)

        if len(incorrect_fake_indices) > 10:
            incorrect_fake_indices = incorrect_fake_indices[:10]
        
        # Create a CSV file to log the misclassifications
        csv_path = os.path.join(save_dir, "misclassified_fake_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['image_path', 'true_label', 'prediction_score'])
            
            for idx in incorrect_fake_indices:
                img_path = paths[idx]
                score = y_pred[idx]
                
                # Save the image
                try:
                    # Copy the image to the save directory
                    filename = os.path.basename(img_path)
                    dest_path = os.path.join(save_dir, filename)
                    shutil.copy(img_path, dest_path)
                    
                    # Log to CSV
                    writer.writerow([img_path, 1, score])
                except Exception as e:
                    print(f"Error saving misclassified image {img_path}: {str(e)}")
        
        print(f"Misclassified fake images saved to {save_dir}")
        print(f"Log file saved to {csv_path}")
    
    if not find_thres:
        return ap, r_acc0, f_acc0, acc0


    # Acc based on the best thres
    best_thres = find_best_threshold(y_true, y_pred)
    r_acc1, f_acc1, acc1 = calculate_acc(y_true, y_pred, best_thres)

    return ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres

    
    



# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = # 




def recursively_read(rootdir, must_contain, exts=["PNG", "png", "jpg", "JPEG", "jpeg", "bmp"]):
    out = [] 
    for r, d, f in os.walk(rootdir):
        for file in f:
            if (file.split('.')[1] in exts)  and  (must_contain in os.path.join(r, file)):
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





class RealFakeDataset(Dataset):
    def __init__(self,  real_path, 
                        fake_path, 
                        data_mode, 
                        max_sample,
                        arch,
                        jpeg_quality=None,
                        gaussian_sigma=None,
                        resolution_thres=None):

        # assert data_mode in ["wang2020", "ours"]
        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        self.resolution_thres = resolution_thres
        
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

        if data_mode == 'wang2020':
            real_list = get_list(real_path, must_contain='0_real')
            fake_list = get_list(fake_path, must_contain='1_fake')
        else:
            real_list = get_list(real_path)
            fake_list = []
            fake_path_list = fake_path.split(",")
            for fake_path in fake_path_list:
                fake_list += get_list(fake_path)

        def filter_by_resolution(image_list):
            if self.resolution_thres is None:
                return image_list
                
            filtered_list = []
            for img_path in tqdm(image_list, desc="筛选分辨率"):
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        area = w * h
                        if self.resolution_thres[0] <= area <= self.resolution_thres[1]:
                            filtered_list.append(img_path)
                except Exception as e:
                    print(f"处理 {img_path} 时出错: {str(e)}")
            return filtered_list
        # 应用分辨率过滤
        real_list = filter_by_resolution(real_list)
        fake_list = filter_by_resolution(fake_list)

        if len(real_list) == 0 or len(fake_list) == 0:
            raise ValueError("经过分辨率过滤后，真实或生成图像数量为0，请检查分辨率阈值设置")


        if max_sample is not None:
            if (max_sample > len(real_list)) or (max_sample > len(fake_list)):
                max_sample = 100
                print("not enough images, max_sample falling to 100")
            random.shuffle(real_list)
            random.shuffle(fake_list)
            real_list = real_list[0:max_sample]
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





if __name__ == '__main__':


    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--real_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--fake_path', type=str, default=None, help='dir name or a pickle')
    parser.add_argument('--data_mode', type=str, default=None, help='wang2020 or ours')
    parser.add_argument('--key', type=str, default=None, help='dataset key')
    parser.add_argument('--max_sample', type=int, default=1000, help='only check this number of images for both fake/real')

    parser.add_argument('--arch', type=str, default='res50')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')

    parser.add_argument('--result_folder', type=str, default='./result', help='')
    parser.add_argument('--batch_size', type=int, default=128)

    parser.add_argument('--no_resize', action='store_true', help='if specified, do not resize the images for data augmentation')
    parser.add_argument('--rz_interp', default='bilinear')
    parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')

    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4.     Used to test robustness of our model. Not apply if None")
    parser.add_argument('--resolution_thres', type=int, nargs=2, default=None)

    parser.add_argument('--gpu_id', type=int, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    parser.add_argument('--save_bad_case', action="store_true")

    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=1.0, help='LoRA scaling factor')
    parser.add_argument('--lora_targets', type=str, default=None, help='LoRA trainable targets')

    opt = parser.parse_args()

    
    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder, exist_ok=True)

    lora_args = {}
    if hasattr(opt, 'lora_rank'):
        lora_args['lora_rank'] = opt.lora_rank
    if hasattr(opt, 'lora_alpha'):
        lora_args['lora_alpha'] = opt.lora_alpha
    if hasattr(opt, 'lora_targets'):
        lora_args['lora_targets'] = opt.lora_targets.split(',') if opt.lora_targets else None
    # model = get_model(opt.arch, **lora_args)
    model = get_model(opt.arch)

    state_dict = torch.load(opt.ckpt, map_location='cpu')['model']
    #model.fc.load_state_dict(state_dict)
    model.load_state_dict(state_dict)
    print ("Model loaded..")
    model.eval()
    model.cuda(opt.gpu_id)

    if (opt.real_path == None) or (opt.fake_path == None) or (opt.data_mode == None):
        dataset_paths = DATASET_PATHS
    else:
        dataset_paths = []
        real_path_list = opt.real_path.split(',')
        fake_path_list = opt.fake_path.split(',')
        key_list = opt.key.split(',')

        for i in range(len(real_path_list)):
            dataset_paths.append( dict(real_path=real_path_list[i], fake_path=fake_path_list[i], data_mode=opt.data_mode, key=key_list[i]) )

        # dataset_paths = [ dict(real_path=opt.real_path, fake_path=opt.fake_path, data_mode=opt.data_mode, key=opt.key) ]

    print(dataset_paths)
    
    for dataset_path in (dataset_paths):
        set_seed()

        dataset = RealFakeDataset(  dataset_path['real_path'], 
                                    dataset_path['fake_path'], 
                                    dataset_path['data_mode'], 
                                    opt.max_sample, 
                                    opt.arch,
                                    jpeg_quality=opt.jpeg_quality, 
                                    gaussian_sigma=opt.gaussian_sigma,
                                    resolution_thres=opt.resolution_thres,
                                    )

        loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
        save_dir = os.path.join(opt.result_folder, f"bad_case/{dataset_path['key']}")
        ap, r_acc0, f_acc0, acc0, r_acc1, f_acc1, acc1, best_thres = validate(model, loader, find_thres=True, gpu_id=opt.gpu_id, save_incorrect=opt.save_bad_case, save_dir=save_dir)

        print("(Val {}) r_acc: {}; f_acc: {}, acc: {}, ap: {}".format(dataset_path['key'], r_acc0, f_acc0, acc0, ap))

        csv_file = os.path.join(opt.result_folder, 'result.csv')
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([dataset_path['key'], r_acc0, f_acc0, acc0, ap])

        with open( os.path.join(opt.result_folder,'ap.txt'), 'a') as f:
            f.write(dataset_path['key']+': ' + str(round(ap*100, 2))+'\n' )

        with open( os.path.join(opt.result_folder,'acc0.txt'), 'a') as f:
            f.write(dataset_path['key']+': ' + str(round(r_acc0*100, 2))+'  '+str(round(f_acc0*100, 2))+'  '+str(round(acc0*100, 2))+'\n' )

    # 所有数据集测试完成后进行csv转置
    import pandas as pd

    df = pd.read_csv(csv_file, header=None)

    df.T.to_csv(csv_file, index=False, header=False)


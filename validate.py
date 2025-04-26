import argparse
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
import random
import shutil
from scipy.ndimage.filters import gaussian_filter
import yaml
import time

SEED = 42
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
        print("Length of dataset: %d" % (len(loader)))
        # Get image paths from dataset
        if save_incorrect:
            paths = loader.dataset.data_list if hasattr(loader.dataset, 'data_list') else []
        img_idx = 0
        incorrect_fake_indices = []
        for img, label in loader:
            if gpu_id is not None:
                in_tens = img.cuda(gpu_id)
            else:
                in_tens = img.cuda()
            batch_preds = model(in_tens).sigmoid().flatten().tolist()
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
    # Get AP 
    ap = average_precision_score(y_true, y_pred)
    # Acc based on 0.5
    r_acc0, f_acc0, acc0 = calculate_acc(y_true, y_pred, 0.5)
    # Save misclassified fake images
    if save_incorrect and len(incorrect_fake_indices) > 0 and len(paths) > 0:
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


def read_images_in_dir(rootdir, must_contain, exts=["PNG", "png", "jpg", "JPEG", "jpeg", "bmp"]):
    out = []
    # Only check files in the root directory, not in subdirectories
    if os.path.exists(rootdir):
        for file in os.listdir(rootdir):
            file_path = os.path.join(rootdir, file)
            # Only include files (not directories) with matching extensions
            if os.path.isfile(file_path) and any(file.lower().endswith(ext.lower()) for ext in exts) and (must_contain in file_path):
                out.append(file_path)
    return out



def get_list(path, must_contain=''):
    if not os.path.exists(path):
        print(f"Warning: Path {path} does not exist")
        return []
    image_list = read_images_in_dir(path, must_contain)
    return image_list


class RealFakeDataset(Dataset):
    def __init__(self, data_path, label,
                      max_sample,
                      arch,
                      jpeg_quality=None,
                      gaussian_sigma=None,
                      resolution_thres=None):

        self.jpeg_quality = jpeg_quality
        self.gaussian_sigma = gaussian_sigma
        self.resolution_thres = resolution_thres
        
        # = = = = = = data path = = = = = = = = = # 
        if type(data_path) == str:
            self.data_list = self.read_path(data_path, max_sample)
        else:
            print(f"wrong in loading {data_path}")
            self.data_list = []
            
        # = = = = = =  label = = = = = = = = = # 
        self.label = label

        stat_from = "imagenet" if arch.lower().startswith("imagenet") else "clip"
        self.transform = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[stat_from], std=STD[stat_from]),
        ])
        
    def is_problematic_image(self, img):
        """Check if image is pure black/white or has very low variance"""
        try:
            img_array = np.array(img)
            # Check if image is too small
            if img_array.shape[0] < 10 or img_array.shape[1] < 10:
                return True
                
            # Check if image is pure color/black/white
            r_var = np.var(img_array[:,:,0])
            g_var = np.var(img_array[:,:,1])
            b_var = np.var(img_array[:,:,2])
            
            total_var = r_var + g_var + b_var
            if total_var < 10.0:  # Threshold for determining a "pure color" image
                return True
            
            # Check if image is mostly black or white
            mean_value = np.mean(img_array)
            if (mean_value < 5 or mean_value > 250) and np.var(img_array) < 100:
                return True
                
            return False
        except:
            return True  # If any error occurs, consider it problematic
        
    def read_path(self, data_path, max_sample):
        if 'GenEval' in data_path and 'GPT-4o' not in data_path:
            data_list = get_list(data_path, must_contain='_0.png')
        else:
            data_list = get_list(data_path, must_contain='')

        def filter_by_resolution(image_list):
            if self.resolution_thres is None:
                return image_list
                
            filtered_list = []
            for img_path in tqdm(image_list, desc="Filtering by resolution"):
                try:
                    with Image.open(img_path) as img:
                        w, h = img.size
                        area = w * h
                        if self.resolution_thres[0] <= area <= self.resolution_thres[1]:
                            filtered_list.append(img_path)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
            return filtered_list

        # Apply resolution filtering
        data_list = filter_by_resolution(data_list)

        if len(data_list) == 0:
            print(f"Warning: After resolution filtering, image count is 0 for path {data_path}")
            return []

        if max_sample is not None and max_sample>0:
            if (max_sample > len(data_list)):
                print(f"Not enough images, max_sample set to {len(data_list)}")
            random.shuffle(data_list)
            data_list = data_list[0:min(max_sample, len(data_list))]
        return data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx], self.label
        
        try:
            img = Image.open(img_path).convert("RGB")
            
            # Check if image is problematic (pure black, etc.)
            if self.is_problematic_image(img):
                print(f"Problematic image detected: {img_path}")
                return None
            
            if self.gaussian_sigma is not None:
                img = gaussian_blur(img, self.gaussian_sigma) 
            if self.jpeg_quality is not None:
                img = png2jpg(img, self.jpeg_quality)

            img = self.transform(img)
            return img, label
            
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            return None


def check_all_paths_exist(config):
    """
    Check if all paths in the configuration exist.
    Returns a tuple (all_exist, missing_paths) where all_exist is a boolean
    and missing_paths is a list of paths that don't exist.
    """
    missing_paths = []
    
    def check_path(path, path_type, dataset, subdataset=None):
        if not os.path.exists(path):
            location = f"{dataset}/{subdataset}" if subdataset else dataset
            missing_paths.append(f"{path_type} path for {location}: {path}")
            return False
        return True
    
    # Check DRCT paths
    if 'DRCT' in config:
        drct_config = config['DRCT']
        
        # Check common real path
        if 'real' in drct_config:
            check_path(drct_config['real'], 'Real', 'DRCT')
        
        # Check fake paths for subdatasets
        for sub_dataset, sub_config in drct_config.items():
            if sub_dataset != 'real' and isinstance(sub_config, dict) and 'fake' in sub_config:
                check_path(sub_config['fake'], 'Fake', 'DRCT', sub_dataset)
    
    # Check GenImage paths
    if 'GenImage' in config:
        genimage_config = config['GenImage']
        
        for sub_dataset, sub_config in genimage_config.items():
            if isinstance(sub_config, dict):
                if 'real' in sub_config:
                    check_path(sub_config['real'], 'Real', 'GenImage', sub_dataset)
                if 'fake' in sub_config:
                    check_path(sub_config['fake'], 'Fake', 'GenImage', sub_dataset)
    
    # Check Chameleon paths
    if 'Chameleon' in config:
        chameleon_config = config['Chameleon']
        
        for sub_dataset, sub_config in chameleon_config.items():
            if isinstance(sub_config, dict):
                if 'real' in sub_config:
                    check_path(sub_config['real'], 'Real', 'Chameleon', sub_dataset)
                if 'fake' in sub_config:
                    check_path(sub_config['fake'], 'Fake', 'Chameleon', sub_dataset)
    
    # Check GenEval paths
    if 'GenEval' in config:
        geneval_config = config['GenEval']
        
        for sub_dataset, fake_path in geneval_config.items():
            if isinstance(fake_path, str):
                check_path(fake_path, 'Fake', 'GenEval', sub_dataset)
                
                
    # Check RobustLDM paths
    if 'RobustLDM' in config:
        robustldm_config = config['RobustLDM']

        for sub_dataset, sub_config in robustldm_config.items():
            if isinstance(sub_config, dict):
                if 'real' in sub_config:
                    check_path(sub_config['real'], 'Real', 'RobustLDM', sub_dataset)
                if 'fake' in sub_config:
                    check_path(sub_config['fake'], 'Fake', 'RobustLDM', sub_dataset)

    
    return len(missing_paths) == 0, missing_paths

def collate_fn_filter_none(batch):
    """
    Custom collate function that filters out None values from batch
    """
    # Filter out None entries
    batch = [b for b in batch if b is not None]
    
    if len(batch) == 0:
        # If all items in batch are None, return empty tensors
        return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.int64)
    
    # Default collate for non-None entries
    images, labels = zip(*batch)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    
    return images, labels

def print_separator(title=None):
    """Print a separator line with optional title"""
    line = "=" * 80
    if title:
        print(f"\n{line}\n{title}\n{line}")
    else:
        print(f"\n{line}\n")

import os
import torch
import yaml
import csv
import time
from validate import validate, RealFakeDataset, collate_fn_filter_none

def validate_and_save_results(model, config_path, result_file, iteration=None, epoch=None, 
                             max_sample=500, batch_size=128, 
                             gpu_id=0, save_bad_case=False, 
                             jpeg_quality=None, gaussian_sigma=None, resolution_thres=None, 
                             arch='res50'):
    """
    Validates model on datasets defined in config file and saves results
    
    Args:
        model: The model to validate
        config_path: Path to YAML config file with dataset definitions
        result_file: Path to the CSV file for saving results
        iteration: Current training iteration (optional)
        epoch: Current epoch (optional)
        max_sample: Maximum number of samples to test for each dataset
        batch_size: Batch size for testing
        gpu_id: GPU ID to use
        save_bad_case: Whether to save misclassified examples
        jpeg_quality: JPEG quality for robustness testing
        gaussian_sigma: Gaussian sigma for robustness testing
        resolution_thres: Resolution threshold range
        arch: Model architecture name
    
    Returns:
        Dictionary containing validation results
    """
    # Load config from YAML file
    with open(config_path, 'r') as f:
        dataset_configs = yaml.safe_load(f)
    
    # Dictionary to store results
    results_dict = {}
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create result directory if it doesn't exist
    result_dir = os.path.dirname(result_file)
    os.makedirs(result_dir, exist_ok=True)
    
    # Function to process dataset validation
    def process_dataset(dataset_name, sub_dataset, real_path, fake_path):
        print(f"\n{'-' * 50}")
        print(f"Evaluating {dataset_name}/{sub_dataset}")
        print(f"{'-' * 50}")
        
        # Create real dataset if provided
        if real_path:
            real_dataset = RealFakeDataset(
                data_path=real_path,
                label=0,  # Real images have label 0
                max_sample=max_sample,
                arch=arch,
                jpeg_quality=jpeg_quality,
                gaussian_sigma=gaussian_sigma,
                resolution_thres=resolution_thres,
            )
            
            # Skip if not enough images
            if len(real_dataset) == 0:
                print(f"Warning: No real images found for {dataset_name}/{sub_dataset}")
                r_acc_real = None
            else:
                # Validate real dataset
                real_loader = torch.utils.data.DataLoader(
                    real_dataset, batch_size=batch_size, shuffle=False, 
                    num_workers=4, collate_fn=collate_fn_filter_none
                )
                ap_real, r_acc_real, f_acc_real, acc_real = validate(
                    model, real_loader, find_thres=False, gpu_id=gpu_id, 
                    save_incorrect=False
                )
        else:
            # If no real path, set accuracy to None
            r_acc_real = None
        
        # Create and validate fake dataset if provided
        if fake_path:
            fake_dataset = RealFakeDataset(
                data_path=fake_path,
                label=1,  # Fake images have label 1
                max_sample=max_sample,
                arch=arch,
                jpeg_quality=jpeg_quality,
                gaussian_sigma=gaussian_sigma,
                resolution_thres=resolution_thres,
            )
            
            # Skip if not enough images
            if len(fake_dataset) == 0:
                print(f"Warning: No fake images found for {dataset_name}/{sub_dataset}")
                f_acc_fake = None
            else:
                # Set up bad case saving directory if needed
                save_dir = None
                if save_bad_case:
                    save_dir = os.path.join(result_dir, f"bad_case/{dataset_name}_{sub_dataset}")
                
                # Validate fake dataset
                fake_loader = torch.utils.data.DataLoader(
                    fake_dataset, batch_size=batch_size, shuffle=False, 
                    num_workers=4, collate_fn=collate_fn_filter_none
                )
                ap_fake, r_acc_fake, f_acc_fake, acc_fake = validate(
                    model, fake_loader, find_thres=False, gpu_id=gpu_id, 
                    save_incorrect=save_bad_case, save_dir=save_dir
                )
        else:
            # If no fake path, set accuracy to None
            f_acc_fake = None
        
        # Calculate average accuracy if both real and fake are available
        if r_acc_real is not None and f_acc_fake is not None:
            avg_acc = (r_acc_real + f_acc_fake) / 2
            print(f"{dataset_name}/{sub_dataset} - Real Acc: {r_acc_real:.4f}, Fake Acc: {f_acc_fake:.4f}, Avg: {avg_acc:.4f}")
        elif r_acc_real is not None:
            avg_acc = None  # Don't calculate average if only real is available
            print(f"{dataset_name}/{sub_dataset} - Real Acc: {r_acc_real:.4f}")
        elif f_acc_fake is not None:
            avg_acc = None  # Don't calculate average if only fake is available
            print(f"{dataset_name}/{sub_dataset} - Fake Acc: {f_acc_fake:.4f}")
        else:
            avg_acc = None
            print(f"{dataset_name}/{sub_dataset} - No accuracy data")
        
        return (r_acc_real, f_acc_fake, avg_acc)
    
    # Process datasets according to config
    
    # Process DRCT dataset
    if 'DRCT' in dataset_configs:
        drct_config = dataset_configs['DRCT']
        real_path = drct_config.get('real')
        
        for sub_dataset, sub_path in drct_config.items():
            if sub_dataset == 'real':
                continue  # Skip the real path entry
            
            if isinstance(sub_path, dict) and 'fake' in sub_path:
                fake_path = sub_path['fake']
                results = process_dataset('DRCT', sub_dataset, real_path, fake_path)
                if results:
                    dataset_key = f"DRCT-{sub_dataset}"
                    results_dict[dataset_key] = results
    
    # Process GenImage dataset
    if 'GenImage' in dataset_configs:
        genimage_config = dataset_configs['GenImage']
        
        for sub_dataset, sub_config in genimage_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get('real')
                fake_path = sub_config.get('fake')
                results = process_dataset('GenImage', sub_dataset, real_path, fake_path)
                if results:
                    dataset_key = f"GenImage-{sub_dataset}"
                    results_dict[dataset_key] = results
    
    # Process Chameleon dataset
    if 'Chameleon' in dataset_configs:
        chameleon_config = dataset_configs['Chameleon']
        
        for sub_dataset, sub_config in chameleon_config.items():
            if isinstance(sub_config, dict):
                real_path = sub_config.get('real')
                fake_path = sub_config.get('fake')
                results = process_dataset('Chameleon', sub_dataset, real_path, fake_path)
                if results:
                    dataset_key = f"Chameleon-{sub_dataset}"
                    results_dict[dataset_key] = results
    
    # Process GenEval dataset
    if 'GenEval' in dataset_configs:
        geneval_config = dataset_configs['GenEval']
        
        for sub_dataset, sub_path in geneval_config.items():
            if isinstance(sub_path, dict) and 'fake' in sub_path:
                fake_path = sub_path['fake']
                results = process_dataset('GenEval', sub_dataset, None, fake_path)
                if results:
                    dataset_key = f"GenEval-{sub_dataset}"
                    results_dict[dataset_key] = results
    
    # Get sorted dataset keys for consistent column order
    dataset_keys = results_dict.keys()
    
    # Check if results file exists, create with header if not
    file_exists = os.path.isfile(result_file)
    
    # Save results to CSV file with transposed format (metrics as rows)
    with open(result_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file is new
        if not file_exists:
            header = ['Timestamp', 'Iteration', 'Epoch', 'MetricType'] + dataset_keys
            writer.writerow(header)
        
        # Write real accuracy row
        real_row = [timestamp, iteration if iteration is not None else '', epoch if epoch is not None else '', 'RealAcc']
        for key in dataset_keys:
            r_acc, _, _ = results_dict[key]
            real_row.append(r_acc if r_acc is not None else '')
        writer.writerow(real_row)
        
        # Write fake accuracy row
        fake_row = [timestamp, iteration if iteration is not None else '', epoch if epoch is not None else '', 'FakeAcc']
        for key in dataset_keys:
            _, f_acc, _ = results_dict[key]
            fake_row.append(f_acc if f_acc is not None else '')
        writer.writerow(fake_row)
        
        # Write average accuracy row
        avg_row = [timestamp, iteration if iteration is not None else '', epoch if epoch is not None else '', 'AvgAcc']
        for key in dataset_keys:
            _, _, avg_acc = results_dict[key]
            # Only write average if we have both real and fake accuracies
            if avg_acc is not None:
                avg_row.append(avg_acc)
            else:
                # Calculate average for datasets with both metrics
                r_acc, f_acc, _ = results_dict[key]
                if r_acc is not None and f_acc is not None:
                    avg_row.append((r_acc + f_acc) / 2)
                else:
                    avg_row.append('')
        writer.writerow(avg_row)
    
    print(f"Results saved to {result_file}")
    
    return results_dict
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--max_sample', type=int, default=-1, help='Only check this number of images for both fake/real')
    parser.add_argument('--arch', type=str, default='res50')
    parser.add_argument('--ckpt', type=str, default='./pretrained_weights/fc_weights.pth')
    parser.add_argument('--result_folder', type=str, default='./result', help='')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--jpeg_quality', type=int, default=None, help="100, 90, 80, ... 30. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--gaussian_sigma', type=int, default=None, help="0,1,2,3,4. Used to test robustness of our model. Not apply if None")
    parser.add_argument('--resolution_thres', type=int, nargs=2, default=None)
    parser.add_argument('--gpu_id', type=int, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--save_bad_case', action="store_true")
    parser.add_argument('--skip_path_check', action="store_true", help="Skip checking if paths exist")
    parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=float, default=1.0, help='LoRA scaling factor')
    parser.add_argument('--lora_targets', type=str, default=None, help='LoRA trainable targets')

    opt = parser.parse_args()
    
    if os.path.exists(opt.result_folder):
        shutil.rmtree(opt.result_folder)
    os.makedirs(opt.result_folder, exist_ok=True)

    # Load config from YAML file
    with open(opt.config, 'r') as f:
        dataset_configs = yaml.safe_load(f)
    
    # Check if all paths exist
    if not opt.skip_path_check:
        print("Checking if all paths in the configuration exist...")
        all_paths_exist, missing_paths = check_all_paths_exist(dataset_configs)
        input()
        
        if not all_paths_exist:
            print("\n===== MISSING PATHS =====")
            for path in missing_paths:
                print(path)
            
            # Write missing paths to file
            missing_paths_file = os.path.join(opt.result_folder, 'missing_paths.txt')
            with open(missing_paths_file, 'w') as f:
                f.write("The following paths don't exist:\n")
                for path in missing_paths:
                    f.write(f"{path}\n")
            
            print(f"\nMissing paths written to {missing_paths_file}")
            print("Exiting due to missing paths. Use --skip_path_check to run anyway.")
            sys.exit(1)
        else:
            print("All paths in the configuration exist.")
    
    lora_args = {}
    if hasattr(opt, 'lora_rank'):
        lora_args['lora_rank'] = opt.lora_rank
    if hasattr(opt, 'lora_alpha'):
        lora_args['lora_alpha'] = opt.lora_alpha
    if hasattr(opt, 'lora_targets'):
        lora_args['lora_targets'] = opt.lora_targets.split(',') if opt.lora_targets else None
    
    model = get_model(opt.arch)

    state_dict = torch.load(opt.ckpt, map_location='cpu')['model']
    model.load_state_dict(state_dict)
    print("Model loaded..")
    model.eval()
    model.cuda(opt.gpu_id)

    # Dictionary to store results in the format (dataset, subdataset) -> (real_acc, fake_acc)
    results_dict = {}
    
    # Process datasets according to config
    set_seed()
    
    # ===== PROCESS DRCT DATASET =====
    if 'DRCT' in dataset_configs:
        print_separator("PROCESSING DRCT DATASET")
        drct_config = dataset_configs['DRCT']
        
        # Get the common real path for all DRCT subdatasets
        real_path = drct_config.get('real')
        if real_path:
            # Create dataset for DRCT real images
            real_dataset = RealFakeDataset(
                data_path=real_path,
                label=0,  # Real images have label 0
                max_sample=opt.max_sample,
                arch=opt.arch,
                jpeg_quality=opt.jpeg_quality,
                gaussian_sigma=opt.gaussian_sigma,
                resolution_thres=opt.resolution_thres,
            )
            
            # Validate real dataset to get real accuracy
            real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn_filter_none)
            ap_real, r_acc_real, f_acc_real, acc_real = validate(
                model, real_loader, find_thres=False, gpu_id=opt.gpu_id, 
                save_incorrect=False
            )
            
            print(f"DRCT Real Accuracy: {r_acc_real:.4f}")
            
            # Process each DRCT subdataset
            for sub_dataset, sub_path in drct_config.items():
                if sub_dataset == 'real':
                    continue  # Skip the real path entry

                if isinstance(sub_path, dict) and 'fake' in sub_path:
                    print(f"\n{'-' * 50}")
                    print(f"Evaluating DRCT/{sub_dataset}")
                    print(f"{'-' * 50}")
                    
                if isinstance(sub_path, dict) and 'fake' in sub_path:
                    fake_path = sub_path['fake']
                    
                    # Create dataset for fake images
                    fake_dataset = RealFakeDataset(
                        data_path=fake_path,
                        label=1,  # Fake images have label 1
                        max_sample=opt.max_sample,
                        arch=opt.arch,
                        jpeg_quality=opt.jpeg_quality,
                        gaussian_sigma=opt.gaussian_sigma,
                        resolution_thres=opt.resolution_thres,
                    )
                    
                    if len(fake_dataset) == 0:
                        print(f"Warning: No images found for DRCT/{sub_dataset}")
                        continue
                        
                    fake_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn_filter_none)
                    save_dir = os.path.join(opt.result_folder, f"bad_case/DRCT_{sub_dataset}")
                    ap_fake, r_acc_fake, f_acc_fake, acc_fake = validate(
                        model, fake_loader, find_thres=False, gpu_id=opt.gpu_id, 
                        save_incorrect=opt.save_bad_case, save_dir=save_dir
                    )
                    
                    print(f"DRCT/{sub_dataset} - Real Acc: {r_acc_real:.4f}, Fake Acc: {f_acc_fake:.4f}")
                    
                    # Store results
                    results_dict[('DRCT', sub_dataset)] = (r_acc_real, f_acc_fake)
    
    # ===== PROCESS GenImage DATASET =====
    if 'GenImage' in dataset_configs:
        print_separator("PROCESSING GenImage DATASET")
        genimage_config = dataset_configs['GenImage']
        
        for sub_dataset, sub_config in genimage_config.items():
            
            print(f"\n{'-' * 50}")
            print(f"Evaluating GenImage/{sub_dataset}")
            print(f"{'-' * 50}")
            
            real_path = sub_config.get('real')
            fake_path = sub_config.get('fake')
            
            if real_path and fake_path:
                # Create datasets for real and fake images
                real_dataset = RealFakeDataset(
                    data_path=real_path,
                    label=0,  # Real images have label 0
                    max_sample=opt.max_sample,
                    arch=opt.arch,
                    jpeg_quality=opt.jpeg_quality,
                    gaussian_sigma=opt.gaussian_sigma,
                    resolution_thres=opt.resolution_thres,
                )
                
                fake_dataset = RealFakeDataset(
                    data_path=fake_path,
                    label=1,  # Fake images have label 1
                    max_sample=opt.max_sample,
                    arch=opt.arch,
                    jpeg_quality=opt.jpeg_quality,
                    gaussian_sigma=opt.gaussian_sigma,
                    resolution_thres=opt.resolution_thres,
                )
                
                if len(real_dataset) == 0 or len(fake_dataset) == 0:
                    print(f"Warning: Not enough images for GenImage/{sub_dataset}")
                    continue
                
                # Validate real dataset
                real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8,collate_fn=collate_fn_filter_none)
                ap_real, r_acc_real, f_acc_real, acc_real = validate(
                    model, real_loader, find_thres=False, gpu_id=opt.gpu_id, 
                    save_incorrect=False
                )
                
                # Validate fake dataset
                fake_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn_filter_none)
                save_dir = os.path.join(opt.result_folder, f"bad_case/GenImage_{sub_dataset}")
                ap_fake, r_acc_fake, f_acc_fake, acc_fake = validate(
                    model, fake_loader, find_thres=False, gpu_id=opt.gpu_id, 
                    save_incorrect=opt.save_bad_case, save_dir=save_dir
                )
                
                print(f"GenImage/{sub_dataset} - Real Acc: {r_acc_real:.4f}, Fake Acc: {f_acc_fake:.4f}")
                
                # Store results
                results_dict[('GenImage', sub_dataset)] = (r_acc_real, f_acc_fake)
    
    # ===== PROCESS Chameleon DATASET =====
    if 'Chameleon' in dataset_configs:
        print_separator("PROCESSING Chameleon DATASET")
        chameleon_config = dataset_configs['Chameleon']
        
        for sub_dataset, sub_config in chameleon_config.items():
            print(f"\n{'-' * 50}")
            print(f"Evaluating Chameleon/{sub_dataset}")
            print(f"{'-' * 50}")
            
            real_path = sub_config.get('real')
            fake_path = sub_config.get('fake')
            
            if real_path and fake_path:
                # Create datasets for real and fake images
                real_dataset = RealFakeDataset(
                    data_path=real_path,
                    label=0,  # Real images have label 0
                    max_sample=opt.max_sample,
                    arch=opt.arch,
                    jpeg_quality=opt.jpeg_quality,
                    gaussian_sigma=opt.gaussian_sigma,
                    resolution_thres=opt.resolution_thres,
                )
                
                fake_dataset = RealFakeDataset(
                    data_path=fake_path,
                    label=1,  # Fake images have label 1
                    max_sample=opt.max_sample,
                    arch=opt.arch,
                    jpeg_quality=opt.jpeg_quality,
                    gaussian_sigma=opt.gaussian_sigma,
                    resolution_thres=opt.resolution_thres,
                )
                
                if len(real_dataset) == 0 or len(fake_dataset) == 0:
                    print(f"Warning: Not enough images for Chameleon/{sub_dataset}")
                    continue
                
                # Validate real dataset
                real_loader = torch.utils.data.DataLoader(real_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn_filter_none)
                ap_real, r_acc_real, f_acc_real, acc_real = validate(
                    model, real_loader, find_thres=False, gpu_id=opt.gpu_id, 
                    save_incorrect=False
                )
                
                # Validate fake dataset
                fake_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn_filter_none)
                save_dir = os.path.join(opt.result_folder, f"bad_case/Chameleon_{sub_dataset}")
                ap_fake, r_acc_fake, f_acc_fake, acc_fake = validate(
                    model, fake_loader, find_thres=False, gpu_id=opt.gpu_id, 
                    save_incorrect=opt.save_bad_case, save_dir=save_dir
                )
                
                print(f"Chameleon/{sub_dataset} - Real Acc: {r_acc_real:.4f}, Fake Acc: {f_acc_fake:.4f}")
                
                # Store results
                results_dict[('Chameleon', sub_dataset)] = (r_acc_real, f_acc_fake)
    
    # ===== PROCESS GenEval DATASET =====
    if 'GenEval' in dataset_configs:
        print_separator("PROCESSING GenEval DATASET")
        geneval_config = dataset_configs['GenEval']
        
        for sub_dataset, fake_path in geneval_config.items():
            print(f"\n{'-' * 50}")
            print(f"Evaluating GenEval/{sub_dataset}")
            print(f"{'-' * 50}")
            
            
            if isinstance(fake_path['fake'], str):
                # Create dataset for fake images
                fake_dataset = RealFakeDataset(
                    data_path=fake_path['fake'],
                    label=1,  # Fake images have label 1
                    max_sample=opt.max_sample,
                    arch=opt.arch,
                    jpeg_quality=opt.jpeg_quality,
                    gaussian_sigma=opt.gaussian_sigma,
                    resolution_thres=opt.resolution_thres,
                )
                
                if len(fake_dataset) == 0:
                    print(f"Warning: No images found for GenEval/{sub_dataset}")
                    continue
                    
                fake_loader = torch.utils.data.DataLoader(fake_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=8, collate_fn=collate_fn_filter_none)
                save_dir = os.path.join(opt.result_folder, f"bad_case/GenEval_{sub_dataset}")
                ap_fake, r_acc_fake, f_acc_fake, acc_fake = validate(
                    model, fake_loader, find_thres=False, gpu_id=opt.gpu_id, 
                    save_incorrect=opt.save_bad_case, save_dir=save_dir
                )
                
                # For GenEval, set real accuracy equal to fake accuracy
                print(f"GenEval/{sub_dataset} - Fake Acc: {f_acc_fake:.4f}")
                
                # Store results
                results_dict[('GenEval', sub_dataset)] = (f_acc_fake, f_acc_fake)  # Set real_acc = fake_acc for GenEval
    
    # Save results to CSV
    csv_file = os.path.join(opt.result_folder, 'result_dict.csv')
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Dataset/Subdataset', 'Real Accuracy', 'Fake Accuracy'])
        
        for (dataset, subdataset), (real_acc, fake_acc) in results_dict.items():
            writer.writerow([f"{dataset}-{subdataset}", real_acc, fake_acc])
    print(f"Results saved to {csv_file}")
    
    # Transpose and save results to CSV
    transpose_file = os.path.join(opt.result_folder, 'result_dict_transposed.csv')
    with open(transpose_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # First row: subdataset names
        subdatasets = [f"{dataset}-{subdataset}" for (dataset, subdataset) in results_dict.keys()]
        writer.writerow(['Metric'] + subdatasets)
        
        # Second row: real accuracies
        real_accs = [real_acc for (real_acc, _) in results_dict.values()]
        writer.writerow(['Real Accuracy'] + real_accs)
        
        # Third row: fake accuracies
        fake_accs = [fake_acc for (_, fake_acc) in results_dict.values()]
        writer.writerow(['Fake Accuracy'] + fake_accs)
        
        # Fourth row: average accuracies
        avg_accs = [(real_acc + fake_acc) / 2 for (real_acc, fake_acc) in results_dict.values()]
        writer.writerow(['Average Accuracy'] + avg_accs)

    print(f"Transposed results saved to {transpose_file}")

    # Also save as a dictionary format for easier reference
    dict_file = os.path.join(opt.result_folder, 'result_dict.py')
    with open(dict_file, 'w') as f:
        f.write("results_dict = {\n")
        for (dataset, subdataset), (real_acc, fake_acc) in results_dict.items():
            f.write(f"    ('{dataset}', '{subdataset}'): ({real_acc:.4f}, {fake_acc:.4f}),\n")
        f.write("}\n")
    
    print(f"Dictionary saved to {dict_file}")
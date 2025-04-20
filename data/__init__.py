import torch
import numpy as np
from torch.utils.data.sampler import WeightedRandomSampler

from .datasets import CustomBatchSampler, RealFakeDataset, custom_collate_fn
from .transforms import create_transformations
from .vae import VAERebuilder

def get_bal_sampler(dataset):
    targets = []
    for d in dataset.datasets:
        targets.extend(d.targets)

    ratio = np.bincount(targets)
    w = 1. / torch.tensor(ratio, dtype=torch.float)
    sample_weights = w[targets]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights))
    return sampler

# def create_dataloader(opt, preprocess=None, return_dataset=False):
#     shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False
#     dataset = RealFakeDataset(opt)
#     if '2b' in opt.arch:
#         dataset.transform = preprocess
#     sampler = get_bal_sampler(dataset) if opt.class_bal else None
#     print(len(dataset))
#     if return_dataset:
#         return dataset
    
#     data_loader = torch.utils.data.DataLoader(dataset,
#                                               batch_size=opt.batch_size,
#                                               shuffle=shuffle if sampler is None else False,
#                                               sampler=sampler,
#                                               num_workers=int(opt.num_threads),
#                                               pin_memory=True,
#                                               drop_last=opt.isTrain,)
#     return data_loader

vae_path_list = [
    "stabilityai/sdxl-vae",
    "stabilityai/sd-vae-ft-mse",
    "stabilityai/sd-vae-ft-ema",
]

def create_dataloader(opt, preprocess=None, return_dataset=False):
    if opt.vae:
        trans_func = create_transformations(opt)

        VAE = []
            
        for vae_path in vae_path_list:
            # vae = VAETransform(opt.gpu_ids[0], vae_model_path=vae_path, batch_process=batch_vae)
            vae = VAERebuilder(opt.gpu_ids[0], vae_model_path=vae_path)
            VAE.append(vae)
        # XL_VAE = VAETransform(opt.gpu_ids[0], vae_model_path="stabilityai/sdxl-vae")
        # MSE_VAE = VAETransform(opt.gpu_ids[0], vae_model_path="stabilityai/sd-vae-ft-mse")
        # VAE = [XL_VAE, MSE_VAE]
        sampler = CustomBatchSampler(opt, VAE, trans_func)

        return sampler
    
    else:
        shuffle = not opt.serial_batches if (opt.isTrain and not opt.class_bal) else False

        fake_num = len(opt.fake_list_path.split(","))
        batch_size = opt.batch_size // ( 1 +fake_num )
        if opt.resize_vae:
            batch_size = batch_size // 2

        dataset = RealFakeDataset(opt)
        if '2b' in opt.arch:
            dataset.transform = preprocess
        sampler = get_bal_sampler(dataset) if opt.class_bal else None
        print(len(dataset))
        if return_dataset:
            return dataset
        
        data_loader = torch.utils.data.DataLoader(dataset,
                                                batch_size=batch_size,
                                                shuffle=shuffle if sampler is None else False,
                                                sampler=sampler,
                                                num_workers=opt.num_threads,
                                                pin_memory=True,
                                                drop_last=opt.isTrain,
                                                collate_fn=custom_collate_fn,)
        return data_loader
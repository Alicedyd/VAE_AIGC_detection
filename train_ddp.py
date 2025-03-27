import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']
    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    
    set_seed(42 + rank)

    opt = TrainOptions().parse(print_options=(rank == 0))
    opt.gpu_ids = [local_rank]
    opt.device = torch.device(f'cuda:{local_rank}')
    

    if hasattr(opt, 'original_batch_size'):
        opt.batch_size = opt.original_batch_size // world_size
    
    if rank == 0:
        print(f"总GPU数: {world_size}, 每GPU批量大小: {opt.batch_size}")
    
    
    val_opt = get_val_opt()
    val_opt.gpu_ids = [local_rank]
    
    model = Trainer(opt)
    
    model.model = DDP(model.model, device_ids=[local_rank], find_unused_parameters=True)
    
    
    train_dataset = create_dataloader(opt, return_dataset=True)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=opt.batch_size,
        sampler=train_sampler,
        num_workers=opt.num_threads,
        pin_memory=True
    )
    
    
    val_loader = None
    if rank == 0:
        val_loader = create_dataloader(val_opt)
    
    train_writer = None
    val_writer = None
    if rank == 0:
        train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
        val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
    
    
    early_stopping = None
    if rank == 0:
        early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    
    
    start_time = time.time()
    
    for epoch in range(opt.niter):
        
        train_sampler.set_epoch(epoch)
        
    
        for i, data in enumerate(train_loader):
            model.total_steps += 1
            
            model.set_input(data)
            model.optimize_parameters()
            
            if rank == 0 and model.total_steps % opt.loss_freq == 0:
                print(f"Train loss: {model.loss} at step: {model.total_steps}")
                train_writer.add_scalar('loss', model.loss, model.total_steps)
                print(f"Iter time: {(time.time()-start_time)/model.total_steps}")
        
        dist.barrier()
        
        if rank == 0:
            if epoch % opt.save_epoch_freq == 0:
                print(f'保存第 {epoch} 个epoch结束时的模型')
                model.save_networks(f'model_epoch_{epoch}.pth')
            
            model.eval()
            ap, r_acc, f_acc, acc = validate(model.model.module, val_loader)
            val_writer.add_scalar('accuracy', acc, model.total_steps)
            val_writer.add_scalar('ap', ap, model.total_steps)
            print(f"(Val @ epoch {epoch}) acc: {acc}; ap: {ap}")
            
            early_stopping(acc, model)
            if early_stopping.early_stop:
                cont_train = model.adjust_learning_rate()
                if cont_train:
                    print("学习率降低10倍，继续训练...")
                    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
                else:
                    print("触发早停。")
                    break
            
            if early_stopping.is_best:
                model.save_networks('model_epoch_best.pth')
            
            model.train()
        
        dist.barrier()
    
    if rank == 0:
        if train_writer:
            train_writer.close()
        if val_writer:
            val_writer.close()
    
    dist.destroy_process_group()

if __name__ == '__main__':
    main()
import os
import time
from tensorboardX import SummaryWriter

from validate import validate, validate_and_save_results
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions

from dataset_paths import RealFakeDataset

import shutil

from dataset_paths import DATASET_PATHS

import torch
import random
import numpy as np
import yaml
import time


def set_seed(seed):
    random.seed(seed)                     # 控制 torchvision.transforms 等
    np.random.seed(seed)                  # 控制 numpy 操作
    torch.manual_seed(seed)               # 控制 torch 操作
    torch.cuda.manual_seed_all(seed)      # 多卡时也生效
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


"""Currently assumes jpg_prob, blur_prob 0 or 1"""
def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.no_resize = False
    val_opt.no_crop = False
    val_opt.serial_batches = True
    val_opt.data_label = 'val'
    val_opt.jpg_method = ['pil']

    val_opt.batch_size = 128
    val_opt.vae = False

    if len(val_opt.blur_sig) == 2:
        b_sig = val_opt.blur_sig
        val_opt.blur_sig = [(b_sig[0] + b_sig[1]) / 2]
    if len(val_opt.jpg_qual) != 1:
        j_qual = val_opt.jpg_qual
        val_opt.jpg_qual = [int((j_qual[0] + j_qual[-1]) / 2)]

    return val_opt


if __name__ == '__main__':
    # set_seed(42)
    
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
 
    model = Trainer(opt)
    
    data_loader = create_dataloader(opt)

    results_file = os.path.join("./checkpoints", opt.name, "validation_results.csv")
    config_path = "./configs/drct_genimage_chameleon_geneval.yaml"  # Add this argument to TrainOptions

    train_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "train"))
    val_writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, "val"))
        
    early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.001, verbose=True)
    start_time = time.time()
    print ("Length of data loader: %d" %(len(data_loader)))
    for epoch in range(opt.niter):
        
        # for i, data in enumerate(data_loader):

        if opt.vae:

            data_loader.set_epoch_start()

            for i in range(len(data_loader)):
                model.total_steps += 1

                inputs, labels = next(data_loader)

                model.set_input([inputs, labels])
                model.optimize_parameters()

                if model.total_steps % opt.loss_freq == 0:
                    print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                    train_writer.add_scalar('loss', model.loss, model.total_steps)
                    print("Iter time: ", ((time.time()-start_time)/model.total_steps)  )

                if model.total_steps % 4000 == 0: # save models for each 1000 steps 
                    model.save_networks('model_iters_%s.pth' % model.total_steps)

            model.finalize_epoch()

        else:

            for i, data in enumerate(data_loader):
                model.total_steps += 1

                model.set_input(data)
                model.optimize_parameters()

                if model.total_steps % opt.loss_freq == 0:
                    print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                    train_writer.add_scalar('loss', model.loss, model.total_steps)
                    print("Iter time: ", ((time.time()-start_time)/model.total_steps)  )

                if model.total_steps % 5000 == 0: # save models at these iters 
                    model.save_networks('model_iters_%s.pth' % model.total_steps)
                    
                    # Run validation and save results
                    model.eval()
                    validate_and_save_results(
                        model.model, 
                        config_path, 
                        results_file,
                        iteration=model.total_steps,
                        epoch=epoch,
                        max_sample=500,
                        batch_size=128,
                        gpu_id=opt.gpu_ids[0],
                        arch=opt.arch
                    )
                    model.train()
                    break

            model.finalize_epoch()

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks( 'model_epoch_best.pth' )
            model.save_networks( 'model_epoch_%s.pth' % epoch )

        model.eval()
        validate_and_save_results(
            model.model, 
            config_path, 
            results_file,
            iteration=model.total_steps,
            epoch=epoch,
            max_sample=500,
            batch_size=128,
            gpu_id=opt.gpu_ids[0],
            arch=opt.arch
        )
        model.train()
        # early_stopping(acc, model)
        # if early_stopping.early_stop:
        #     cont_train = model.adjust_learning_rate()
        #     if cont_train:
        #         print("Learning rate dropped by 10, continue training...")
        #         early_stopping = EarlyStopping(patience=opt.earlystop_epoch, delta=-0.002, verbose=True)
        #     else:
        #         print("Early stopping.")
        #         break
        # model.train()


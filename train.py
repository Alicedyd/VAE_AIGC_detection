import os
import time
from tensorboardX import SummaryWriter

from validate import validate
from data import create_dataloader
from earlystop import EarlyStopping
from networks.trainer import Trainer
from options.train_options import TrainOptions

import shutil


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
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
 
    model = Trainer(opt)
    
    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

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

                if model.total_steps in [100,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]: # save models at these iters 
                    model.save_networks('model_iters_%s.pth' % model.total_steps)

        else:

            for i, data in enumerate(data_loader):
                model.total_steps += 1

                model.set_input(data)
                model.optimize_parameters()

                if model.total_steps % opt.loss_freq == 0:
                    print("Train loss: {} at step: {}".format(model.loss, model.total_steps))
                    train_writer.add_scalar('loss', model.loss, model.total_steps)
                    print("Iter time: ", ((time.time()-start_time)/model.total_steps)  )

                if model.total_steps in [100,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]: # save models at these iters 
                    model.save_networks('model_iters_%s.pth' % model.total_steps)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d' % (epoch))
            model.save_networks( 'model_epoch_best.pth' )
            model.save_networks( 'model_epoch_%s.pth' % epoch )

        # Validation
        model.eval()
        ap, r_acc, f_acc, acc = validate(model.model, val_loader, gpu_id=opt.gpu_ids[0])
        val_writer.add_scalar('accuracy', acc, model.total_steps)
        val_writer.add_scalar('ap', ap, model.total_steps)
        print("(Val @ epoch {}) r_acc: {}; f_acc: {}, acc: {}, ap: {}".format(epoch, r_acc, f_acc, acc, ap))

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


# import functools
# import torch
# import torch.nn as nn
# from networks.base_model import BaseModel, init_weights
# import sys
# from models import get_model

# class Trainer(BaseModel):
#     def name(self):
#         return 'Trainer'

#     def __init__(self, opt):
#         super(Trainer, self).__init__(opt)
#         self.opt = opt  
#         self.model = get_model(opt.arch)
#         torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

#         if opt.fix_backbone:
#             params = []
#             for name, p in self.model.named_parameters():
#                 if  name=="fc.weight" or name=="fc.bias": 
#                     params.append(p) 
#                 else:
#                     p.requires_grad = False
#         else:
#             print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
#             import time 
#             time.sleep(3)
#             params = self.model.parameters()

        

#         if opt.optim == 'adam':
#             self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
#         elif opt.optim == 'sgd':
#             self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
#         else:
#             raise ValueError("optim should be [adam, sgd]")

#         self.loss_fn = nn.BCEWithLogitsLoss()

#         self.model.to(opt.gpu_ids[0])


#     def adjust_learning_rate(self, min_lr=1e-6):
#         for param_group in self.optimizer.param_groups:
#             param_group['lr'] /= 10.
#             if param_group['lr'] < min_lr:
#                 return False
#         return True


#     def set_input(self, input):
#         self.input = input[0].to(self.device)
#         self.label = input[1].to(self.device).float()


#     def forward(self):
#         self.output = self.model(self.input)
#         self.output = self.output.view(-1).unsqueeze(1)


#     def get_loss(self):
#         return self.loss_fn(self.output.squeeze(1), self.label)

#     def optimize_parameters(self):
#         self.forward()
#         self.loss = self.loss_fn(self.output.squeeze(1), self.label) 
#         self.optimizer.zero_grad()
#         self.loss.backward()
#         self.optimizer.step()

import functools
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt  
        
        
        lora_args = {}
        if hasattr(opt, 'lora_rank'):
            lora_args['lora_rank'] = opt.lora_rank
        if hasattr(opt, 'lora_alpha'):
            lora_args['lora_alpha'] = opt.lora_alpha
        if hasattr(opt, 'lora_targets'):
            lora_args['lora_targets'] = opt.lora_targets.split(',') if opt.lora_targets else None
            
        self.model = get_model(opt.arch, **lora_args)
        
        using_lora = opt.arch.startswith('CLIP-LoRA:') or opt.arch.startswith('DINOv2-LoRA:')
        if opt.arch.startswith('DINOv2-LoRA:'):
            torch.nn.init.normal_(self.model.base_model.fc.weight.data, 0.0, opt.init_gain)
        else:
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        if opt.fix_backbone or using_lora:
            
            if using_lora and hasattr(self.model, 'get_trainable_params'):
                params = self.model.get_trainable_params()
                
            else:
                params = []
                for name, p in self.model.named_parameters():
                    if name == "fc.weight" or name == "fc.bias": 
                        params.append(p) 
                    else:
                        p.requires_grad = False
        else:
            print("Your backbone is not fixed. Are you sure you want to proceed? If this is a mistake, enable the --fix_backbone command during training and rerun")
            import time 
            time.sleep(3)
            params = self.model.parameters()
        
        # print trainable params
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print(name)

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,eta_min=opt.lr * 0.001, T_max=1000)
        self.loss_fn = nn.BCEWithLogitsLoss()

        if hasattr(opt, 'device'):
            self.device = opt.device
        self.model.to(self.device)

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.label = input[1].to(self.device).float()

    def forward(self):
        self.output = self.model(self.input)
        if hasattr(self.output, 'view'):  
            self.output = self.output.view(-1).unsqueeze(1)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.forward()
        self.loss = self.loss_fn(self.output.squeeze(1), self.label) 
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        
    def train(self):
        self.model.train()
        
    def eval(self):
        self.model.eval()


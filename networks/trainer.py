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
import torchvision.transforms.functional as F

from pytorch_metric_learning import losses

class TokenWiseContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(TokenWiseContrastiveLoss, self).__init__()
        self.temperature = temperature
        
    def forward(self, token_features, token_masks):
        """
        Args:
            token_features: Tensor of shape [B, N, D] where B is batch size, 
                          N is number of tokens, D is feature dim
            token_masks: Binary tensor of shape [B, N] where 1 indicates 
                        fake tokens and 0 indicates real tokens
        """
        # Ensure proper shape
        B, N, D = token_features.shape
        token_features = token_features.reshape(-1, D)  # [B*N, D]
        
        # Create token labels based on image labels
        token_labels = token_masks.reshape(-1)  # [B*N]
        
        # Normalize features
        token_features = torch.nn.functional.normalize(token_features, p=2, dim=1)  # Use full path and correct parameters
        
        # Get real and fake token indices
        real_indices = (token_labels == 0).nonzero(as_tuple=True)[0]
        fake_indices = (token_labels == 1).nonzero(as_tuple=True)[0]
        
        # If either real or fake tokens are missing, return zero loss
        if len(real_indices) == 0 or len(fake_indices) == 0:
            return torch.tensor(0.0, device=token_features.device)
        
        # Get features of real and fake tokens
        real_features = token_features[real_indices]
        fake_features = token_features[fake_indices]
        
        # Compute pairwise similarities
        similarity_matrix = torch.matmul(real_features, fake_features.T) / self.temperature
        
        # Compute contrastive loss
        real_to_fake_max, _ = torch.max(similarity_matrix, dim=1)
        real_to_fake_loss = -torch.log(1 - torch.sigmoid(real_to_fake_max) + 1e-6).mean()
        
        fake_to_real_max, _ = torch.max(similarity_matrix.T, dim=1)
        fake_to_real_loss = -torch.log(1 - torch.sigmoid(fake_to_real_max) + 1e-6).mean()
        
        # Combined loss
        total_loss = (real_to_fake_loss + fake_to_real_loss) / 2
        
        return total_loss

class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt  
        
         # Add gradient accumulation parameters
        self.accumulation_steps = opt.accumulation_steps if hasattr(opt, 'accumulation_steps') else 1
        self.current_step = 0  # Track steps for gradient accumulation
        
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

        # Regular Contrastive Learning
        self.contrastive = opt.contrastive
        if self.contrastive:
            self.contrastive_loss_fn = losses.ContrastiveLoss(pos_margin=0.0, neg_margin=1.0)
            self.contrastive_alpha = 0.5

        # Token-wise Contrastive Learning
        self.token_contrastive = opt.token_contrastive if hasattr(opt, 'token_contrastive') else False
        if self.token_contrastive:
            self.token_contrastive_loss_fn = TokenWiseContrastiveLoss(temperature=opt.temperature if hasattr(opt, 'temperature') else 0.07)
            self.token_contrastive_alpha = opt.token_contrastive_alpha if hasattr(opt, 'token_contrastive_alpha') else 0.3

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
        # self.input = input[0].to(self.device)
        # self.label = input[1].to(self.device).float()

        # self.input = torch.cat([input["real"], input["real_resized"], input["fake"], input["fake_resized"], input["interpolate"]], dim=0).to(self.device)
        input_stack = []
        for key in ["real", "real_resized", "fake", "fake_resized"]:
            if input[key] is not None:
                input_stack.append(input[key])
        self.input = torch.cat(input_stack, dim=0).to(self.device)

        # self.label = [0] * len(input["real"]) + [0] * len(input["real_resized"]) + [1] * len(input["fake"]) + [1] * len(input["fake_resized"]) + [1] * len(input["interpolate"])
        LABELS = {
            "real": 0,
            "real_resized": 0,
            "fake": 1,
            "fake_resized": 1,
        }
        label_stack = []
        for key in ["real", "real_resized", "fake", "fake_resized"]:
            if input[key] is not None:
                label_stack += [LABELS[key]] * len(input[key])
        self.label = torch.tensor(label_stack).to(self.device).float()

    def forward(self):
        if self.contrastive and self.token_contrastive:
            # Get global features, token features, and output
            self.feature, self.token_features, self.output = self.model(self.input, return_feature=True, return_tokens=True)
        elif self.contrastive:
            # Get only global features and output
            self.feature, self.output = self.model(self.input, return_feature=True)
        elif self.token_contrastive:
            # Get only token features and output
            _, self.token_features, self.output = self.model(self.input, return_feature=False, return_tokens=True)
        else:
            # Get only output
            self.output = self.model(self.input)
        
        if hasattr(self.output, 'view'):  
            self.output = self.output.view(-1).unsqueeze(1)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.current_step += 1
        self.forward()

        # Calculate classification loss
        cls_loss = self.loss_fn(self.output.squeeze(1), self.label)
        
        # Initialize total loss with classification loss
        total_loss = cls_loss
        
        # Add global contrastive loss if enabled
        if self.contrastive:
            contrastive_loss = self.contrastive_loss_fn(self.feature, self.label)
            total_loss = (1 - self.contrastive_alpha) * total_loss + self.contrastive_alpha * contrastive_loss
        
        # Add token-wise contrastive loss if enabled
        if self.token_contrastive:
            # Create token masks based on image labels
            B = self.label.size(0)
            N = self.token_features.size(1)  # Number of tokens per image
            
            # Repeat each image label for all of its tokens
            token_masks = self.label.unsqueeze(1).repeat(1, N)
            
            token_contrastive_loss = self.token_contrastive_loss_fn(self.token_features, token_masks)
            total_loss = (1 - self.token_contrastive_alpha) * total_loss + self.token_contrastive_alpha * token_contrastive_loss
        
        # Save the final loss
        self.loss = total_loss
        
        # Apply gradient accumulation
        self.loss = self.loss / self.accumulation_steps
        self.loss.backward()


        if self.current_step % self.accumulation_steps == 0:            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()  # Reset gradients after update

        # self.optimizer.zero_grad()
        # self.loss.backward()
        # self.optimizer.step()
        # self.scheduler.step()
        
    def train(self):
        self.model.train()
        
    def eval(self):
        self.model.eval()

    # Handle remaining gradients at the end of epoch
    def finalize_epoch(self):
        if self.current_step % self.accumulation_steps != 0:
            # Update with remaining gradients
            self.optimizer.step()
            self.optimizer.zero_grad()


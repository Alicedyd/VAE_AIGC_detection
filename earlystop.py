import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation metric doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last improvement.
                            Default: 7
            verbose (bool): If True, prints a message for each improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_best = -np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.is_best = False
        
    def __call__(self, val_metric, model):
        score = val_metric
        self.is_best = False

        if self.best_score is None:
            self.best_score = score
            self.val_best = val_metric
            self.is_best = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_best = val_metric
            self.is_best = True
            self.counter = 0
import numpy as np
import torch
import os


class EarlyStopping:
    """Early stop the training if validation acc doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=0, trace_func=print, save_all_checkpoint=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
            save_all_checkpoint (bool): If True, Save the model of all checkpoints during training, requires large storage space
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_max = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.save_all_checkpoint = save_all_checkpoint

    def __call__(self, val_acc, model, path):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model, path)
            self.counter = 0

    def save_checkpoint(self, val_acc, model, path):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(f'Validation acc increased ({self.val_acc_max:.6f} --> {val_acc:.6f}).  Saving model ...')
        if not self.save_all_checkpoint:
            path = os.path.join(os.path.dirname(path), 'model.pkl')
        torch.save(model.state_dict(), path)
        self.val_acc_max = val_acc
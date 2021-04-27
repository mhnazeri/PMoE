"""Utils for nn module"""
from typing import List
import numpy as np
import torch.nn as nn
from thop import profile, clever_format


def check_grad_norm(net: nn.Module):
    """Compute and return the grad norm of all parameters of the network.
    To see gradients flowing in the network or not
    """
    total_norm = 0
    for p in list(filter(lambda p: p.grad is not None, net.parameters())):
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


def freeze(model: nn.Module, exclude: List, verbose: bool = False) -> nn.Module:
    """Freezes the layers of the model except the exclusion layer list.

    Args:
        model: (nn.Module) The model itself.
        exclude: (List) The list of layers name the you want to keep unfrozen.
        verbose: (bool) Show statistics of the model parameters.

    Returns:
        model: (nn.Module) returns the frozen model.
    """
    frozen_layers_list = []
    frozen_params_list = [len(p) for p in model.parameters() if p.requires_grad]
    if verbose:
        print(f"The model has {len([p for p in model.parameters()])} layers.")
        print(f"Before freezing the model had {sum(frozen_params_list)} parameters.")

    for name, child in model.named_parameters():
        if not any(layer in name for layer in exclude):
            frozen_layers_list.append(name)
            child.requires_grad_(False)

    frozen_params_list = [len(p) for p in model.parameters() if p.requires_grad]
    if verbose:
        print(f"{len(frozen_layers_list)} layers have been frozen.")
        print(f"After freezing the model has {sum(frozen_params_list)} parameters.")

    return model


def init_weights_normal(m: nn.Module, mean: float = 0.0, std: float = 0.5):
    """Initialize the network's weights based on normal distribution

    Args:
        m: (nn.Module) network itself
        mean: (float) mean of normal distribution
        std: (float) standard deviation for normal distribution
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, mean=mean, std=std)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, mean=mean, std=std)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.normal_(m.weight.data, mean=mean, std=std)
        nn.init.constant_(m.bias.data, 0)


def init_weights_uniform(m: nn.Module, low: float = 0.0, high: float = 1.0):
    """Initialize the network's weights based on uniform distribution

    Args:
        m: (nn.Module) network itself
        low: (float) minimum threshold for uniform distribution
        high: (float) maximum threshold for uniform distribution
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.uniform_(m.weight.data, a=low, b=high)
    elif classname.find("BatchNorm") != -1:
        nn.init.uniform_(m.weight.data, a=low, b=high)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.uniform_(m.weight.data, a=low, b=high)
        nn.init.constant_(m.bias.data, 0)


def init_weights_xavier_normal(m: nn.Module):
    """Initialize the network's weights based on xaviar normal distribution"""
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.xavier_normal_(m.weight.data)
    elif classname.find("BatchNorm") != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find("Linear") != -1:
        nn.init.xavier_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


def op_counter(model, sample):
    model.eval()
    macs, params = profile(model, inputs=(sample,))
    macs, params = clever_format([macs, params], "%.3f")
    return macs, params


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
    Code from https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

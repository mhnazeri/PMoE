"""Building block modules to use in network architecture"""
from math import log2
from collections import OrderedDict
from typing import List

import torch
import torch.nn as nn


def make_mlp(
    dims: List, act: str, l_act: bool = False, bn: bool = True, dropout: float = 0.0
):
    """Create a simple MLP with batch-norm and dropout

    Args:
        dims: (List) a list containing the dimensions of MLP
        act: (str) activation function to be used. Valid activations are [relu, tanh, sigmoid]
        l_act: (bool) whether to use activation after the last linear layer
        bn: (bool) use batch-norm or not. Default is True
        dropout: (float) dropout percentage
    """
    layers = []
    activation = {
        "relu": nn.ReLU(inplace=True),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "elu": nn.ELU(),
    }[act.lower()]

    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(in_dim, out_dim, bias=not bn))
        if i != (len(dims) - 2):
            if bn:
                layers.append(nn.BatchNorm1d(out_dim))

            layers.append(activation)

            if dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))

    if l_act:
        layers.append(activation)

    return nn.Sequential(*layers)


def conv3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """forward path in each layer with padding"""
    layer = nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

    return layer


class EfficientBlock(nn.Module):
    """ECA block based on https://ieeexplore.ieee.org/document/9156697/"""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        t = int(abs((log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)

    def forward(self, x):
        y = nn.functional.adaptive_avg_pool2d(x, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = torch.sigmoid(y)

        return x * y.expand_as(x)


class EfficientConvBlock(nn.Module):
    """ECA conv block based on https://ieeexplore.ieee.org/document/9156697/"""

    def __init__(
        self, in_ch: int, out_ch: int, stride: int = 1, gamma: int = 2, b: int = 1
    ):
        super().__init__()
        self.layer1 = nn.Sequential(
            OrderedDict(
                [
                    ("eca1", EfficientBlock(in_ch, gamma, b)),
                    (
                        "conv1",
                        nn.Sequential(
                            nn.Conv2d(
                                in_ch,
                                64,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(64),
                            nn.ReLU(inplace=True),
                        ),
                    ),
                ]
            )
        )
        self.layer2 = nn.Sequential(
            OrderedDict(
                [
                    ("eca2", EfficientBlock(64, gamma, b)),
                    (
                        "conv2",
                        nn.Sequential(
                            nn.Conv2d(
                                64,
                                out_ch,
                                kernel_size=3,
                                stride=stride,
                                padding=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(out_ch),
                            nn.ReLU(inplace=True),
                        ),
                    ),
                ]
            )
        )

    def forward(self, x):
        y = self.layer1(x)
        y = self.layer2(y)
        return y

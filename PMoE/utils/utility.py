"""General utility functions"""
from time import time
import functools

import torch
from omegaconf import OmegaConf


def get_conf(name: str):
    """Returns yaml config file in DictConfig format

    Args:
        name: (str) name of the yaml file without .yaml extension
    """
    name = name if name.split(".")[-1] == "yaml" else name + ".yaml"
    cfg = OmegaConf.load(name)
    return cfg


def one_hot(x, num_digits=4, start=1):
    N = x.size()[0]
    x = x.long()[:, None] - start
    x = torch.clamp(x, 0, num_digits - 1)
    y = torch.FloatTensor(N, num_digits)
    y.zero_()
    y.scatter_(1, x, 1)
    return y


def timeit(fn):
    """Calculate time taken by fn().

    A function decorator to calculate the time a function needed for completion on GPU.
    returns: the function result and the time taken
    """
    # first, check if cuda is available
    cuda = True if torch.cuda.is_available() else False
    if cuda:

        @functools.wraps(fn)
        def wrapper_fn(*args, **kwargs):
            torch.cuda.synchronize()
            t1 = time()
            result = fn(*args, **kwargs)
            torch.cuda.synchronize()
            t2 = time()
            take = t2 - t1
            return result, take

    else:

        @functools.wraps(fn)
        def wrapper_fn(*args, **kwargs):
            t1 = time()
            result = fn(*args, **kwargs)
            t2 = time()
            take = t2 - t1
            return result, take

    return wrapper_fn


class_labels = {
    0: "Unlabeled",
    1: "Building",
    2: "Fence",
    3: "Other",
    4: "Pedestrian",
    5: "Pole",
    6: "RoadLine",
    7: "Road",
    8: "SideWalk",
    9: "Vegetation",
    10: "Vehicles",
    11: "Wall",
    12: "TrafficSign",
    13: "Sky",
    14: "Ground",
    15: "Bridge",
    16: "RailTrack",
    17: "GuardRail",
    18: "TrafficLight",
    19: "Static",
    20: "Dynamic",
    21: "water",
    22: "terrain",
}

"""Creating a dataset for carla image segmentation"""
import sys

try:
    sys.path.append("../")
except:
    raise RuntimeError("Can't append root directory of the project the path")
from pathlib import Path
from typing import Union, List, Tuple
import random
import json
from PIL import Image

import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from model.augmenter import get_augmenter, Crop, MaskPILToTensor


def imread(address: str):
    img = cv2.imread(address, cv2.IMREAD_UNCHANGED)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def read_json(address: str):
    with open(address, "r") as f:
        data = json.load(f)

    return data


class Augment(object):
    def __init__(self, seq):
        self.seq = seq

    def __call__(self, im):
        return Image.fromarray(self.seq.augment_images([np.array(im)])[0])


class CarlaSeg(Dataset):
    """DataLoader for Image Segmentation task"""

    def __init__(
        self,
        root: str = "../data/train",
        # val_percent: float = 0.2,
        aug_type: str = "segmentation",
        mode: str = "train",
        seed: int = 0,
        crop: Union[List, Tuple] = (125, 90),
        resize: Union[List, Tuple] = (224, 224),
    ):
        # assert (
        #     1 > val_percent > 0
        # ), "validation percentage should be a value between (0 , 1)."
        random.seed(seed)
        torch.random.manual_seed(seed)
        # read directories
        root = Path(root).resolve()
        dirs = [x for x in root.iterdir() if x.is_dir()]

        self.img_address = sorted(
            [
                str(x)
                for d in dirs
                for x in Path(d / "rgb").iterdir()
                if x.suffix == ".png"
            ]
        )
        self.mask_address = sorted(
            [
                str(x)
                for d in dirs
                for x in Path(d / "mask").iterdir()
                if x.suffix == ".png"
            ]
        )

        n_samples = len(self.img_address)
        # val_samples = int(val_percent * n_samples)
        shuffled_data = torch.randperm(n_samples)

        if mode.lower() == "train":
            self.indices = shuffled_data
            self.transform = transforms.Compose(
                [
                    Crop(crop),
                    transforms.Resize(resize),
                    Augment(get_augmenter(iteration=None, aug_type=aug_type, bsz=None)),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean hardcoded
                    #                      std=[0.229, 0.224, 0.225])  # ImageNet std hardcoded
                ]
            )
        elif mode.lower() == "val":
            self.indices = shuffled_data
            self.transform = transforms.Compose(
                [
                    Crop(crop),
                    transforms.Resize(resize),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean hardcoded
                    #                      std=[0.229, 0.224, 0.225])  # ImageNet std hardcoded
                ]
            )
        else:
            raise ValueError(
                "Unknown parameter for mode, it should be 'train' or 'val'"
            )

        # transform mask
        self.transform_mask = transforms.Compose(
            [Crop(crop), transforms.Resize(resize), MaskPILToTensor()]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        img = imread(self.img_address[self.indices[index]])
        mask = self.transform_mask(imread(self.mask_address[self.indices[index]]))
        img = self.transform(img)
        # img = img.refine_names(..., 'channels', 'height', 'width')
        # mask = mask.refine_names(..., 'height', 'width')

        return img, mask


class CarlaSegPred(Dataset):
    """DataLoader for video segmentation prediction task"""

    def __init__(
        self,
        root: str = "../data/train",
        past_frames: int = 4,
        future_frames: int = 6,
        # val_percent: float = 0.2,
        aug_type: str = "segmentation",
        mode: str = "train",
        seed: int = 0,
        load_measurements: bool = False,
        batch_size: int = 32,
        boost: int = 1,
        crop: Union[List, Tuple] = (125, 90),
        resize: Union[List, Tuple] = (224, 224),
        speed_factor: int = 10,
        n_commands: int = 4,
    ):
        # assert (
        #     1 > val_percent > 0
        # ), "validation percentage should be a value between (0 , 1)."
        random.seed(seed)
        torch.random.manual_seed(seed)
        self.load_measurements = load_measurements
        self.aug_type = aug_type
        self.batch_size = batch_size
        self.boost = boost
        self.crop = crop
        self.mode = mode
        self.resize = resize
        self.speed_factor = speed_factor
        self.n_commands = n_commands
        # read directories
        root = Path(root).resolve()
        dirs = [x for x in root.iterdir() if x.is_dir()]

        self.img_address = []
        if load_measurements:
            self.measurements = []
        else:
            self.mask_address = []
        seq_len = past_frames + future_frames

        for d in dirs:
            rgb_files = sorted(
                [str(x) for x in Path(d / "rgb").iterdir() if x.suffix == ".png"]
            )
            if load_measurements:
                measurements_files = sorted(
                    [
                        str(x)
                        for x in Path(d / "measurements").iterdir()
                        if x.suffix == ".json"
                    ]
                )
            else:
                mask_files = sorted(
                    [str(x) for x in Path(d / "mask").iterdir() if x.suffix == ".png"]
                )
            for i in range(len(rgb_files) - seq_len):
                self.img_address.append(rgb_files[i : i + past_frames])
                if load_measurements:
                    self.measurements.append(measurements_files[i + past_frames])
                else:
                    self.mask_address.append(mask_files[i + past_frames : i + seq_len])

        n_samples = len(self.img_address)
        # val_samples = int(val_percent * n_samples)
        shuffled_data = torch.randperm(n_samples)
        self.batch_read_number = 0

        if mode.lower() == "train":
            self.indices = shuffled_data

        elif mode.lower() == "val":
            self.indices = shuffled_data

        else:
            raise ValueError(
                "Unknown parameter for mode, it should be 'train' or 'val'"
            )

    def __len__(self):
        return len(self.indices)

    def _preprocess_measurements(self, measurements):
        steer = measurements["steer"]
        brake = measurements["brake"]
        throttle = measurements["throttle"]
        speed = torch.tensor(
            measurements["speed"] / self.speed_factor, dtype=torch.float
        )
        target_speed = torch.tensor(
            measurements["target_speed"] / self.speed_factor, dtype=torch.float
        )
        command = torch.zeros(self.n_commands).scatter_(
            dim=0,
            index=torch.tensor(measurements["command"] - 1, dtype=torch.long),
            value=1
        )
        # convert throttle and brake into a single action space in range [-1, 1]
        if brake > 0.05:
            pedal = -brake
        else:
            pedal = throttle
        control = torch.tensor([steer, pedal], dtype=torch.float)

        return {
            "control": control,
            "speed": speed,
            "target_speed": target_speed,
            "command": command,
        }

    def __getitem__(self, index):
        """If load_measurements is False, returns a tuple containing a tensor of shape (T, C, H, W)
        for past frames where T is number of past frames, and a tensor of (T`, H, W) for future mask data
        where T` is the number of future frames. Else returns a tuple containing a tensor of shape (T, C, H, W)
        for past frames and a dictionary containing actual measurements.
        """
        imgs = []
        iteration = self.boost * self.batch_read_number
        self.batch_read_number += 1

        if self.mode.lower() == "train":
            transform = transforms.Compose(
                [
                    Crop(self.crop),
                    transforms.Resize(self.resize),
                    Augment(
                        get_augmenter(
                            iteration=iteration,
                            aug_type=self.aug_type,
                            bsz=self.batch_size,
                        )
                    ),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean hardcoded
                    #                      std=[0.229, 0.224, 0.225])  # ImageNet std hardcoded
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    Crop(self.crop),
                    transforms.Resize(self.resize),
                    transforms.ToTensor(),
                    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    #                      std=[0.229, 0.224, 0.225])
                ]
            )

        if not self.load_measurements:
            masks = []
            transform_mask = transforms.Compose(
                [Crop(self.crop), transforms.Resize(self.resize), MaskPILToTensor()]
            )

        for img_address in self.img_address[self.indices[index]]:
            # read all images and append them to a list
            img = imread(img_address)
            img = transform(img)
            # img = img.refine_names(..., 'channels', 'height', 'width')
            imgs.append(img)
        if self.load_measurements:
            measurements = self._preprocess_measurements(
                read_json(self.measurements[self.indices[index]])
            )
            # ims: (T_Past, C, H, W), measurements: dict
            return torch.stack(imgs, dim=0), measurements
        else:
            for mask_address in self.mask_address[self.indices[index]]:
                # read all mask data and append them to a list
                mask = transform_mask(imread(mask_address))
                # mask = mask.refine_names(..., 'height', 'width')
                masks.append(mask)
            # ims: (T_Past, C, H, W), masks: (T_Future, H, W)
            return torch.stack(imgs, dim=0), torch.stack(masks, dim=0)


def decode_mask(mask, nc: int = 23):
    """Decode segmentation map to an RGB image

    class labels are based on:
        https://carla.readthedocs.io/en/latest/ref_sensors/#semantic-segmentation-camera

    Args:
        mask: (numpy.ndarray) the segmentation image
        nc: (int) number of classes that segmentation have
    """
    if len(mask.shape) == 3:
        mask = np.argmax(mask, axis=0)

    label_colors = np.array(
        [
            (0, 0, 0),  # 0=Unlabeled
            # 1=Building, 2=Fence, 3=Other   , 4=Pedestrian, 5=Pole
            (70, 70, 70),
            (100, 40, 40),
            (55, 90, 80),
            (220, 20, 60),
            (153, 153, 153),
            # 6=RoadLine, 7=Road, 8=SideWalk, 9=Vegetation, 10=Vehicles
            (157, 234, 50),
            (128, 64, 128),
            (244, 35, 232),
            (107, 142, 35),
            (0, 0, 142),
            # 11=Wall, 12=TrafficSign, 13=Sky, 14=Ground, 15=Bridge
            (102, 102, 156),
            (220, 220, 0),
            (70, 130, 180),
            (81, 0, 81),
            (150, 100, 100),
            # 16=RailTrack, 17=GuardRail, 18=TrafficLight, 19=Static, 20=Dynamic
            (230, 150, 140),
            (180, 165, 180),
            (250, 170, 30),
            (110, 190, 160),
            (170, 120, 50),
            # 21=water, 22=terrain
            (45, 60, 150),
            (145, 170, 100),
        ]
    )

    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)

    for l in range(nc):
        idx = mask == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


if __name__ == "__main__":
    # data = CarlaSeg(mode='train')
    # print(len(data))
    # d = data[0]
    # mask = d[1]
    # img = d[0]
    # seg_decoded = decode_mask(mask)
    # print(img.shape, img.dtype, mask.shape, mask.dtype)
    # print(f"\n{mask = }\n\n\n\n {seg_decoded = }")
    # # plotting
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 15))
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    #
    # ax1.set_title("Image")
    # ax2.set_title("Mask")
    # # print(f"{img_a.shape = } | {seg_a.shape = }")
    # ax1.imshow(img.permute(1, 2, 0))
    # ax2.imshow(seg_decoded)
    # plt.show()
    # plot CarlaSegPred
    # data = CarlaSegPred(mode='train', past_frames=2, future_frames=6)
    # print(len(data))
    # d = data[41]
    # seg_decoded = decode_mask(d[1][0])
    # print(len(d[0]), d[0].shape, len(d[1]), d[1].shape)
    # print(f"\n{d[1][0] = }\n\n\n\n {seg_decoded = }")
    # # plotting
    # fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(20, 20))
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    # ax3.set_axis_off()
    # ax4.set_axis_off()
    # ax5.set_axis_off()
    # ax6.set_axis_off()
    # ax7.set_axis_off()
    # ax8.set_axis_off()
    #
    # ax1.imshow(d[0][0].permute(1, 2, 0))
    # ax2.imshow(d[0][1].permute(1, 2, 0))
    # ax3.imshow(decode_mask(d[1][0]))
    # ax4.imshow(decode_mask(d[1][1]))
    # ax5.imshow(decode_mask(d[1][2]))
    # ax6.imshow(decode_mask(d[1][3]))
    # ax7.imshow(decode_mask(d[1][4]))
    # ax8.imshow(decode_mask(d[1][5]))
    # plt.show()

    # plot CarlaSegPred measurements
    data = CarlaSegPred(
        load_measurements=True, mode="train", past_frames=4, future_frames=6
    )
    print(len(data))
    d = data[10]
    print(len(d[0]), d[0][0].dtype, type(d[1]))
    print(f"{d[0][0].shape = }, {d[0][-1].min() = }, {d[0][-1].max() = }")
    measurements = d[1]
    print(
        f"{measurements['control'] = }\t{measurements['speed'] = }\t{measurements['target_speed'] = }\t"
        f"{measurements['command'] = }"
    )
    # plotting
    # fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 20))
    # ax1.set_axis_off()
    # ax2.set_axis_off()
    # ax3.set_axis_off()
    # ax4.set_axis_off()
    #
    # ax1.imshow(d[0][0].permute(1, 2, 0))
    # ax2.imshow(d[0][1].permute(1, 2, 0))
    # ax3.imshow(d[0][2].permute(1, 2, 0))
    # ax4.imshow(d[0][3].permute(1, 2, 0))
    # plt.show()

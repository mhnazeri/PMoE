"""Implementation of Predictive U-Net for semantic segmentation prediction"""

from collections import deque

import torch
import torch.nn as nn

from .blocks.unet import UNet
from .blocks.basics import EfficientConvBlock


class PredictiveUnet(nn.Module):
    """Autoregressive UNet to predict future segmentation maps"""

    def __init__(
        self,
        past_frames: int = 4,
        future_frames: int = 4,
        in_features: int = 3,
        num_classes: int = 23,
        gamma: int = 2,
        b: int = 1,
        inter_repr: bool = False,
        unet_inter_repr: bool = False,
        model_name: str = "unet-swa",
        model_path: str = "unet.pth",
    ):
        super().__init__()
        self.n_past_frames = past_frames
        self.n_future_frames = future_frames
        self.inter_repr = inter_repr
        self.unet_inter_repr = unet_inter_repr
        self.unet = UNet(
            in_features=in_features,
            out_features=num_classes,
            gamma=gamma,
            b=b,
            inter_repr=unet_inter_repr,
        )
        checkpoint = torch.load(model_path)
        # unet_weights = self.unet.state_dict()
        pre_trained_unet_weights = checkpoint[model_name]
        # pre_trained_unet_weights = {
        #     k: v
        #     for k, v in pre_trained_unet_weights.items()
        #     if k in self.unet.state_dict().keys()
        # }
        # unet_weights.update(pre_trained_unet_weights)
        # self.unet.load_state_dict(unet_weights)
        self.unet.load_state_dict(pre_trained_unet_weights, strict=False)
        # freeze unet weights
        for p in self.unet.parameters():
            p.requires_grad = False

        self.unet.eval()

        self.entry_block = EfficientConvBlock(
            in_ch=past_frames * num_classes, out_ch=in_features, gamma=gamma, b=b
        )
        self.pred_unet = UNet(
            in_features=in_features,
            out_features=num_classes,
            gamma=gamma,
            b=b,
            inter_repr=inter_repr,
        )

        # warmup predictive unet
        # pre_trained_unet_weights = {
        #     k: v for k, v in pre_trained_unet_weights.items()
        #     if k in self.unet.state_dict().keys() and k != 'dwn_1.0.weight'
        # }
        # self.pred_unet.load_state_dict(pre_trained_unet_weights, strict=False)

    def forward(self, img_list: torch.Tensor) -> torch.Tensor:
        """Forward computation of PU-Net network.

        Args:
            img_list: (torch.Tensor) A tensor of past frames with dim=(B, T, C, H, W)

        Returns:
            torch.Tensor
        """
        assert (
            img_list.shape[-4] == self.n_past_frames
        ), "Number of images should match number of past frames"

        mask_list = deque(
            [self.unet(img_list[:, i, ...]) for i in range(self.n_past_frames)],
            maxlen=self.n_past_frames,
        )

        if self.n_future_frames == 0:
            # if number of future frames is zero, return current frame segmentation mask
            if self.unet_inter_repr:
                return mask_list[-1][0]
            else:
                return mask_list[-1]

        if self.inter_repr:
            # not suitable for training
            inter_masks = None
            for _ in range(self.n_future_frames):
                masks = torch.cat(list(mask_list), dim=-3)  # concat along channel dim
                masks = self.entry_block(masks)
                inter_masks, masks = self.pred_unet(masks)
                mask_list.append(masks)

            return inter_masks
        else:
            outs = []
            for _ in range(self.n_future_frames):
                masks = torch.cat(list(mask_list), dim=-3)  # concat along channel dim
                masks = self.entry_block(masks)
                masks = self.pred_unet(masks)  # predict next frame segmentation mask
                mask_list.append(masks)
                outs.append(masks)

            # shape (B, T, C, H, W)
            return torch.stack(outs, dim=1)


if __name__ == "__main__":
    pass

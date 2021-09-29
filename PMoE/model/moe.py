"""All the models that are used in the experiments:
Mixture of Experts (MoE)(Alternative) with/out shared weights
Predictive U-Net (PU-Net)
Predictive Mixture of Experts (PMoE)
"""
from pathlib import Path
import sys

try:
    sys.path.append(str(Path("../").resolve()))
except:
    raise RuntimeError("Can't append root directory of the project the path")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from .punet import PredictiveUnet
from .blocks.basics import make_mlp
from .blocks.backbone import get_backbone, get_unet
from utils.nn import freeze


def get_model(cfg):
    model_type = cfg.type
    assert model_type is not None, "Network type can not be None"
    if model_type in ["moe", "moe_alt"]:
        return MixtureOfExperts(cfg)
    elif model_type in ["moe_shared"]:
        return MixtureOfExpertsShared(cfg)
    elif model_type in ["punet", "punet_inter"]:
        return PUNetExpert(cfg)
    elif model_type in ["pmoe", "pmoe+pretrained"]:
        assert (
            cfg.pmoe.moe_dir != ""
        ), "MoE pretrained weights directory should be specified"
        if model_type == "pmoe+pretrained":
            assert (
                cfg.pmoe.punet_dir != ""
            ), "PU-Net pretrained weights directory should be specified"
        return PMoE(cfg)
    else:
        raise ValueError(
            f"{model_type} is UNKNOWN, model type should be one of 'moe', 'punet', "
            f"'punet_inter', 'pmoe', 'pmoe+pretrained', 'moe_alt'"
        )


class BaseExpert(nn.Module):
    """Expert model"""

    def __init__(self, params):
        super().__init__()
        self.speed_encoder = make_mlp(**params.speed_encoder)
        self.command_encoder = make_mlp(**params.command_encoder)
        backbone_cfg = (
            {**params.backbone.rgb, "n_frames": params.backbone.n_frames}
            if params.backbone.type == "rgb"
            else {**params.backbone.segmentation, "n_frames": params.backbone.n_frames}
        )
        self.backbone = (
            get_backbone(**backbone_cfg)
            if params.backbone.type == "rgb"
            else get_unet(**backbone_cfg)
        )
        self.speed_pred = make_mlp(**params.speed_prediction)
        self.action_features = make_mlp(**params.action_head)
        # separate co-efficients to easily unfroze them later
        action_layer_out_features = params.action_head.dims[-1]
        self.alpha = nn.Linear(action_layer_out_features, 1)
        self.action_pred = nn.Linear(action_layer_out_features, 4)

    def forward(
        self, images: torch.Tensor, speed: torch.Tensor, command: torch.Tensor):
        """Forward pass of expert model.

        Args:
            images: (torch.Tensor) with dimensions: B, T, C, H, W
            speed: (torch.Tensor) with dimensions: B, 1
            command: (torch.Tensor) with dimensions: B, 4

        Returns:
            alphas: (torch.Tensor) predicted coefficients
            actions: (torch.Tensor) predicted actions [steer, pedal]
            pred_speed: (torch.Tensor) predicted speed
        """
        speed = self.speed_encoder(speed)
        command = self.command_encoder(command)
        images = images.view(
            images.shape[0], -1, images.shape[-2], images.shape[-1]
        )  # cat along time dimension
        img = self.backbone(images)
        # concat features along the last dim to produce tensor (B, 3 * 512)
        features = torch.cat([img, speed, command], dim=-1)
        pred_speed = self.speed_pred(features)
        action_features = self.action_features(features)
        mean, std = self.action_pred(action_features).split(2, dim=-1)
        std = F.elu(std) + 1
        alpha = torch.relu(self.alpha(action_features))
        return alpha, mean, std, pred_speed


class BaseExpertAlt(BaseExpert):
    """Alternative expert model which alpha uses input of the network"""

    def __init__(self, params):
        super().__init__(params)
        self.alpha = nn.Sequential(
            *[nn.Linear(1536, 512), nn.ReLU(inplace=True), nn.Linear(512, 1)]
        )

    def forward(
        self, images: torch.Tensor, speed: torch.Tensor, command: torch.Tensor):
        speed = self.speed_encoder(speed)
        command = self.command_encoder(command)
        images = images.view(
            images.shape[0], -1, images.shape[-2], images.shape[-1]
        )  # cat along time dimension
        img = self.backbone(images)
        # concat features along the last dim to produce tensor (B, 3 * 512)
        features = torch.cat([img, speed, command], dim=-1)
        pred_speed = self.speed_pred(features)
        action_features = self.action_features(features)
        mean, std = self.action_pred(action_features).split(2, dim=-1)
        std = F.elu(std) + 1
        alpha = self.alpha(features)
        return alpha, mean, std, pred_speed


class MixtureOfExperts(nn.Module):
    def __init__(self, params):
        super().__init__()
        # number of experts
        self.k = params.n_experts
        # experts does not share weights
        base = BaseExpert if params.type == "moe" else BaseExpertAlt
        self.moe = nn.ModuleList([base(params) for _ in range(self.k)])

    def forward(self, images: torch.Tensor, speed: torch.Tensor, command: torch.Tensor):
        out = [moe(images, speed, command) for moe in self.moe]
        alphas, mean, std, speeds = [], [], [], []
        for expert in out:
            alphas.append(expert[0])
            mean.append(expert[1])
            std.append(expert[2])
            speeds.append(expert[3])

        # alpha: (Batch, #Experts)
        alphas = torch.cat(alphas, dim=1)
        alphas = F.softmax(alphas, dim=1)
        mean = torch.stack(mean, dim=1)
        std = torch.stack(std, dim=1)
        mixtures = D.Categorical(alphas)
        components = D.Independent(D.Normal(mean, std), 1)
        actions = D.MixtureSameFamily(mixtures, components)
        # speed: (Batch, Expert_idx, Pred_Speed)
        return actions, torch.stack(speeds, dim=1)

    def sample(
        self, images: torch.Tensor, speed: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        out = [moe(images, speed, command) for moe in self.moe]
        alphas, mean, std = [], [], []
        for expert in out:
            alphas.append(expert[0])
            mean.append(expert[1])
            std.append(expert[2])

        alphas = torch.cat(alphas, dim=1)
        alphas = F.softmax(alphas, dim=1)
        mean = torch.stack(mean, dim=1)
        std = torch.stack(std, dim=1)
        mixtures = D.Categorical(alphas)
        components = D.Independent(D.Normal(mean, std), 1)
        actions = D.MixtureSameFamily(mixtures, components)
        return actions.sample()


class MixtureOfExpertsShared(nn.Module):
    """Mixture of Expert (MoE) model with share backbone"""

    def __init__(self, params):
        super().__init__()
        self.speed_encoder = make_mlp(**params.speed_encoder)
        self.command_encoder = make_mlp(**params.command_encoder)
        backbone_cfg = (
            {**params.backbone.rgb, "n_frames": params.backbone.n_frames}
            if params.backbone.type == "rgb"
            else {**params.backbone.segmentation, "n_frames": params.backbone.n_frames}
        )
        self.backbone = (
            get_backbone(**backbone_cfg)
            if params.backbone.type == "rgb"
            else get_unet(**backbone_cfg)
        )
        self.speed_pred = make_mlp(**params.speed_prediction)
        self.action_features = make_mlp(**params.action_head)
        # separate co-efficients to easily unfroze them later
        action_layer_out_features = params.action_head.dims[-1]
        self.n_experts = params.n_experts
        self.alpha = nn.Linear(action_layer_out_features, params.n_experts)
        self.action_pred = nn.Linear(action_layer_out_features, 4 * params.n_experts)

    def forward(self, images: torch.Tensor, speed: torch.Tensor, command: torch.Tensor):
        """Forward pass of expert model.

        Args:
            images: (torch.Tensor) with dimensions: B, T, C, H, W
            speed: (torch.Tensor) with dimensions: B, 1
            command: (torch.Tensor) with dimensions: B, 4

        Returns:
            actions: (torch.Tensor) predicted actions [steer, pedal]
            pred_speed: (torch.Tensor) predicted speed
        """
        speed = self.speed_encoder(speed)
        command = self.command_encoder(command)
        images = images.view(
            images.shape[0], -1, images.shape[-2], images.shape[-1]
        )  # cat along time dimension
        img = self.backbone(images)
        # concat features along the last dim to produce tensor (B, 3 * 512)
        features = torch.cat([img, speed, command], dim=-1)
        pred_speed = self.speed_pred(features)
        action_features = self.action_features(features)
        # calculate mean and std
        mean, std = (
            self.action_pred(action_features)
            .view(speed.shape[0], self.n_experts, -1)
            .split(2, dim=-1)
        )
        std = F.elu(std) + 1
        alpha = F.softmax(self.alpha(action_features), dim=1)
        # mixture coefficients
        mixture = D.Categorical(alpha)
        components = D.Independent(D.Normal(mean, std), 1)
        actions = D.MixtureSameFamily(mixture, components)
        return actions, pred_speed

    def sample(
        self, images: torch.Tensor, speed: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        speed = self.speed_encoder(speed)
        command = self.command_encoder(command)
        images = images.view(
            images.shape[0], -1, images.shape[-2], images.shape[-1]
        )  # cat along time dimension
        img = self.backbone(images)
        # concat features along the last dim to produce tensor (B, 3 * 512)
        features = torch.cat([img, speed, command], dim=-1)
        action_features = self.action_features(features)
        # calculate mean and std
        mean, std = (
            self.action_pred(action_features)
            .view(speed.shape[0], self.n_experts, -1)
            .split(2, dim=-1)
        )
        std = F.elu(std) + 1
        alpha = F.softmax(self.alpha(action_features), dim=1)
        # mixture coefficients
        mixture = D.Categorical(alpha)
        components = D.Independent(D.Normal(mean, std), 1)
        actions = D.MixtureSameFamily(mixture, components)
        return actions.sample()


class PUNetExpert(nn.Module):
    """PU-Net as action prediction"""

    def __init__(self, params):
        super().__init__()
        self.return_inter = True if params.type == "punet_inter" else False
        params.punet.inter_repr = self.return_inter
        self.speed_encoder = make_mlp(**params.speed_encoder)
        self.command_encoder = make_mlp(**params.command_encoder)
        self.punet = PredictiveUnet(**params.punet)
        punet_weights = torch.load(params.punet_path, map_location=params.device)
        self.punet.load_state_dict(punet_weights["model"])
        self.punet = freeze(self.punet)
        # use backbone if PU-Net does not return a vector as the result
        self.backbone = (
            None
            if self.return_inter
            else get_backbone(
                **{
                    **params.backbone.rgb,
                    "n_frames": params.punet.future_frames,
                    "n_channels": params.punet.num_classes,
                }
            )
        )
        self.speed_pred = make_mlp(**params.speed_prediction)
        # return actions, use tanh to squash output in range [-1, 1]
        # params.action_head.act = 'tanh'
        self.action_pred = nn.Sequential(
            *[
                make_mlp(**params.action_head),
                nn.Linear(params.action_head.dims[-1], 2),
            ],
        )

    def forward(self, images: torch.Tensor, speed: torch.Tensor, command: torch.Tensor):
        speed = self.speed_encoder(speed)
        command = self.command_encoder(command)
        if not self.return_inter:
            images = self.punet(images)
            images = images.view(
                images.shape[0], -1, images.shape[-2], images.shape[-1]
            )  # cat time x channel together
            img = self.backbone(images)
        else:
            img = self.punet(images)
        # concat features along the last dim to produce tensor (B, 3 * 512)
        features = torch.cat([img, speed, command], dim=-1)

        return torch.tanh(self.action_pred(features)), self.speed_pred(features)

    def sample(
        self, images: torch.Tensor, speed: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        action, _ = self.forward(images, speed, command)
        return action


class PMoE(nn.Module):
    """Predictive mixture of experts (PMoE) implementation"""

    def __init__(self, params):
        super().__init__()
        assert params.pmoe.moe_dir is not None, "MoE weights should be provided"
        # initialize MoE model
        moe_model_dir = params.pmoe.moe_dir
        self.moe = MixtureOfExperts(params)
        # you may want to use SWA model, therefore strict should be False
        self.moe.load_state_dict(torch.load(moe_model_dir), strict=False)
        self.moe = freeze(self.moe, params.exclude_freeze, params.verbose)
        # initialize PU-Net model
        punet_model_dir = params.pmoe.punet_dir
        self.punet = PUNetExpert(params)
        if punet_model_dir:
            self.punet.load_state_dict(torch.load(punet_model_dir), strict=False)
            self.punet = freeze(self.punet, params.exclude_freeze, params.verbose)

        self.lat_weights = nn.Linear(2, 1)
        self.long_weights = nn.Linear(2, 1)

    def forward(
        self, images: torch.Tensor, speed: torch.Tensor, command: torch.Tensor):
        punet_actions, _ = self.punet(images.clone(), speed.clone(), command.clone())
        dists, _ = self.moe(images, speed, command)
        moe_actions = dists.sample()
        lat_actions = self.lat_weights(torch.cat([moe_actions[:, 0: 1], punet_actions[:, 0: 1]], dim=-1))
        long_actions = self.long_weights(torch.cat([moe_actions[:, 1:], punet_actions[:, 1:]], dim=-1))
        # -1 is just a dummy variable as speed prediction for the sake of interface consistency
        return torch.tanh(torch.cat([lat_actions, long_actions], dim=-1)), -1

    def sample(
        self, images: torch.Tensor, speed: torch.Tensor, command: torch.Tensor
    ) -> torch.Tensor:
        """For the interface to be consistent with other networks"""
        actions, _ = self.forward(images, speed, command)
        return actions


if __name__ == "__main__":
    pass

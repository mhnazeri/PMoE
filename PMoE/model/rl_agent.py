from pathlib import Path
import sys

try:
    sys.path.append(str(Path("../").resolve()))
except:
    raise RuntimeError("Can't append root directory of the project the path")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .blocks.backbone import get_backbone, get_unet
from .blocks.basics import make_mlp
from .moe import get_model
from .replay_memory import ReplayMemory, Transition
from utils.nn import freeze
from utils.noise import OrnsteinUhlenbeckActionNoise as OU


class Actor(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.actor = get_model(params.actor)
        model_dir = params.actor.model_dir
        model_state = torch.load(model_dir)
        self.actor.load_state_dict(model_state["actor"])
        self.actor = freeze(
            self.actor, params.actor.exclude_freeze, params.actor.verbose
        )
        self.device = torch.device(params.actor.device)
        self.to(device=self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=params.actor.lr)

    def forward(self, images, speed, command):
        action, speed = self.actor(images, speed, command)
        return action


class Critic(nn.Module):
    def __init__(self, params):
        super().__init__()
        backbone_cfg = (
            {**params.critic.backbone.rgb, "n_frames": params.critic.backbone.n_frames}
            if params.backbone.type == "rgb"
            else {
                **params.critic.backbone.segmentation,
                "n_frames": params.critic.backbone.n_frames,
            }
        )
        self.backbone = (
            get_backbone(**backbone_cfg)
            if params.critic.backbone.type == "rgb"
            else get_unet(**backbone_cfg)
        )
        self.speed_encoder = make_mlp(**params.critic.speed_encoder)
        self.command_encoder = make_mlp(**params.critic.command_encoder)
        self.action_encoder = make_mlp(**params.critic.action_encoder)
        self.vale_pred = make_mlp(**params.critic.value_prediction)

        self.device = torch.device(params.critic.device)
        self.to(device=self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=params.critic.lr)

    def forward(self, images, speed, command, action):
        state = self.backbone(images)
        speed = self.speed_encoder(speed)
        action = self.action_encoder(action)
        command = self.command_encoder(command)

        return self.vale_pred(torch.cat([state, speed, command, action], dim=-1))


class Agent:
    def __init__(self, params):
        self.gamma = params.gamma
        self.tau = params.tau
        self.memory = ReplayMemory(params.buffer_size)
        self.batch_size = params.batch_size

        self.actor = Actor(params)
        self.target_actor = Actor(params)

        self.critic = Critic(params)

        self.target_critic = Critic(params)

        self.noise = OU(
            mu=np.array(params.OU.mu),
            theta=np.array(params.OU.theta),
            sigma=np.array(params.OU.sigma),
        )

        self.update_network_parameters(tau=1)

    def sample(
        self, images: torch.Tensor, speed: torch.Tensor, command: torch.Tensor
    ):
        self.actor.eval()
        # observation = torch.tensor(observation, dtype=torch.float).to(self.device)
        mu = self.actor(images, speed, command)
        mu_prime = mu + torch.tensor(self.noise(), dtype=torch.float).to(
            device=self.actor.device
        )

        self.actor.train()
        return mu_prime.cpu().detach().numpy()

    def remember(
        self,
        images,
        speed,
        command,
        action,
        reward,
        new_images,
        new_speed,
        new_command,
        done,
    ):
        self.memory.push(
            images,
            speed,
            command,
            action,
            reward,
            new_images,
            new_speed,
            new_command,
            done,
        )

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        reward = torch.tensor(batch.reward, dtype=torch.float).to(self.critic.device)
        done = torch.tensor(batch.done).to(self.critic.device)
        new_images = torch.tensor(batch.new_images, dtype=torch.float).to(
            self.critic.device
        )
        new_speed = torch.tensor(batch.new_speed, dtype=torch.float).to(
            self.critic.device
        )
        new_command = torch.tensor(batch.new_command, dtype=torch.float).to(
            self.critic.device
        )
        action = torch.tensor(batch.action, dtype=torch.float).to(self.critic.device)
        images = torch.tensor(batch.images, dtype=torch.float).to(self.critic.device)
        speed = torch.tensor(batch.speed, dtype=torch.float).to(self.critic.device)
        command = torch.tensor(batch.command, dtype=torch.float).to(self.critic.device)

        self.target_actor.eval()
        self.target_critic.eval()
        self.critic.eval()

        target_actions = self.target_actor(new_images, new_speed, new_command)
        critic_value_ = self.target_critic(
            new_images, new_speed, new_command, target_actions
        )
        critic_value = self.critic(images, speed, command, action)

        target = []
        for j in range(self.batch_size):
            target.append(reward[j] + self.gamma * critic_value_[j] * done[j])

        target = torch.tensor(target).to(self.critic.device)
        target = target.vew(self.batch_size, 1)

        self.critic.train()
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.critic.eval()
        self.actor.optimizer.zero_grad()
        mu = self.actor(images, speed, command)
        self.actor.train()
        actor_loss = -self.critic(images, speed, command, mu)
        actor_loss = torch.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_dict = dict(target_critic_params)
        target_actor_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = (
                tau * critic_state_dict[name].clone()
                + (1 - tau) * target_critic_dict[name].clone()
            )

        self.target_critic.load_state_dict(critic_state_dict)

        for name in actor_state_dict:
            actor_state_dict[name] = (
                tau * actor_state_dict[name].clone()
                + (1 - tau) * target_actor_dict[name].clone()
            )

        self.target_actor.load_state_dict(actor_state_dict)

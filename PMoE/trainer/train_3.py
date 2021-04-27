"""DDPG train phase"""
from datetime import datetime
from collections import deque

import comet_ml
import numpy as np
import carla
import tqdm
import torch
from torchvision import transforms

import glob
import sys

try:
    sys.path.append(glob.glob("../utils/PythonAPI")[0])
    sys.path.append(glob.glob("../")[0])
except IndexError as e:
    pass

import utils.carla_utils as cu
from utils.vision import draw_on_image
from utils.utility import get_conf
from utils.io import save_checkpoint
from utils.benchmark import make_suite
from model.rl_agent import Agent
from model.augmentor import Crop


def postprocess(action: torch.Tensor):
    control = carla.VehicleControl()
    control.steer = torch.clip(action[0], -1.0, 1.0).cpu().numpy()
    if action[1] > 0.05:
        control.throttle = torch.clip(action[1], 0.0, 0.75).cpu().numpy()
        control.brake = 0
    else:
        control.throttle = 0
        control.brake = -torch.clip(action[1], 0.0, 1.0).cpu().numpy()

    control.manual_gear_shift = False

    return control


def train(config, planner="new"):
    # initialize the logger
    logger = comet_ml.Experiment(
        disabled=config.logger.disabled,
        project_name=config.project,
        auto_histogram_weight_logging=True,
        auto_histogram_gradient_logging=True,
        auto_histogram_activation_logging=True,
    )
    logger.add_tags(config.logger.tags.split())
    logger.log_parameters(config)

    weathers = list(cu.TRAIN_WEATHERS.keys())
    img_list = deque(
        [
            torch.zeros(3, *config.dataset.resize)
            for _ in range(config.model.actor.backbone.n_frames)
        ],
        maxlen=config.model.actor.backbone.n_frames,
    )
    # create a transform to crop the sky and resize images
    transform = transforms.Compose(
        [
            Crop(config.dataset.crop),
            transforms.Resize(config.dataset.resize),
            transforms.ToTensor(),
        ]
    )

    for episode in tqdm.tqdm(range(config.train_params.max_episodes), desc="Episode"):
        progress = tqdm.tqdm(
            range(config.train_params.episode_length * len(weathers)), desc="Frame"
        )
        score_history = []
        for weather in weathers:

            data_cntr = 0

            while data_cntr < config.train_params.episode_length:

                with make_suite(
                    config.env.town, port=config.env.port, planner=planner
                ) as env:
                    start, target = env.pose_tasks[
                        np.random.randint(len(env.pose_tasks))
                    ]
                    env_params = {
                        "weather": weather,
                        "start": start,
                        "target": target,
                        "n_pedestrians": config.env.n_pedestrians,
                        "n_vehicles": config.env.n_vehicles,
                    }

                    env.init(**env_params)
                    env.success_dist = 5.0

                    agent = Agent(config.model)
                    env.tick()
                    observations = env.get_observations()
                    rgb = transform(observations["rgb"].copy())
                    img_list.append(rgb)
                    speed = (
                        torch.tensor(
                            np.linalg.norm(observations["velocity"])
                            / config.dataset.speed_factor,
                            dtype=torch.float,
                        )
                        .unsqueeze_(0)
                        .to(config.model.device)
                    )
                    _cmd = int(observations["command"])
                    command = (
                        torch.zeros(config.train_params.n_commands)
                        .scatter_(
                            dim=0,
                            index=torch.tensor(int(observations["command"]) - 1),
                            value=1,
                        )
                        .unsqueeze_(0)
                        .to(config.model.device)
                    )
                    images = (
                        torch.tensor(torch.stack(list(img_list), dim=0))
                        .unsqueeze_(0)
                        .to(config.model.device)
                    )
                    score = 0
                    while not env.is_success() and not env.collided:
                        action = agent.choose_action(images, speed, command)
                        old_images = images
                        control = postprocess(action)
                        info = env.apply_control(control)
                        score += info["reward"]
                        # get new observations
                        env.tick()
                        observations = env.get_observations()
                        rgb = transform(observations["rgb"].copy())
                        img_list.append(rgb)
                        new_speed = (
                            torch.tensor(
                                np.linalg.norm(observations["velocity"])
                                / config.dataset.speed_factor,
                                dtype=torch.float,
                            )
                            .unsqueeze_(0)
                            .to(config.model.device)
                        )
                        _cmd = int(observations["command"])
                        # command = self.one_hot[int(observations['command']) - 1]
                        new_command = (
                            torch.zeros(config.train_params.n_commands)
                            .scatter_(
                                dim=0,
                                index=torch.tensor(int(observations["command"]) - 1),
                                value=1,
                            )
                            .unsqueeze_(0)
                            .to(config.model.device)
                        )
                        images = (
                            torch.tensor(torch.stack(list(img_list), dim=0))
                            .unsqueeze_(0)
                            .to(config.model.device)
                        )
                        agent.remember(
                            old_images.squeeze(),
                            speed.squeeze(),
                            command.squeeze(),
                            action,
                            info["reward"],
                            images.squeeze(),
                            new_speed.squeeze(),
                            new_command.squeeze(),
                            int(env.is_success() or env.collided),
                        )
                        agent.learn()
                        speed = new_speed
                        command = new_command

                        progress.update(1)

                        if data_cntr >= config.train_params.episode_length:
                            break

        score_history.append(score)
        print(
            f"{datetime.now():%Y-%m-%d %H:%M:%S} Episode {episode:03}, score is {score:03}"
        )
        logger.log_metrics(
            {
                "episode": episode,
                "score": score,
            }
        )
        measurements = {
            "control": action,
            "speed": speed,
            "command": command,
        }
        log_image = draw_on_image(images.squeeze()[-2, ...].cpu(), measurements, action)
        logger.log_image(log_image, name="sample_" + str(episode))

        if episode % config.train_params.save_every == 0:
            save(
                agent,
                episode,
                np.mean(score_history),
                score,
                config.directory.save,
                config.directory.model_name,
            )


def save(model, episode, score, current_score, dir, name=None):
    checkpoint = {
        "actor": model.actor.state_dict(),
        "target_actor": model.target_actor.state_dict(),
        "critic": model.critic.state_dict(),
        "target_critic": model.target_critic.state_dict(),
        "episode": episode,
        "score": score,
    }

    save_name = name + "-e" + str(episode)

    if current_score > score:
        save_checkpoint(checkpoint, True, dir, save_name)
    else:
        save_checkpoint(checkpoint, False, dir, save_name)


if __name__ == "__main__":
    cfg = get_conf("../conf/stage_3.yaml")
    train(cfg)

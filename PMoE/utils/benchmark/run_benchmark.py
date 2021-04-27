"""source: https://github.com/dotchen/LearningByCheating"""
from collections import deque

import tqdm
import glob
import sys

try:
    sys.path.append(glob.glob("../")[0])
    sys.path.append(glob.glob("../../")[0])
except IndexError as e:
    pass

import numpy as np
import carla
import pandas as pd
import torch
from torchvision import transforms

import carla_utils as cu
from model.augmenter import Crop


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


def run_single(env, weather, start, target, agent_maker, seed):
    # HACK: deterministic vehicle spawns.
    env.seed = seed
    env.init(start=start, target=target, weather=cu.PRESET_WEATHERS[weather])

    agent = agent_maker

    diagnostics = list()
    result = {
        "weather": weather,
        "start": start,
        "target": target,
        "success": None,
        "t": None,
        "total_lights_ran": None,
        "total_lights": None,
        "collided": None,
    }

    img_list = deque([torch.zeros(3, 224, 224) for _ in range(4)], maxlen=4)
    # create a transform to crop the sky and resize images
    transform = transforms.Compose(
        [
            Crop([125, 90]),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    while env.tick():

        observations = env.get_observations()
        rgb = transform(observations["rgb"].copy())
        img_list.append(rgb)
        speed = (
            torch.tensor(
                np.linalg.norm(observations["velocity"]) / 10, dtype=torch.float
            )
            .unsqueeze_(0)
            .to(agent.device)
        )
        _cmd = int(observations["command"])
        command = (
            torch.zeros(4)
            .scatter_(
                dim=0, index=torch.tensor(int(observations["command"]) - 1), value=1
            )
            .unsqueeze_(0)
            .to(agent.device)
        )
        images = (
            torch.tensor(torch.stack(list(img_list), dim=0))
            .unsqueeze_(0)
            .to(agent.device)
        )

        action = agent.choose_action(images, speed, command)
        control = postprocess(action)
        diagnostic = env.apply_control(control)

        diagnostic.pop("viz_img")
        diagnostics.append(diagnostic)

        if env.is_failure() or env.is_success():
            result["success"] = env.is_success()
            result["total_lights_ran"] = env.traffic_tracker.total_lights_ran
            result["total_lights"] = env.traffic_tracker.total_lights
            result["collided"] = env.collided
            result["t"] = env._tick
            break

    return result, diagnostics


def run_benchmark(agent_maker, env, benchmark_dir, seed, resume, max_run=5):
    """
    benchmark_dir must be an instance of pathlib.Path
    """
    summary_csv = benchmark_dir / "summary.csv"
    diagnostics_dir = benchmark_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    summary = list()
    total = len(list(env.all_tasks))

    if summary_csv.exists() and resume:
        summary = pd.read_csv(summary_csv)
    else:
        summary = pd.DataFrame()

    num_run = 0

    for weather, (start, target), run_name in tqdm.tqdm(env.all_tasks, total=total):
        if (
            resume
            and len(summary) > 0
            and (
                (summary["start"] == start)
                & (summary["target"] == target)
                & (summary["weather"] == weather)
            ).any()
        ):
            print(weather, start, target)
            continue

        diagnostics_csv = str(diagnostics_dir / ("%s.csv" % run_name))

        result, diagnostics = run_single(env, weather, start, target, agent_maker, seed)

        summary = summary.append(result, ignore_index=True)

        # Do this every timestep just in case.
        pd.DataFrame(summary).to_csv(summary_csv, index=False)
        pd.DataFrame(diagnostics).to_csv(diagnostics_csv, index=False)

        num_run += 1

        if num_run >= max_run:
            break

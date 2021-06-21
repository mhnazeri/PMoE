import os
import math
import sys
from pathlib import Path
from collections import deque

try:
    sys.path.append(str(Path("../").resolve()))
except:
    raise RuntimeError("Can't append root directory of the project the path")

import comet_ml
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
import carla

from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track
# from utils import visualize_obs
from utils.vision import draw_on_image
from utils.utility import get_conf
from model.moe import get_model
from model.rl_agent import Actor
from model.augmenter import Crop
from waypointer import Waypointer

def get_entry_point():
    return 'ImageAgent'

class ImageAgent(AutonomousAgent):
    
    """
    Trained image agent
    """
    
    def setup(self, config):
        """
        Setup the agent parameters
        """

        self.track = Track.SENSORS
        self.num_frames = 0

        if isinstance(config, str):
            self.config = get_conf(config)

        self.device = torch.device(self.config.model.actor.device)
        if self.config.model.actor.type == "rl_agent":
            self.model = Actor(self.config.model.actor)
            state_dict = torch.load(self.config.model.actor.model_dir)
            self.model.load_state_dict(state_dict["actor"])
            self.model = self.model.to(device=self.config.model.actor.device)
        else:
            self.model = get_model(self.config.model.actor)
            state_dict = torch.load(self.config.model.actor.model_dir, map_location=self.device)
            self.model.load_state_dict(state_dict["model"])
            self.model.choose_action = self.model.sample
            self.model = self.model.to(device=self.device)

        self.model.eval()
        self.img_list = deque([torch.zeros(3, 224, 224, dtype=torch.float) for _ in range(self.config.model.actor.backbone.n_frames)],
                              maxlen=self.config.model.actor.backbone.n_frames)

        self.vizs = []

        self.waypointer = None
        self.logger = self.init_logger(self.config.logger)
        # create a transform to crop the sky and resize images
        self.transform = transforms.Compose(
            [
                Crop([125, 90]),
                # Image.fromarray,
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

    def destroy(self):
        if len(self.vizs) == 0:
            return

        self.flush_data()
        
        del self.waypointer
        del self.model
        del self.img_list
    
    def flush_data(self):
        if self.logger:
            out = cv2.VideoWriter('./log.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (224, 224))
            for i in range(len(self.vizs)):
                out.write(self.vizs[i])

            out.release()
            self.logger.log_asset("./log.avi")
            
        self.vizs.clear()

    def sensors(self):
        sensors = [
            {'type': 'sensor.collision', 'id': 'COLLISION'},
            {'type': 'sensor.speedometer', 'id': 'EGO'},
            {'type': 'sensor.other.gnss', 'x': 0., 'y': 0.0, 'z': self.config.sensors.camera_z, 'id': 'GPS'},
            {'type': 'sensor.camera.rgb', 'x': self.config.sensors.camera_x,
             'y': 0, 'z': self.config.sensors.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': self.config.sensors.img_width, 'height': self.config.sensors.img_height,
             'fov': self.config.sensors.fov, 'id': f'Narrow_RGB'},
        ]
        
        return sensors

    def postprocess(self, action: torch.Tensor):
        control = carla.VehicleControl()
        control.steer = torch.clip(action[0], -1.0, 1.0).item()
        if action[1] < -0.5:
            control.throttle = 0
            control.brake = torch.clip(-action[1], 0.0, 1.0).item()  # .cpu().numpy()
            control.steer = 0
        else:
            control.throttle = max(torch.clip(action[1], 0.0, 0.75).item(), 0.4)  # .cpu().numpy()
            control.brake = 0

        return control

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        _, _rgb = input_data.get(f'Narrow_RGB')

        # # Crop images
        rgb = _rgb[...,:3].copy()
        rgb = rgb[..., ::-1] # BGR -> RGB

        rgb = self.transform(rgb)
        self.img_list.append(rgb)
        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')


        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)

        _, _, cmd = self.waypointer.tick(gps)

        spd = ego.get('spd')
        
        cmd_value = cmd.value-1
        cmd_value = 3 if cmd_value < 0 else cmd_value

        speed = torch.tensor(
            [spd / 10.0, ], dtype=torch.float
        ).unsqueeze_(0).to(self.device)

        command = torch.zeros(self.config.model.actor.n_commands).scatter_(
            dim=0, index=torch.tensor(int(cmd_value)), value=1
        ).unsqueeze_(0).to(self.device)
        images = torch.stack(list(self.img_list), dim=0).unsqueeze_(0).to(self.device)
        action = self.model.choose_action(images, speed, command).squeeze().cpu()
        control = self.postprocess(action)

        measurements = {
            "control": [action[0], action[1]],
            "speed": speed,
            "command": command,
        }
        log_image = draw_on_image(
            rgb.cpu(), measurements, action.cpu(), False
        )
        self.vizs.append(log_image)

        if len(self.vizs) > 1000:
            self.flush_data()

        self.num_frames += 1

        return control

    def init_logger(self, cfg):
        # Check to see if there is a key in environment:
        EXPERIMENT_KEY = cfg.experiment_key

        # First, let's see if we continue or start fresh:
        CONTINUE_RUN = cfg.resume
        if EXPERIMENT_KEY and CONTINUE_RUN:
            # There is one, but the experiment might not exist yet:
            api = comet_ml.API()  # Assumes API key is set in config/env
            try:
                api_experiment = api.get_experiment_by_id(EXPERIMENT_KEY)
            except Exception:
                api_experiment = None
            if api_experiment is not None:
                CONTINUE_RUN = True
                # We can get the last details logged here, if logged:
                # step = int(api_experiment.get_parameters_summary("batch")["valueCurrent"])
                # epoch = int(api_experiment.get_parameters_summary("epochs")["valueCurrent"])

        if CONTINUE_RUN:
            # 1. Recreate the state of ML system before creating experiment
            # otherwise it could try to log params, graph, etc. again
            # ...
            # 2. Setup the existing experiment to carry on:
            logger = comet_ml.ExistingExperiment(
                previous_experiment=EXPERIMENT_KEY,
                log_env_details=True,  # to continue env logging
                log_env_gpu=True,  # to continue GPU logging
                log_env_cpu=True,  # to continue CPU logging
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=True,
                auto_histogram_activation_logging=True,
            )
            # Retrieved from above APIExperiment
            # self.logger.set_epoch(epoch)

        else:
            # 1. Create the experiment first
            #    This will use the COMET_EXPERIMENT_KEY if defined in env.
            #    Otherwise, you could manually set it here. If you don't
            #    set COMET_EXPERIMENT_KEY, the experiment will get a
            #    random key!
            logger = comet_ml.Experiment(
                disabled=cfg.disabled,
                project_name=cfg.project,
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=True,
                auto_histogram_activation_logging=True,
            )
            logger.set_name(cfg.experiment_name)
            logger.add_tags(cfg.tags.split())
            logger.log_parameters(cfg)

        return logger


if __name__ == "__main__":
    pass
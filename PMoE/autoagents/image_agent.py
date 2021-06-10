import os
import math
import sys
from pathlib import Path
from collections import deque

try:
    sys.path.append(str(Path("../").resolve()))
except:
    raise RuntimeError("Can't append root directory of the project the path")

# import yaml
# import lmdb
import comet_ml
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms
import carla
# import random
# import string
#
# from torch.distributions.categorical import Categorical

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
        # self.config = get_conf(config)

        if isinstance(config, str):
            self.config = get_conf(config)

        # with open(path_to_conf_file, 'r') as f:
        #     config = yaml.safe_load(f)

        # for key, value in config.items():
        #     setattr(self, key, value)

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
        # print(f"{self.config}")

        # if self.log_wandb:
        #     wandb.init(project='carla_evaluate')
            
        # self.steers = torch.tensor(np.linspace(-self.max_steers,self.max_steers,self.num_steers)).float().to(self.device)
        # self.throts = torch.tensor(np.linspace(0,self.max_throts,self.num_throts)).float().to(self.device)

        # self.prev_steer = 0
        # self.lane_change_counter = 0
        # self.stop_counter = 0

    def destroy(self):
        if len(self.vizs) == 0:
            return

        self.flush_data()
        # self.prev_steer = 0
        # self.lane_change_counter = 0
        # self.stop_counter = 0
        # self.lane_changed = None
        
        del self.waypointer
        del self.model
        del self.img_list
    
    def flush_data(self):

        # if self.log_wandb:
        #     wandb.log({
        #         'vid': wandb.Video(np.stack(self.vizs).transpose((0,3,1,2)), fps=20, format='mp4')
        #     })
        if self.logger:
            # log_image = torch.cat(self.viz, dim=-1)  # cat widths
            out = cv2.VideoWriter('./log.avi', cv2.VideoWriter_fourcc(*'DIVX'), 20, (224, 224))
            # self.logger.log_image(
            #     log_image, image_channels="first", step=self.iteration
            # )
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
            # {'type': 'sensor.stitch_camera.rgb', 'x': self.camera_x, 'y': 0, 'z': self.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            # 'width': 160, 'height': 240, 'fov': 60, 'id': f'Wide_RGB'},
            {'type': 'sensor.camera.rgb', 'x': self.config.sensors.camera_x,
             'y': 0, 'z': self.config.sensors.camera_z, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
             'width': self.config.sensors.img_width, 'height': self.config.sensors.img_height,
             'fov': self.config.sensors.fov, 'id': f'Narrow_RGB'},
        ]
        
        return sensors

    def postprocess(self, action: torch.Tensor):
        control = carla.VehicleControl()
        # print(f"{action = }")
        control.steer = torch.clip(action[0], -1.0, 1.0).item()
        if action[1] < -0.5:
            control.throttle = 0
            control.brake = torch.clip(-action[1], 0.0, 1.0).item()  # .cpu().numpy()
            control.steer = 0
        else:
            control.throttle = max(torch.clip(action[1], 0.0, 0.75).item(), 0.4)  # .cpu().numpy()
            control.brake = 0

        # control.manual_gear_shift = False

        return control

    @torch.no_grad()
    def run_step(self, input_data, timestamp):
        
        # _, wide_rgb = input_data.get(f'Wide_RGB')
        _, _rgb = input_data.get(f'Narrow_RGB')

        # # Crop images
        # _wide_rgb = wide_rgb[self.wide_crop_top:,:,:3]
        rgb = _rgb[...,:3].copy()
        rgb = rgb[..., ::-1] # BGR -> RGB
        # self.logger.log_image(rgb, f"before_{self.num_frames:05}")


        # _wide_rgb = _wide_rgb[...,::-1].copy()
        rgb = self.transform(rgb)
        # print(f"{rgb.shape = }")
        # self.logger.log_image(rgb, f"after_{self.num_frames}", image_channels='First')
        self.img_list.append(rgb)
        # log_i = torch.cat(list(self.img_list), dim=-1)
        # self.logger.log_image(log_i.permute(1, 2, 0).cpu(), f"after_{self.num_frames:05}")

        _, ego = input_data.get('EGO')
        _, gps = input_data.get('GPS')


        if self.waypointer is None:
            self.waypointer = Waypointer(self._global_plan, gps)

        _, _, cmd = self.waypointer.tick(gps)

        spd = ego.get('spd')
        
        cmd_value = cmd.value-1
        cmd_value = 3 if cmd_value < 0 else cmd_value
        print(f"{spd = }, {cmd_value = }")
        speed = torch.tensor(
            [spd / 10.0, ], dtype=torch.float
        ).unsqueeze_(0).to(self.device)
        # )
        # _cmd = int(observations["command"])
        command = torch.zeros(self.config.model.actor.n_commands).scatter_(
            dim=0, index=torch.tensor(int(cmd_value)), value=1
        ).unsqueeze_(0).to(self.device)
        # )
        images = torch.stack(list(self.img_list), dim=0).unsqueeze_(0).to(self.device)
        action = self.model.choose_action(images, speed, command).squeeze().cpu()
        control = self.postprocess(action)
        # print(f"{control}")
        measurements = {
            "control": [action[0], action[1]],
            "speed": speed,
            "command": command,
        }
        # # control = carla.VehicleControl(steer=0, brake=0, throttle=1.0)
        log_image = draw_on_image(
            rgb.cpu(), measurements, action.cpu(), False
        )
        # self.logger.log_image(log_image, f"after_{self.num_frames:05}")
        self.vizs.append(log_image)

        if len(self.vizs) > 1000:
            self.flush_data()

        self.num_frames += 1

        # if self.num_frames == 1:
        # #     control = carla.VehicleControl()
        # control = carla.VehicleControl(steer=0, brake=0, throttle=1.0)
        # control.steer = 0.0
        # control.brake = 0.0
        # control.throttle = 1.0
        # print(f"{control}")

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
    
    # def _lerp(self, v, x):
    #     D = v.shape[0]
    #
    #     min_val = self.min_speeds
    #     max_val = self.max_speeds
    #
    #     x = (x - min_val)/(max_val - min_val)*(D-1)
    #
    #     x0, x1 = max(min(math.floor(x), D-1),0), max(min(math.ceil(x), D-1),0)
    #     w = x - x0
    #
    #     return (1-w) * v[x0] + w * v[x1]
    #
    # def action_prob(self, steer_logit, throt_logit, brake_logit):
    #
    #     steer_logit = steer_logit.repeat(self.num_throts)
    #     throt_logit = throt_logit.repeat_interleave(self.num_steers)
    #
    #     action_logit = torch.cat([steer_logit, throt_logit, brake_logit[None]])
    #
    #     return torch.softmax(action_logit, dim=0)
    #
    # def post_process(self, steer, throt, brake_prob, spd, cmd):
    #
    #     if brake_prob > 0.5:
    #         steer, throt, brake = 0, 0, 1
    #     else:
    #         brake = 0
    #         throt = max(0.4, throt)
    #
    #     # # To compensate for non-linearity of throttle<->acceleration
    #     # if throt > 0.1 and throt < 0.4:
    #     #     throt = 0.4
    #     # elif throt < 0.1 and brake_prob > 0.3:
    #     #     brake = 1
    #
    #     if spd > {0:10,1:10}.get(cmd, 20)/3.6: # 10 km/h for turning, 15km/h elsewhere
    #         throt = 0
    #
    #     # if cmd == 2:
    #     #     steer = min(max(steer, -0.2), 0.2)
    #
    #     # if cmd in [4,5]:
    #     #     steer = min(max(steer, -0.4), 0.4) # no crazy steerings when lane changing
    #
    #     return steer, throt, brake
    
# def load_state_dict(model, path):
#
#     from collections import OrderedDict
#     new_state_dict = OrderedDict()
#     state_dict = torch.load(path)
#
#     for k, v in state_dict.items():
#         name = k[7:] # remove `module.`
#         new_state_dict[name] = v
#
#     model.load_state_dict(new_state_dict)

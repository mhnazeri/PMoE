"""Train stage 2: Train PMoE and its derivatives to predict actions"""

import sys

try:
    sys.path.append("../")
except:
    raise RuntimeError("Can't append root directory of the project the path")

import gc
from pathlib import Path
from datetime import datetime
from typing import Union

import comet_ml
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR
from thop import profile, clever_format

from loss import moe_loss, punet_loss, pmoe_loss
from model.moe import get_model
from model.data_loader import CarlaSegPred
from utils.nn import check_grad_norm, init_weights_normal, EarlyStopping, op_counter
from utils.vision import draw_on_image
from utils.io import save_checkpoint, load_checkpoint, worker_init_fn
from utils.utility import get_conf, timeit, class_labels


DEBUG = False


class Learner:
    def __init__(self, cfg_dir: Union[str, Path]):
        self.cfg = get_conf(cfg_dir)
        self.logger = self.init_logger(self.cfg.logger)
        self.dataset = CarlaSegPred(mode="train", **self.cfg.dataset)
        self.data = DataLoader(
            self.dataset, **self.cfg.dataloader, worker_init_fn=worker_init_fn
        )
        self.val_dataset = CarlaSegPred(mode="val", **self.cfg.val_dataset)
        self.val_data = DataLoader(self.val_dataset, **self.cfg.dataloader)
        if DEBUG:
            print(
                f"Train dataset length is {len(self.dataset)}"
                f"Validation dataset length is {len(self.val_dataset)}"
            )

        self.logger.log_parameters(
            {"tr_len": len(self.dataset), "val_len": len(self.val_dataset)}
        )
        self.model = get_model(self.cfg.model)
        if DEBUG:
            print(f"Model architecture:\n {self.model}")

        self.device = self.cfg.train_params.device
        self.model = self.model.to(device=self.device)

        if self.cfg.train_params.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                **self.cfg.adam,
            )
        elif self.cfg.train_params.optimizer.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                **self.cfg.rmsprop,
            )
        else:
            raise ValueError(f"Unknown optimizer {self.cfg.train_params.optimizer}")

        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.cfg.train_params.epochs
        )
        if self.cfg.model.type in ["moe", "moe_alt", "moe_shared"]:
            self.criterion = moe_loss
        elif self.cfg.model.type in ["punet", "punet_inter"]:
            self.criterion = punet_loss
        elif self.cfg.model.type in ["pmoe", "pmoe+pretrained"]:
            self.criterion = pmoe_loss
        else:
            raise ValueError(f"There is no loss for {self.cfg.model.type}")

        if self.cfg.logger.resume:
            # load checkpoint
            print("Loading checkpoint")
            load_dir = self.cfg.directory.load
            checkpoint = load_checkpoint(load_dir, self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint["epoch"]
            self.iteration = checkpoint["iteration"]
            self.logger.set_epoch(self.epoch)
            self.best = checkpoint["best"]
            self.e_loss = checkpoint["e_loss"]
            self.dice = checkpoint["dice"]
            print(
                f"Loading checkpoint was successful, start from epoch {self.epoch}"
                f" and loss {self.best}"
            )
        else:
            self.epoch = 1
            self.iteration = 1
            self.best = np.inf
            self.e_loss = []

        # initialize the early_stopping object
        self.early_stopping = EarlyStopping(
            patience=self.cfg.train_params.patience,
            verbose=True,
            delta=self.cfg.train_params.early_stopping_delta,
        )

        # stochastic weight averaging
        self.swa_model = AveragedModel(self.model)
        self.swa_scheduler = SWALR(self.optimizer, **self.cfg.SWA)

    def train(self):
        print_swa_start = True
        if DEBUG:
            torch.autograd.set_detect_anomaly(True)

        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []
            self.model.train()
            np.random.seed()
            bar = tqdm(
                enumerate(self.data),
                desc=f"Epoch {self.epoch}/{self.cfg.train_params.epochs}",
            )
            for idx, (img, measurements) in bar:
                # move data to device
                img = img.to(device=self.device)
                # measurements = {'control', 'speed', 'target_speed', 'command'}
                control = measurements["control"].to(device=self.device)
                speed = measurements["speed"].to(device=self.device).unsqueeze(-1)
                target_speed = (
                    measurements["target_speed"].to(device=self.device).unsqueeze(-1)
                )
                command = measurements["command"].to(device=self.device)

                # all models accepts: [images, speed, command] as input arguments
                # and returns actions, speed as predictions
                actions, speed_pred = self.model(img, speed, command)
                loss = self.criterion(
                    actions,
                    speed_pred,
                    control,
                    target_speed,
                    self.cfg.model.loss_coefs,
                )
                self.optimizer.zero_grad()
                loss.backward()
                # clip weights
                if self.cfg.train_params.grad_clipping > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train_params.grad_clipping)
                # check grad norm for debugging
                grad_norm = check_grad_norm(self.model)
                # update
                self.optimizer.step()

                running_loss.append(loss.item())
                bar.set_postfix(loss=loss.item(), Grad_Norm=grad_norm)
                self.logger.log_metrics(
                    {
                        "batch_loss": loss.item(),
                        "grad_norm": grad_norm,
                    },
                    epoch=self.epoch,
                    step=self.iteration
                )
                self.iteration += 1
            bar.close()
            if self.epoch >= self.cfg.train_params.swa_start:
                if print_swa_start:
                    print(f"Epoch {self.epoch}, step {self.iteration}, starting SWA!")
                    print_swa_start = False

                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                self.lr_scheduler.step()

            # validate on val set
            val_loss, t = self.validate()
            t /= len(self.val_dataset)  # time is calculated over the whole val_data

            # average loss for an epoch
            self.e_loss.append(np.mean(running_loss))  # epoch loss
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03} summary: train loss: {self.e_loss[-1]:.2f} \t| val loss: {val_loss:.2f}"
                f"\t| time: {t:.3f} seconds\n"
            )
            self.logger.log_metrics(
                {
                    "train_loss": self.e_loss[-1],
                    "val_loss": val_loss,
                    "time": t,
                },
                epoch=self.epoch
            )

            # early_stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop and self.cfg.train_params.early_stopping:
                print("Early stopping")
                self.save()
                break

            if self.epoch % self.cfg.train_params.save_every == 0 or (self.e_loss[-1] < self.best and
                                                                      self.epoch % self.cfg.train_params.start_saving_best == 0):

                self.save()

            gc.collect()
            self.epoch += 1

        # Update bn statistics for the swa_model at the end and save the model
        if self.epoch >= self.cfg.train_params.swa_start:
        #     torch.optim.swa_utils.update_bn(self.data, self.swa_model)
            for img, measurements in self.data:
                img = img.to(device=self.device)
                speed = measurements["speed"].to(device=self.device).unsqueeze(-1)
                command = measurements["command"].to(device=self.device)

                self.swa_model(img, speed, command)

            self.save(name=self.cfg.directory.model_name + "-final-swa")

        # macs, params = op_counter(self.model, sample=(img, speed, command))
        self.model.eval()
        macs, params = profile(self.model, inputs=(img, speed, command))
        macs, params = clever_format([macs, params], "%.3f")
        print(macs, params)
        self.logger.log_metrics({"GFLOPS": macs[:-1], "#Params": params[:-1]})
        print("ALL Finished!")

    @timeit
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = []

        for idx, (img, measurements) in tqdm(
            enumerate(self.val_data), desc="Validation"
        ):
            # move data to device
            img = img.to(device=self.device)
            # measurements = {'control', 'speed', 'target_speed', 'command'}
            control = measurements["control"].to(device=self.device)
            speed = measurements["speed"].to(device=self.device).unsqueeze(-1)
            command = measurements["command"].to(device=self.device)

            # forward, backward
            if self.epoch >= self.cfg.train_params.swa_start:
                action, _ = self.swa_model(img, speed, command)
                if 'punet' in self.cfg.model.type:
                    action = action.clamp(-1.0, 1.0)
                else:
                    action = action.sample().clamp(-1.0, 1.0)
            else:
                action = self.model.sample(img, speed, command).clamp(-1.0, 1.0)

            loss = torch.nn.functional.l1_loss(action, control)
            running_loss.append(loss.item())

        # if self.epoch % self.cfg.train_params.save_every == 0:
        # select a random sample to log
        idx = np.random.randint(0, img.size(0))
        measurements = {
            "control": measurements["control"][idx],
            "speed": measurements["speed"][idx],
            "command": measurements["command"][idx],
        }
        log_image = draw_on_image(
            img[idx, -1, ...].cpu(), measurements, action[idx].cpu()
        )
        self.logger.log_image(log_image, name=f"sample_{self.epoch:03}", step=self.iteration)

        # average loss
        loss = np.mean(running_loss)
        return loss

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
            logger.log_parameters(self.cfg)

        return logger

    def save(self, name=None):
        checkpoint = {
            "epoch": self.epoch,
            "iteration": self.iteration,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "best": self.best,
            "e_loss": self.e_loss,
        }

        if name is None and self.epoch >= self.cfg.train_params.swa_start:
            save_name = self.cfg.directory.model_name + "-e" + str(self.epoch) + "-swa"
            checkpoint["model-swa"] = self.swa_model.state_dict()
        elif name is None:
            save_name = self.cfg.directory.model_name + "-e" + str(self.epoch)
        else:
            save_name = name

        if self.e_loss[-1] < self.best:
            self.best = self.e_loss[-1]
            checkpoint["best"] = self.best
            save_checkpoint(checkpoint, True, self.cfg.directory.save, save_name)
        else:
            save_checkpoint(checkpoint, False, self.cfg.directory.save, save_name)


if __name__ == "__main__":
    cfg_path = "PMoE/conf/stage_2"
    learner = Learner(cfg_path)
    learner.train()

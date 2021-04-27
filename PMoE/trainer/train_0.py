"""Train stage 0: Train U-Net for image segmentation"""

import gc
from pathlib import Path
from datetime import datetime
import sys

try:
    sys.path.append(str(Path("../").resolve()))
except:
    raise RuntimeError("Can't append root directory of the project the path")

import comet_ml
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR

from loss import cross_entropy_dice_weighted_loss, class_dice
from model.blocks.unet import UNet
from model.data_loader import CarlaSeg
from utils.nn import check_grad_norm, init_weights_normal, EarlyStopping, op_counter
from utils.vision import decode_mask
from utils.io import save_checkpoint, load_checkpoint, worker_init_fn
from utils.utility import get_conf, timeit, class_labels


DEBUG = False


class Learner:
    def __init__(self, cfg_dir: str):
        self.cfg = get_conf(cfg_dir)
        self.logger = self.init_logger(self.cfg.logger)
        self.dataset = CarlaSeg(mode="train", **self.cfg.dataset)
        self.data = DataLoader(
            self.dataset, **self.cfg.dataloader, worker_init_fn=worker_init_fn
        )
        self.val_dataset = CarlaSeg(mode="val", **self.cfg.val_dataset)
        self.val_data = DataLoader(self.val_dataset, **self.cfg.dataloader)
        if DEBUG:
            print(
                f"Train dataset length is {len(self.dataset)}"
                f"Validation dataset length is {len(self.val_dataset)}"
            )

        self.logger.log_parameters(
            {"tr_len": len(self.dataset), "val_len": len(self.val_dataset)}
        )
        self.model = UNet(**self.cfg.model)
        if DEBUG:
            print(f"Model architecture:\n {self.model}")
        self.model.apply(init_weights_normal)
        self.device = self.cfg.train_params.device
        self.model = self.model.to(device=self.device)
        if self.cfg.train_params.optimizer.lower() == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), **self.cfg.adam)
        elif self.cfg.train_params.optimizer.lower() == "rmsprop":
            self.optimizer = optim.RMSprop(self.model.parameters(), **self.cfg.rmsprop)
        else:
            raise ValueError(f"Unknown optimizer {self.cfg.train_params.optimizer}")

        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=100
        )
        self.criterion = cross_entropy_dice_weighted_loss

        if self.cfg.logger.resume:
            # load checkpoint
            print("Loading checkpoint")
            load_dir = self.cfg.directory.load
            checkpoint = load_checkpoint(load_dir, self.device)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            self.epoch = checkpoint["epoch"]
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
            self.best = -np.inf
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
        if DEBUG:
            torch.autograd.set_detect_anomaly(True)
        while self.epoch <= self.cfg.train_params.epochs:
            running_loss = []
            self.model.train()
            np.random.seed()  # reset seed
            bar = tqdm(
                enumerate(self.data),
                desc=f"Epoch {self.epoch}/{self.cfg.train_params.epochs}",
            )
            for idx, (img, mask) in bar:
                self.optimizer.zero_grad()
                # move data to device
                img = img.to(device=self.device)
                mask = mask.to(device=self.device)

                # forward, backward
                out = self.model(img)
                loss = self.criterion(out, mask)
                loss.backward()
                # check grad norm for debugging
                grad_norm = check_grad_norm(self.model)
                # update
                self.optimizer.step()

                running_loss.append(loss.item())
                bar.set_postfix(loss=loss.item(), Grad_Norm=grad_norm)
                self.logger.log_metrics(
                    {
                        "epoch": self.epoch,
                        "batch": idx,
                        "batch_loss": loss.item(),
                        "GradNorm": grad_norm,
                    }
                )
            bar.close()
            # validate on val set
            val_loss, t = self.validate()
            t /= len(self.val_dataset)  # time is calculated over the whole val_data
            self.dice = val_loss[1]

            if self.epoch >= self.cfg.train_params.swa_start:
                self.swa_model.update_parameters(self.model)
                self.swa_scheduler.step()
            else:
                self.lr_scheduler.step()

            # average loss for an epoch
            self.e_loss.append(np.mean(running_loss))  # epoch loss
            print(
                f"{datetime.now():%Y-%m-%d %H:%M:%S} Epoch {self.epoch:03} summary: train loss: {self.e_loss[-1]:.2f} \t| val loss: {val_loss[0]:.2f}"
                f"\t| dice: {val_loss[1]:.2f} \t| time: {t:.3f} seconds"
            )
            self.logger.log_metrics(
                {
                    "epoch": self.epoch,
                    "epoch_loss": self.e_loss[-1],
                    "val_loss": val_loss[0],
                    "dice": val_loss[1],
                    "time": t,
                }
            )

            # early_stopping needs the validation loss to check if it has decreased,
            # and if it has, it will make a checkpoint of the current model
            self.early_stopping(val_loss[0], self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                self.save()
                break

            if self.epoch % self.cfg.train_params.save_every == 0:
                self.save()

            gc.collect()
            self.epoch += 1

        # Update bn statistics for the swa_model at the end and save the model
        if self.epoch >= self.cfg.train_params.swa_start:
            torch.optim.swa_utils.update_bn(self.data, self.swa_model)
            self.save(name=self.cfg.directory.model_name + "-final-swa")

        macs, params = op_counter(self.model, sample=img)
        print(macs, params)
        self.logger.log_metrics({"GFLOPS": macs[:-1], "#Params": params[:-1]})
        print("ALL Finished!")

    @timeit
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        running_loss = []
        running_dice = []

        for idx, (img, mask) in tqdm(enumerate(self.val_data), desc="Validation"):
            # move data to device
            img = img.to(device=self.device)
            mask = mask.to(device=self.device)

            # forward, backward
            if self.epoch >= self.cfg.train_params.swa_start:
                out = self.swa_model(img)
            else:
                out = self.model(img)

            loss = self.criterion(out, mask)
            running_loss.append(loss.item())
            d_loss = class_dice(out, mask)
            running_dice.append(d_loss)

        if self.epoch % self.cfg.train_params.save_every == 0:
            # only log 4 data samples from the last iteration
            for img_id, (image, predicted, gt) in enumerate(
                zip(img[:4, ...], out[:4, ...], mask[:4, ...])
            ):
                out_decoded = torch.from_numpy(decode_mask(predicted.cpu()))
                mask_decoded = torch.from_numpy(decode_mask(gt.cpu()))
                log_image = torch.cat([image.cpu(), out_decoded, mask_decoded], dim=-1)
                self.logger.log_image(
                    log_image, name="val_sample_" + str(img_id), image_channels="first"
                )

        # average loss
        loss = np.mean(running_loss)
        running_dice = torch.stack(running_dice, dim=0)
        dice = torch.mean(running_dice, dim=0).numpy()

        log_dice_class = {}
        for key, score in zip(class_labels.values(), dice):
            log_dice_class[key] = score

        self.logger.log_metrics(log_dice_class)
        # average all dices across all classes
        return loss, np.mean(dice)

    def init_logger(self, cfg):
        # Check to see if there is a key in environment:
        EXPERIMENT_KEY = cfg.experiment_key

        # First, let's see if we continue or start fresh:
        CONTINUE_RUN = cfg.resume
        if EXPERIMENT_KEY is not None:
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
                project_name=cfg.project,
                auto_histogram_weight_logging=True,
                auto_histogram_gradient_logging=True,
                auto_histogram_activation_logging=True,
            )
            logger.add_tags(cfg.tags.split())
            logger.log_parameters(self.cfg)

        return logger

    def save(self, name=None):
        checkpoint = {
            "epoch": self.epoch,
            "unet": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "best": self.best,
            "dice": self.dice,
            "e_loss": self.e_loss,
        }

        if name is None and self.epoch >= self.cfg.train_params.swa_start:
            save_name = self.cfg.directory.model_name + "-e" + str(self.epoch) + "-swa"
            checkpoint["unet-swa"] = self.swa_model.state_dict()
        elif name is None:
            save_name = self.cfg.directory.model_name + "-e" + str(self.epoch)
        else:
            save_name = name

        if self.dice > self.best:
            self.best = self.dice
            checkpoint["best"] = self.best
            save_checkpoint(checkpoint, True, self.cfg.directory.save, save_name)
        else:
            save_checkpoint(checkpoint, False, self.cfg.directory.save, save_name)


if __name__ == "__main__":
    cfg_path = "../conf/stage_0"
    learner = Learner(cfg_path)
    learner.train()
from typing import Any
import os
import random
import sys
from logging import getLogger
from pathlib import Path

import hydra
import librosa.display
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import streamvc.models
from streamvc.datasets import AudioFeatDataset, SingleCollator
from streamvc.utils.trainer import BaseTrainer

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


# A logger for this file
logger = getLogger(__name__)


def calc_label_accuracy(logits, targets):
    predictions = torch.argmax(logits, dim=1)
    correct_frames = (predictions == targets).float()
    accuracy = correct_frames.mean().item()

    return accuracy


class Trainer(BaseTrainer):
    def __init__(
        self,
        config,
        steps,
        epochs,
        data_loader,
        model,
        criterion,
        optimizer,
        scheduler,
        device=torch.device("cpu"),
    ):
        super().__init__(
            config=config,
            steps=steps,
            epochs=epochs,
            data_loader=data_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )

    def _train_step(self, batch):
        x, _, _, _, c_out = batch
        x = x.to(self.device)
        c_out = c_out.to(self.device)  # NOTE: Can only use discrete label

        with torch.autocast(
            device_type=self.device.type, enabled=self.config.train.amp.enabled
        ):
            # encoder forward
            outs = self.model["generator"].forward_encoder(x)
            outs = self.model["postnet"](outs)

            # calculate spectral loss
            enc_loss = self.criterion["cross_entropy"](outs, c_out.long())
            self.total_train_loss["train/encoder_loss"] += enc_loss.item()

            # update encoder/postnet
        self.optimizer["encoder"].zero_grad()

        self.scaler.scale(enc_loss).backward()
        if self.config.train.generator_grad_norm > 0:
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer["encoder"])
            # Since the gradients of optimizer's assigned params are unscaled
            torch.nn.utils.clip_grad_norm_(
                list(self.model["generator"].encoder.parameters())
                + list(self.model["postnet"].parameters()),
                self.config.train.generator_grad_norm,
            )

        scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer["encoder"])
        self.scaler.update()
        skip_scheduler = scale > self.scaler.get_scale()
        if not skip_scheduler:
            self.scheduler["encoder"].step()

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        x, _, _, _, c_out = batch
        x = x.to(self.device)
        c_out = c_out.to(self.device)  # NOTE: Can only use discrete label

        with torch.autocast(
            device_type=self.device.type, enabled=self.config.train.amp.enabled
        ):
            # encoder forward
            outs = self.model["generator"].forward_encoder(x)
            outs = self.model["postnet"](outs)

            enc_loss = self.criterion["cross_entropy"](outs, c_out.long())
            self.total_eval_loss["eval/encoder_loss"] += enc_loss.item()
            self.total_eval_loss["eval/encoder_accuracy"] += calc_label_accuracy(
                outs, c_out
            )

    @torch.no_grad()
    def _generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        pass


@hydra.main(version_base=None, config_path="config", config_name="train_encoder")
def main(config: DictConfig) -> None:
    """Run training process."""

    if "cuda" in config.device:
        device = torch.device(config.device)
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    # fix seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # check directory existence
    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # write config to yaml file
    with open(out_dir.joinpath("config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config))
    logger.info(OmegaConf.to_yaml(config))

    # get dataset
    if config.data.remove_short_samples:
        feat_length_threshold = config.data.batch_max_length // config.data.hop_size
    else:
        feat_length_threshold = None

    train_dataset = AudioFeatDataset(
        audio_list=config.data.train_audio,
        feat_list=config.data.train_feat,
        feat_length_threshold=feat_length_threshold,
        allow_cache=config.data.allow_cache,
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
        # NOTE: Only use discrete label for training encoder
        aux_feats=["discrete_label"],
    )
    logger.info(f"The number of training files = {len(train_dataset)}.")

    valid_dataset = AudioFeatDataset(
        audio_list=config.data.valid_audio,
        feat_list=config.data.valid_feat,
        feat_length_threshold=feat_length_threshold,
        allow_cache=config.data.allow_cache,
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
        aux_feats=["discrete_label"],
    )
    logger.info(f"The number of validation files = {len(valid_dataset)}.")

    dataset = {"train": train_dataset, "valid": valid_dataset}

    # get data loader
    collator = SingleCollator(
        batch_max_length=config.data.batch_max_length,
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
        lookahead_frames=config.data.lookahead_frames,
    )
    train_sampler, valid_sampler = None, None
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=True,
            collate_fn=collator,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            sampler=train_sampler,
            pin_memory=config.data.pin_memory,
        ),
        "valid": DataLoader(
            dataset=dataset["valid"],
            shuffle=True,
            collate_fn=collator,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            sampler=valid_sampler,
            pin_memory=config.data.pin_memory,
        ),
    }

    # define models and optimizers
    model = {
        "generator": streamvc.models.StreamVC(config.generator).to(device),
        "postnet": hydra.utils.instantiate(config.generator.postnet).to(device),
    }

    # define training criteria
    criterion = {
        "cross_entropy": hydra.utils.instantiate(config.train.cross_entropy_loss).to(
            device
        ),
    }

    # define optimizers and schedulers
    encoder_params = list(model["generator"].encoder.parameters()) + list(
        model["postnet"].parameters()
    )
    optimizer = {
        "encoder": hydra.utils.instantiate(
            config.train.optimizer,
            params=encoder_params,
        ),
    }
    scheduler = {
        "encoder": hydra.utils.instantiate(
            config.train.scheduler, optimizer=optimizer["encoder"]
        ),
    }

    # define trainer
    trainer = Trainer(
        config=config,
        steps=0,
        epochs=0,
        data_loader=data_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    # load trained parameters from checkpoint
    if config.train.resume:
        resume = out_dir.joinpath(
            "checkpoints", f"checkpoint-{config.train.resume}steps.pkl"
        )
        if resume.exists():
            trainer.load_checkpoint(resume)
            logger.info(f"Successfully resumed from {resume}.")
        else:
            logger.info(f"Failed to resume from {resume}.")
            sys.exit(0)
    else:
        logger.info("Start a new training process.")

    trainer.run()


if __name__ == "__main__":
    main()

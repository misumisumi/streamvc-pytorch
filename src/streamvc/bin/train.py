from typing import Any
import os
import random
import sys
from logging import getLogger
from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
import librosa.display
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

import streamvc.models
import streamvc.losses
from streamvc.datasets import AudioFeatDataset, SingleCollator
from streamvc.utils.trainer import BaseTrainer

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


# A logger for this file
logger = getLogger(__name__)


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
        x, c, s, y, c_out = batch

        x = x.to(self.device)
        c = c.to(self.device)
        s = s.to(self.device) if s is not None else None
        y = y.to(self.device)
        c_out = c_out.to(self.device)

        with torch.autocast(
            device_type=self.device.type, enabled=self.config.train.amp.enabled
        ):
            # generator forward
            y_hat = self.model["generator"](x, c, s=s)

            # calculate spectral loss
            mel_loss = self.criterion["mel"](y_hat, y)
            gen_loss = self.config.train.lambda_mel * mel_loss
            self.total_train_loss["train/mel_loss"] += mel_loss.item()

            # calculate source regularization loss
            # if self.config.train.lambda_reg > 0:
            #     s = outs[1]
            #     if isinstance(self.criterion["reg"], streamvc.losses.ResidualLoss):
            #         reg_loss = self.criterion["reg"](s, y, f0)
            #         gen_loss += self.config.train.lambda_reg * reg_loss
            #         self.total_train_loss["train/reg_loss"] += reg_loss.item()
            #     else:
            #         reg_loss = self.criterion["reg"](s, f0)
            #         gen_loss += self.config.train.lambda_reg * reg_loss
            #         self.total_train_loss["train/reg_loss"] += reg_loss.item()

            # calculate discriminator related losses
            if self.steps > self.config.train.discriminator_train_start_steps:
                # calculate feature matching loss
                if self.config.train.lambda_fm > 0:
                    p_fake, fmaps_fake = self.model["discriminator"](
                        y_hat, return_fmaps=True
                    )
                    with torch.no_grad():
                        p_real, fmaps_real = self.model["discriminator"](
                            y, return_fmaps=True
                        )
                    fm_loss = self.criterion["fm"](fmaps_fake, fmaps_real)
                    gen_loss += self.config.train.lambda_fm * fm_loss
                    self.total_train_loss["train/fm_loss"] += fm_loss.item()
                else:
                    p_fake = self.model["discriminator"](y_hat)
                # calculate adversarial loss
                adv_loss = self.criterion["adv"](p_fake)
                gen_loss += self.config.train.lambda_adv * adv_loss
                self.total_train_loss["train/adv_loss"] += adv_loss.item()

            self.total_train_loss["train/generator_loss"] += gen_loss.item()

            # update generator
        self.optimizer["generator"].zero_grad()

        self.scaler.scale(gen_loss).backward()
        if self.config.train.generator_grad_norm > 0:
            # Unscales the gradients of optimizer's assigned params in-place
            self.scaler.unscale_(self.optimizer["generator"])
            # Since the gradients of optimizer's assigned params are unscaled
            torch.nn.utils.clip_grad_norm_(
                self.model["generator"].parameters(),
                self.config.train.generator_grad_norm,
            )

        scale = self.scaler.get_scale()
        self.scaler.step(self.optimizer["generator"])
        skip_scheduler = scale > self.scaler.get_scale()
        if not skip_scheduler:
            self.scheduler["generator"].step()

        # discriminator
        if self.steps > self.config.train.discriminator_train_start_steps:
            with torch.autocast(
                device_type=self.device.type, enabled=self.config.train.amp.enabled
            ):
                # re-compute y_
                with torch.no_grad():
                    y_hat = self.model["generator"](x, c, s=s)
                # calculate discriminator loss
                p_fake = self.model["discriminator"](y_hat.detach())
                p_real = self.model["discriminator"](y)
                # NOTE: the first argument must to be the fake samples
                fake_loss, real_loss = self.criterion["adv"](p_fake, p_real)
                dis_loss = fake_loss + real_loss
                self.total_train_loss["train/fake_loss"] += fake_loss.item()
                self.total_train_loss["train/real_loss"] += real_loss.item()
                self.total_train_loss["train/discriminator_loss"] += dis_loss.item()

            # update discriminator
            self.optimizer["discriminator"].zero_grad()

            self.scaler.scale(dis_loss).backward()
            if self.config.train.discriminator_grad_norm > 0:
                self.scaler.unscale_(self.optimizer["discriminator"])
                torch.nn.utils.clip_grad_norm_(
                    self.model["discriminator"].parameters(),
                    self.config.train.discriminator_grad_norm,
                )
            scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer["discriminator"])
            skip_scheduler = scale > self.scaler.get_scale()
            if not skip_scheduler:
                self.scheduler["discriminator"].step()

        self.scaler.update()

    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        x, c, s, y, c_out = batch

        x = x.to(self.device)
        c = c.to(self.device)
        s = s.to(self.device) if s is not None else None
        y = y.to(self.device)
        c_out = c_out.to(self.device)

        with torch.autocast(
            device_type=self.device.type, enabled=self.config.train.amp.enabled
        ):
            # generator forward
            y_hat = self.model["generator"](x, c, s=s)
            # y_hat = outs[0]

        # calculate spectral loss
        mel_loss = self.criterion["mel"](y_hat, y)
        gen_loss = self.config.train.lambda_mel * mel_loss
        self.total_eval_loss["eval/mel_loss"] += mel_loss.item()

        # calculate source regularization loss for formant_hifigan-based models
        # if self.config.train.lambda_reg > 0:
        #     s = outs[1]
        #     if isinstance(
        #         self.criterion["reg"],
        #         streamvc.losses.ResidualLoss,
        #     ):
        #         reg_loss = self.criterion["reg"](s, y, f0)
        #         gen_loss += self.config.train.lambda_reg * reg_loss
        #         self.total_eval_loss["eval/reg_loss"] += reg_loss.item()
        #     else:
        #         reg_loss = self.criterion["reg"](s, f0)
        #         gen_loss += self.config.train.lambda_reg * reg_loss
        #         self.total_eval_loss["eval/reg_loss"] += reg_loss.item()

        # calculate discriminator related losses
        if self.steps > self.config.train.discriminator_train_start_steps:
            # calculate feature matching loss
            if self.config.train.lambda_fm > 0:
                p_fake, fmaps_fake = self.model["discriminator"](
                    y_hat, return_fmaps=True
                )
                p_real, fmaps_real = self.model["discriminator"](y, return_fmaps=True)
                fm_loss = self.criterion["fm"](fmaps_fake, fmaps_real)
                gen_loss += self.config.train.lambda_fm * fm_loss
                self.total_eval_loss["eval/fm_loss"] += fm_loss.item()
            else:
                p_fake = self.model["discriminator"](y_hat)
            # calculate adversarial loss
            adv_loss = self.criterion["adv"](p_fake)
            gen_loss += self.config.train.lambda_adv * adv_loss
            self.total_eval_loss["eval/adv_loss"] += adv_loss.item()

        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()

        # discriminator
        if self.steps > self.config.train.discriminator_train_start_steps:
            # calculate discriminator loss
            p_real = self.model["discriminator"](y)
            # NOTE: the first augment must to be the fake sample
            fake_loss, real_loss = self.criterion["adv"](p_fake, p_real)
            dis_loss = fake_loss + real_loss
            self.total_eval_loss["eval/fake_loss"] += fake_loss.item()
            self.total_eval_loss["eval/real_loss"] += real_loss.item()
            self.total_eval_loss["eval/discriminator_loss"] += dis_loss.item()

    @torch.no_grad()
    def _generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        x, c, s, y, c_out = batch

        x = x[:1].to(self.device)
        c = c[:1].to(self.device)
        s = s[:1].to(self.device) if s is not None else None
        y = y[:1].to(self.device)
        c_out = c_out[:1].to(self.device)

        # generator forward
        y_hat = self.model["generator"](x, c, s=s)

        len50ms = int(self.config.data.sample_rate * 0.05)
        start = np.random.randint(0, self.config.data.batch_max_length - len50ms)
        end = start + len50ms

        # for audio, name in zip((y,) + outs, ["real", "fake", "source"]):
        for audio, name in zip((y, y_hat), ["real", "fake"]):
            if audio is not None:
                audio = audio.view(-1).cpu().numpy()

                # plot spectrogram
                fig = plt.figure(figsize=(8, 6))
                spectrogram = np.abs(
                    librosa.stft(
                        y=audio,
                        n_fft=1024,
                        hop_length=128,
                        win_length=1024,
                        window="hann",
                    )
                )
                spectrogram_db = librosa.amplitude_to_db(spectrogram, ref=np.max)
                librosa.display.specshow(
                    spectrogram_db,
                    sr=self.config.data.sample_rate,
                    y_axis="linear",
                    x_axis="time",
                    hop_length=128,
                )
                self.writer.add_figure(f"spectrogram/{name}", fig, self.steps)
                plt.clf()
                plt.close()

                # plot full waveform
                fig = plt.figure(figsize=(6, 3))
                plt.plot(audio, linewidth=1)
                self.writer.add_figure(f"waveform/{name}", fig, self.steps)
                plt.clf()
                plt.close()

                # plot short term waveform
                fig = plt.figure(figsize=(6, 3))
                plt.plot(audio[start:end], linewidth=1)
                self.writer.add_figure(f"short_waveform/{name}", fig, self.steps)
                plt.clf()
                plt.close()

                # save as wavfile
                self.writer.add_audio(
                    f"audio_{name}.wav",
                    audio,
                    self.steps,
                    self.config.data.sample_rate,
                )


@hydra.main(version_base=None, config_path="config", config_name="train_decoder")
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
        allow_cache=config.data.allow_cache,
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
        aux_feats=config.data.aux_feats,
        use_spk_emb=config.data.use_spk_emb,
    )
    logger.info(f"The number of training files = {len(train_dataset)}.")

    valid_dataset = AudioFeatDataset(
        audio_list=config.data.valid_audio,
        feat_list=config.data.valid_feat,
        feat_length_threshold=feat_length_threshold,
        allow_cache=config.data.allow_cache,
        sample_rate=config.data.sample_rate,
        hop_size=config.data.hop_size,
        aux_feats=config.data.aux_feats,
        use_spk_emb=config.data.use_spk_emb,
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
        "discriminator": hydra.utils.instantiate(config.discriminator).to(device),
    }

    # define training criteria
    criterion = {
        "mel": hydra.utils.instantiate(config.train.mel_loss).to(device),
        "adv": hydra.utils.instantiate(config.train.adv_loss).to(device),
    }
    if config.train.lambda_fm > 0:
        criterion["fm"] = hydra.utils.instantiate(config.train.fm_loss).to(device)
    if config.train.lambda_reg > 0:
        criterion["reg"] = hydra.utils.instantiate(config.train.reg_loss).to(device)

    # define optimizers and schedulers
    optimizer = {
        "generator": hydra.utils.instantiate(
            config.train.generator_optimizer,
            params=model["generator"].parameters(),
        ),
        "discriminator": hydra.utils.instantiate(
            config.train.discriminator_optimizer,
            params=model["discriminator"].parameters(),
        ),
    }
    scheduler = {
        "generator": hydra.utils.instantiate(
            config.train.generator_scheduler, optimizer=optimizer["generator"]
        ),
        "discriminator": hydra.utils.instantiate(
            config.train.discriminator_scheduler, optimizer=optimizer["discriminator"]
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
    elif config.train.resume_path:
        resume = Path(to_absolute_path(config.train.resume_path))
        if resume.exists():
            trainer.load_checkpoint(resume, load_only_model=True)
            logger.info(f"Successfully resumed from {resume}.")
        else:
            logger.info(f"Failed to resume from {resume}.")
            sys.exit(0)
    else:
        logger.info("Start a new training process.")

    trainer.run()


if __name__ == "__main__":
    main()

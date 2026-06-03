# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based loss modules.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN

"""

from typing import Optional

from librosa.filters import mel as librosa_mel
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def stft(
    x,
    fft_size,
    hop_size,
    win_length,
    window,
    center=True,
    onesided=True,
    normalized=False,
    power=False,
):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    x_stft = torch.stft(
        x,
        fft_size,
        hop_size,
        win_length,
        window,
        center=center,
        onesided=onesided,
        normalized=normalized,
        return_complex=True,
    )
    real = x_stft.real
    imag = x_stft.imag

    if power:
        return torch.clamp(real**2 + imag**2, min=1e-7).transpose(2, 1)
    else:
        return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class MelSpectrogramLoss(nn.Module):
    """Mel-spectral L1 loss module."""

    def __init__(
        self,
        n_fft=1024,
        hop_length=120,
        win_length=1024,
        window="hann_window",
        fs=24000,
        n_mels=80,
        fmin=0,
        fmax=None,
        log_base=None,
        center=True,
        onesided=True,
        normalized=False,
    ):
        """Initialize MelSpectrogramLoss loss.

        Args:
            n_fft (int): FFT points.
            hop_length (int): Hop length.
            win_length (Optional[int]): Window length.
            window (str): Window type.
            fs (int): Sampling rate.
            n_mels (int): Number of Mel basis.
            fmin (Optional[int]): Minimum frequency of mel-filter-bank.
            fmax (Optional[int]): Maximum frequency of mel-filter-bank.

        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.register_buffer("window", getattr(torch, window)(self.win_length))
        self.fs = fs
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else fs / 2
        self.center = center
        self.normalized = normalized
        self.onesided = onesided
        melmat = librosa_mel(sr=fs, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax).T
        self.register_buffer("melmat", torch.from_numpy(melmat).float())
        if log_base is None:
            self.log_func = torch.log
        elif log_base == 2.0:
            self.log_func = torch.log2
        elif log_base == 10.0:
            self.log_func = torch.log10
        else:
            self.log_func = lambda x: torch.log(x) / torch.log(torch.tensor(log_base))

    def forward(self, x, y, use_mse=False):
        """Calculate Mel-spectral L1 loss.

        Args:
            x (Tensor): Generated waveform tensor (B, 1, T).
            y (Tensor): Groundtruth waveform tensor (B, 1, T).

        Returns:
            Tensor: Mel-spectral L1 loss value.

        """
        x = x.squeeze(1)
        y = y.squeeze(1)
        x_mag = stft(
            x,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            self.center,
            self.onesided,
            self.normalized,
        )
        y_mag = stft(
            y,
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            self.center,
            self.onesided,
            self.normalized,
        )
        x_mel = torch.clamp(torch.matmul(x_mag, self.melmat), min=1e-7)
        y_mel = torch.clamp(torch.matmul(y_mag, self.melmat), min=1e-7)

        if use_mse:
            mel_loss = F.mse_loss(self.log_func(x_mel), self.log_func(y_mel))
        else:
            mel_loss = F.l1_loss(x_mel, y_mel)

        return mel_loss


class MultiScaleMelSpectrogramLoss(nn.Module):
    """Multi-Scale spectrogram loss.

    Args:
        fs (int): Sampling rate.
        range_start (int): Power of 2 to use for the first scale.
        range_stop (int): Power of 2 to use for the last scale.
        window (str): Window type.
        n_mels (int): Number of mel bins.
        fmin (Optional[int]): Minimum frequency for Mel.
        fmax (Optional[int]): Maximum frequency for Mel.
        center (bool): Whether to use center window.
        normalized (bool): Whether to use normalized one.
        onesided (bool): Whether to use oneseded one.
        log_base (Optional[float]): Log base value.
        alphas (bool): Whether to use alphas as coefficients or not..
    """

    def __init__(
        self,
        fs: int = 22050,
        range_start: int = 6,
        range_end: int = 12,
        window: str = "hann_window",
        n_mels: int = 80,
        fmin: Optional[int] = 0,
        fmax: Optional[int] = None,
        center: bool = True,
        normalized: bool = False,
        onesided: bool = True,
        log_base: Optional[float] = 10.0,
        alphas: bool = True,
    ):
        super().__init__()
        mel_loss = list()
        self.alphas = list()
        self.total = 0
        self.normalized = normalized
        assert range_end > range_start, "error in index"
        for i in range(range_start, range_end):
            assert range_start > 2, "range start should be more than 2 for hop_length"
            mel_loss.append(
                MelSpectrogramLoss(
                    fs=fs,
                    n_fft=int(2**i),
                    hop_length=2 ** (i - 2),
                    win_length=2**i,
                    window=window,
                    n_mels=n_mels,
                    fmin=fmin,
                    fmax=fmax,
                    center=center,
                    normalized=normalized,
                    onesided=onesided,
                    log_base=log_base,
                )
            )
            if alphas:
                self.alphas.append(np.sqrt(2 ** (i - 1)))  # √(2/a)
            else:
                self.alphas.append(1)
            self.total += self.alphas[-1] + 1

        self.mel_loss = torch.nn.ModuleList(mel_loss)

    def forward(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate Mel-spectrogram loss.

        Args:
            y_hat (Tensor): Generated waveform tensor (B, 1, T).
            y (Tensor): Groundtruth waveform tensor (B, 1, T).


        Returns:
            Tensor: Mel-spectrogram loss value.
        """
        loss = 0.0
        for i in range(len(self.alphas)):
            l1 = self.mel_loss[i](y_hat, y)
            l2 = self.mel_loss[i](y_hat, y, use_mse=True)
            loss += l1 + self.alphas[i] * l2
        loss = loss / self.total

        return loss

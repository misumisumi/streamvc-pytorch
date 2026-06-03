import copy
from logging import getLogger

import torch
import torch.nn as nn
from torchaudio.functional import spectrogram
from .hifigan import HiFiGANMultiPeriodDiscriminator

# A logger for this file
logger = getLogger(__name__)


class UnivNetSpectralDiscriminator(nn.Module):
    """UnivNet spectral discriminator module."""

    def __init__(
        self,
        fft_size,
        hop_size,
        win_length,
        window="hann_window",
        kernel_sizes=[(3, 9), (3, 9), (3, 9), (3, 9), (3, 3), (3, 3)],
        strides=[(1, 1), (1, 2), (1, 2), (1, 2), (1, 1), (1, 1)],
        channels=32,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        use_weight_norm=True,
    ):
        """Initialize HiFiGAN scale discriminator module.

        Args:
            fft_size (list): FFT size.
            hop_size (int): Hop size.
            win_length (int): Window length.
            window (stt): Name of window function.
            kernel_sizes (list): List of kernel sizes in down-sampling CNNs.
            strides (list): List of stride sizes in down-sampling CNNs.
            channels (int): Number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()

        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))

        self.layers = nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == len(strides)

        # add first layer
        self.layers += [
            nn.Sequential(
                nn.Conv2d(
                    1,
                    channels,
                    kernel_sizes[0],
                    stride=strides[0],
                    bias=bias,
                ),
                getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        for i in range(1, len(kernel_sizes) - 2):
            self.layers += [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        bias=bias,
                    ),
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                )
            ]

        # add final layers
        self.layers += [
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel_sizes[-2],
                    stride=strides[-2],
                    bias=bias,
                ),
                getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            nn.Conv2d(
                channels,
                1,
                kernel_size=kernel_sizes[-1],
                stride=strides[-1],
                bias=bias,
            )
        ]

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of output tensors of each layer.

        """
        x = spectrogram(
            x,
            pad=self.win_length // 2,
            window=self.window,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            power=1.0,
            normalized=False,
        ).transpose(-1, -2)

        fmap = []
        for f in self.layers:
            x = f(x)
            if return_fmaps:
                fmap.append(x)

        if return_fmaps:
            return x, fmap
        else:
            return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.parametrizations.weight_norm(m)
                logger.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)


class UnivNetMultiResolutionSpectralDiscriminator(nn.Module):
    """UnivNet multi-resolution spectral discriminator module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        discriminator_params={
            "channels": 32,
            "kernel_sizes": [(3, 9), (3, 9), (3, 9), (3, 9), (3, 3), (3, 3)],
            "strides": [(1, 1), (1, 2), (1, 2), (1, 2), (1, 1), (1, 1)],
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.2},
        },
    ):
        """Initialize UnivNetMultiResolutionSpectralDiscriminator module.

        Args:
            fft_sizes (list): FFT sizes for each spectral discriminator.
            hop_sizes (list): Hop sizes for each spectral discriminator.
            win_lengths (list): Window lengths for each spectral discriminator.
            window (stt): Name of window function.
            discriminator_params (dict): Parameters for univ-net spectral discriminator module.

        """
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.discriminators = nn.ModuleList()

        # add discriminators
        for i in range(len(fft_sizes)):
            params = copy.deepcopy(discriminator_params)
            self.discriminators += [
                UnivNetSpectralDiscriminator(
                    fft_size=fft_sizes[i],
                    hop_size=hop_sizes[i],
                    win_length=win_lengths[i],
                    window=window,
                    **params,
                )
            ]

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        outs, fmaps = [], []
        for f in self.discriminators:
            if return_fmaps:
                out, fmap = f(x, return_fmaps)
                fmaps.extend(fmap)
            else:
                out = f(x)
            outs.append(out)

        if return_fmaps:
            return outs, fmaps
        else:
            return outs


class UnivNetMultiResolutionMultiPeriodDiscriminator(nn.Module):
    """UnivNet multi-resolution + multi-period discriminator module."""

    def __init__(
        self,
        # Multi-resolution discriminator related
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        spectral_discriminator_params={
            "channels": 32,
            "kernel_sizes": [(3, 9), (3, 9), (3, 9), (3, 9), (3, 3), (3, 3)],
            "strides": [(1, 1), (1, 2), (1, 2), (1, 2), (1, 1), (1, 1)],
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.2},
        },
        # Multi-period discriminator related
        periods=[2, 3, 5, 7, 11],
        period_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):
        """Initialize UnivNetMultiResolutionMultiPeriodDiscriminator module.

        Args:
            fft_sizes (list): FFT sizes for each spectral discriminator.
            hop_sizes (list): Hop sizes for each spectral discriminator.
            win_lengths (list): Window lengths for each spectral discriminator.
            window (stt): Name of window function.
            sperctral_discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.

        """
        super().__init__()
        self.mrd = UnivNetMultiResolutionSpectralDiscriminator(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            window=window,
            discriminator_params=spectral_discriminator_params,
        )
        self.mpd = HiFiGANMultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=period_discriminator_params,
        )

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of list of each discriminator outputs,
                which consists of each layer output tensors.
                Multi scale and multi period ones are concatenated.

        """
        if return_fmaps:
            mrd_outs, mrd_fmaps = self.mrd(x, return_fmaps)
            mpd_outs, mpd_fmaps = self.mpd(x, return_fmaps)
            outs = mrd_outs + mpd_outs
            fmaps = mrd_fmaps + mpd_fmaps

            return outs, fmaps
        else:
            mrd_outs = self.mrd(x)
            mpd_outs = self.mpd(x)
            outs = mrd_outs + mpd_outs

            return outs

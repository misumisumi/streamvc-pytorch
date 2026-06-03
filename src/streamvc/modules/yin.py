from torchyin import yin
import torch
import torch.nn as nn


# from https://github.com/brentspell/torch-yin/blob/main/torchyin/yin.py
class YinExtractor(nn.Module):
    def __init__(
        self,
        sample_rate: float,
        minf0: float = 75,
        maxf0: float = 600,
        frame_stride: float = 0.02,
        thresholds: list[float] = [0.05, 0.1, 0.15],
    ):
        self.sample_rate = sample_rate
        self.minf0 = minf0
        self.maxf0 = maxf0
        self.frame_stride = frame_stride
        self.thresholds = thresholds

    def process_one(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tau_min = int(self.sample_rate / self.maxf0)
        tau_max = int(self.sample_rate / self.minf0)
        frame_length = 2 * tau_max
        frame_stride = int(self.frame_stride * self.sample_rate)

        # Pad so YIN framing aligns with 20ms hop count (e.g., HuBERT)
        pad = max(0, (frame_length - frame_stride) // 2)
        if pad > 0:
            signal = torch.nn.functional.pad(signal, (pad, pad))

        # compute the fundamental periods
        frames = yin._frame(signal, frame_length, frame_stride)
        cmdf = yin._diff(frames, tau_max)[..., tau_min:]
        tau = yin._search(cmdf, tau_max, self.threshold)
        # gather で各フレームの推定周期でのCMDF値を取得
        cmdf_at_tau = torch.gather(cmdf, dim=-1, index=tau.unsqueeze(-1)).squeeze(-1)
        vuv = torch.where(
            cmdf_at_tau < self.threshold,
            torch.tensor(1, device=tau.device),
            torch.tensor(0, device=tau.device),
        )
        f0 = torch.where(
            tau > 0,
            self.sample_rate / (tau + tau_min + 1).type(signal.dtype),
            torch.tensor(0, device=tau.device).type(signal.dtype),
        )
        # normalize the f0 by the mean and std of the non-zero f0 values
        voiced = f0 > 0
        if voiced.any():
            mean = f0[voiced].mean()
            std = torch.clamp(f0[voiced].std(correction=0), min=1e-8)
            f0_whitening = (f0 - mean) / std
            f0_whitening = torch.where(
                voiced, f0_whitening, torch.tensor(0, device=f0.device).type(f0.dtype)
            )
        else:
            f0_whitening = torch.zeros_like(f0)

        f0_estimates = torch.stack([f0_whitening, vuv, cmdf_at_tau], dim=-1)

        # f0, V/UV, 累積平均差分関数 # (T, D)
        return f0.squeeze(0), f0_estimates.squeeze(0)

    def forward(self, signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f0s, f0_estimates_results = [], []
        for threshold in self.thresholds:
            f0, f0_estimates = self.process_one(signal)
            f0s.append(f0.reshape(-1, 1))  # (T, 1)
            f0_estimates_results.append(f0_estimates)  # (T, 3)

        f0 = torch.cat(f0s, dim=-1)  # (T, num_thresholds)
        # (T, 3*num_thresholds)
        f0_estimates_results = torch.stack(f0_estimates_results, dim=-1)

        return f0, f0_estimates_results

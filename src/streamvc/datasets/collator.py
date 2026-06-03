from logging import getLogger
import numpy as np
import torch

logger = getLogger(__name__)


def peak_normalize(waveform: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    音声波形のピーク正規化を行う関数
    """
    max_val = np.max(np.abs(waveform))
    normalized_waveform = waveform / (max_val + eps)

    return normalized_waveform


class SingleCollator(object):
    """Customized collator for Pytorch DataLoader in training."""

    def __init__(
        self,
        batch_max_length: int = 12000,
        sample_rate: int = 24000,
        hop_size: int = 120,
        lookahead_frames: int = 0,
        is_norm: bool = False,
    ):
        """Initialize customized collator for PyTorch DataLoader.

        Args:
            batch_max_length (int): The maximum length of batch.
            sample_rate (int): Sampling rate.
            hop_size (int): Hop size of auxiliary features.

        """
        if batch_max_length % hop_size != 0:
            batch_max_length += -(batch_max_length % hop_size)
        self.batch_max_length = batch_max_length
        self.batch_max_frames = batch_max_length // hop_size
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.lookahead_frames = lookahead_frames
        self.is_norm = is_norm

    def __call__(self, batch: list):
        """Convert into batch tensors.

        Args:
            batch: list of tuple of the pair of audio and features.

        Returns:
            Tensor, FloatTensor: Audio batch (B, 1, T).
            Tensor: Auxiliary feature batch (B, C, T').
            Tensor: Speaker Embedding batch (B, D).
        """
        x_batch, c_batch, spk_batch, y_batch, c_out_batch = [], [], [], [], []
        for idx in range(len(batch)):
            x, c, spk_emb = batch[idx]
            if len(c) > self.batch_max_frames:
                # randomly pickup with the batch_max_length length of the part
                start_frame = np.random.randint(
                    self.lookahead_frames, len(c) - self.batch_max_frames
                )
                start_step = start_frame * self.hop_size
                end_step = (start_frame + self.batch_max_frames) * self.hop_size
                x_ = x[start_step:end_step]
                if self.is_norm:
                    x_ = peak_normalize(x_)
                c_ = c[start_frame : start_frame + self.batch_max_frames]

                la_start_frame = start_frame - self.lookahead_frames
                la_start_step = la_start_frame * self.hop_size
                la_end_step = (la_start_frame + self.batch_max_frames) * self.hop_size
                y = x[la_start_step:la_end_step]
                c_out = c[la_start_frame : la_start_frame + self.batch_max_frames]
            else:
                logger.warn(
                    f"Removed short sample from batch ({x_.shape=}, {c_.shape=})."
                )
                continue
            x_batch += [x_.astype(np.float32)[np.newaxis, :]]  # [(1, T), ...]
            c_batch += [c_.astype(np.float32)]  # [(T', D), ...]
            y_batch += [y.astype(np.float32)[np.newaxis, :]]  # [(1, T), ...]
            c_out_batch += [c_out.astype(np.float32)]  # [(T', D), ...]
            if spk_emb is not None:
                spk_batch += [spk_emb]  # [(D,), ...]
        x_batch = torch.FloatTensor(np.array(x_batch))  # (B, 1, T)
        c_batch = torch.FloatTensor(np.array(c_batch))
        y_batch = torch.FloatTensor(np.array(y_batch))  # (B, 1, T)
        c_out_batch = torch.FloatTensor(np.array(c_out_batch))  # (B, T', D)
        if c_batch.ndim == 3:
            c_batch = c_batch.transpose(2, 1)  # (B, D, T')
            c_out_batch = c_out_batch.transpose(2, 1)  # (B, D, T')
        # NOTE: Set spk_feat=None, when not provid it
        spk_batch = torch.FloatTensor(np.array(spk_batch)) if spk_batch != [] else None

        return x_batch, c_batch, spk_batch, y_batch, c_out_batch

    def _check_length(self, x, c):
        """Assert the audio and feature lengths are correctly adjusted for upsamping."""
        assert len(x) == len(c) * self.hop_size

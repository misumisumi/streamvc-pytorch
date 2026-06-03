import random
from logging import getLogger
from multiprocessing import Manager
from os import PathLike
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
import torch
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset

from streamvc.utils.audio_io import load_wav
from streamvc.utils.file_io import check_filename, read_hdf5, read_txt

# A logger for this file
logger = getLogger(__name__)


class AudioFeatDataset(Dataset):
    """PyTorch compatible audio and acoustic feat. dataset."""

    def __init__(
        self,
        audio_list: str,
        feat_list: str,
        audio_length_threshold: Optional[int] = None,
        feat_length_threshold: Optional[int] = None,
        return_filename: bool = False,
        allow_cache: bool = False,
        sample_rate: int = 24000,
        hop_size: int = 120,
        aux_feats: list[str] = ["hubert"],
        use_spk_emb: bool = False,
        f0_factor: float = 1.0,
        formants_factor: list[float] = [1.0, 1.0, 1.0, 1.0],
    ):
        audio_datas = read_txt(to_absolute_path(audio_list))
        feat_datas = read_txt(to_absolute_path(feat_list))

        assert len(audio_datas) == len(feat_datas), (
            "The number of audio files and feature files must be the same."
        )

        self.datasets = list(zip(audio_datas, feat_datas))
        self.return_filename = return_filename
        self.allow_cache = allow_cache
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.aux_feats = aux_feats
        self.use_spk_emb = use_spk_emb
        self.f0_factor = f0_factor
        self.formants_factor = formants_factor
        logger.info(f"Feature type : {self.aux_feats}")

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(self.datasets))]

    def _load_sample(self, audio_file: str, feat_file: str) -> tuple:
        """Load feature from file.

        Args:
            audio_file: Path to the audio file (unused).
            feat_file: Path to the feature file.
            spk_feat_file: Path to the speaker feature file.

        Returns:
            ndarray: Auxiliary features (T', C).
            ndarray: Speaker feature (D,) or None.
        """
        wav, sr = load_wav(audio_file, self.sample_rate)

        # get auxiliary features
        aux_feats = []
        for feat_type in self.aux_feats:
            if feat_type in ["lcf0"]:
                aux_feat = read_hdf5(
                    to_absolute_path(feat_file), f"/{feat_type.replace('l', '')}"
                )
                aux_feat = np.log(aux_feat) + np.log(self.f0_factor)
            if feat_type in ["f0_estimates"]:
                aux_feat = read_hdf5(to_absolute_path(feat_file), f"/{feat_type}")
                aux_feat = aux_feat.reshape(-1, aux_feat.shape[-1])
            else:
                aux_feat = read_hdf5(to_absolute_path(feat_file), f"/{feat_type}")
                if feat_type in ["cf1", "cf2", "cf3", "cf4"]:
                    aux_feat *= self.formants_factor[int(feat_type[-1]) - 1]
            aux_feats += [aux_feat]
        if len(aux_feats) == 1:
            aux_feats = aux_feats[0]
        else:
            aux_feats = np.concatenate(aux_feats, axis=1, dtype=np.float32)

        if self.use_spk_emb:
            spk_emb = read_hdf5(to_absolute_path(feat_file), "/spk_emb").squeeze(0)
        else:
            spk_emb = None

        if self.return_filename:
            return Path(audio_file).name, wav, aux_feats, spk_emb
        else:
            return wav, aux_feats, spk_emb

    def __getitem__(self, idx):
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]
        items = self._load_sample(*self.datasets[idx])

        if self.allow_cache:
            self.caches[idx] = items

        return items  # return spk_id, items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.datasets)

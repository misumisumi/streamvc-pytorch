from typing import Union
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModel, AutoFeatureExtractor


class HuBERTDiscreteEncoder(nn.Module):
    def __init__(
        self,
        kmean_model: str,
        use_hubert_layer_idx: int = 6,
        hubert_model: str = "facebook/hubert-base-ls960",
    ):
        super().__init__()
        assert Path(kmean_model).is_file(), f"{kmean_model} not found!"

        self.hubert = AutoModel.from_pretrained(hubert_model)
        self.preprocessor = AutoFeatureExtractor.from_pretrained(hubert_model)
        self.use_hubert_layer_idx = use_hubert_layer_idx

        kmean_model = torch.load(kmean_model, map_location="cpu", weights_only=False)
        self.in_dim = kmean_model["n_features_in_"]
        self.cluster_centers_ = nn.Parameter(
            torch.from_numpy(kmean_model["cluster_centers_"])
        )

    def _batch_units(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"x.dim() should be 3, but got {x.dim()}"
        assert x.shape[-1] == self.in_dim, (
            f"features dim should be {self.in_dim}, but got {x.shape[-1]}"
        )
        b, t, d = x.shape
        x = x.contiguous().view(-1, x.shape[-1])  # (B*T, D)
        distances = torch.cdist(x, self.cluster_centers_, p=2.0)  # (B*T, n_clusters)
        labels = distances.argmin(dim=1)

        return labels.contiguous().view(b, t)  # (B, T)

    @torch.inference_mode()
    def forward(
        self, wav: torch.Tensor, return_hubert: bool = False
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # NOTE: HuBERT frame shift 20ms, window 25ms
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        z = self.hubert(wav, output_hidden_states=True).hidden_states[
            self.use_hubert_layer_idx
        ]
        units = self._batch_units(z)

        if return_hubert:
            return units, z
        else:
            return units


class HuBERTEncoder(nn.Module):
    def __init__(
        self,
        use_hubert_layer_idx: int = 6,
        hubert_model: str = "facebook/hubert-base-ls960",
    ):
        super().__init__()

        self.hubert = AutoModel.from_pretrained(hubert_model)
        self.preprocessor = AutoFeatureExtractor.from_pretrained(hubert_model)
        self.use_hubert_layer_idx = use_hubert_layer_idx

    @torch.inference_mode()
    def forward(
        self, wav: torch.Tensor, return_hubert: bool = False
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        # NOTE: HuBERT frame shift 20ms, window 25ms
        if wav.ndim == 3:
            wav = wav.squeeze(1)  # B, T
        wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
        z = self.hubert(wav, output_hidden_states=True).hidden_states[
            self.use_hubert_layer_idx
        ]
        z = z.transpose(1, 2)  # (B, D, T)

        return z

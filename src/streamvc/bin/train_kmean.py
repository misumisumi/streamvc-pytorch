import logging
import numpy as np
from pathlib import Path
from streamvc.utils import audio_io, file_io

import torch
from sklearn.cluster import MiniBatchKMeans

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModel

logger = logging.getLogger(__name__)


def batch_generator(files: list[str], batch_size: int):
    """ファイルリストをバッチサイズごとにジェネレータで返す"""
    for i in range(0, len(files), batch_size):
        yield files[i : i + batch_size]


def process(files: list[str], cfg: DictConfig) -> list[str]:
    hubert = AutoModel.from_pretrained(cfg.hubert_model).to(cfg.device)
    kmeans = MiniBatchKMeans(init="k-means++", n_clusters=cfg.n_clusters, n_init="auto")

    n_batches = (len(files) + cfg.batch_size - 1) // cfg.batch_size

    for batch in tqdm(batch_generator(files, cfg.batch_size), total=n_batches):
        features = []
        for file in batch:
            file = Path(to_absolute_path(file))
            wav, sr = audio_io.load_wav_to_torch(file, target_sr=16000)
            wav = wav.to(cfg.device).unsqueeze(0)
            with torch.inference_mode():
                outputs = hubert(wav, output_hidden_states=True)
                feature = outputs.hidden_states[cfg.use_hubert_layer_idx]
            features.append(feature.squeeze(0).cpu().numpy())
        features = np.concatenate(features, axis=0)
        kmeans.partial_fit(features)

    out_dir = Path(to_absolute_path(cfg.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir.joinpath(f"kmeans_{cfg.n_clusters}clusters.pt")
    torch.save(
        {
            "n_features_in_": kmeans.n_features_in_,
            "_n_threads": kmeans._n_threads,
            "cluster_centers_": kmeans.cluster_centers_,
        },
        out_path,
    )
    logger.info(f"Kmeans checkpoint is saved to {out_path}")


@hydra.main(version_base=None, config_path="config", config_name="train_kmean")
def main(cfg: DictConfig):
    file_list = Path(to_absolute_path(cfg.file_list))
    files = file_io.read_txt(file_list)

    process(files, cfg)


if __name__ == "__main__":
    main()

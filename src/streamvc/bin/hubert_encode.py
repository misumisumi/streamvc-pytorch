import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import torch.nn.functional as F

import torch
from transformers import AutoFeatureExtractor, AutoModel
from streamvc.utils import audio_io, file_io

from sklearn.metrics import pairwise_distances_argmin

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def extract_hubert_feature(out_dir, files: list[str], cfg: DictConfig) -> list[str]:
    hubert = AutoModel.from_pretrained(cfg.hubert_model).to(cfg.device)

    out_files = []
    for in_path in tqdm(files, total=len(files)):
        in_path = Path(to_absolute_path(in_path))
        wav, sr = audio_io.load_wav_to_torch(in_path, target_sr=16000)
        wav = wav.to(cfg.device).unsqueeze(0)
        with torch.inference_mode():
            outputs = hubert(wav, output_hidden_states=True)
        output = outputs.hidden_states[cfg.use_hubert_layer_idx]

        out_path = file_io.path_replace(in_path, cfg.in_dir, cfg.out_dir, ext=".npy")
        np.save(out_path, output.squeeze(0).cpu().numpy())
        out_files += [str(out_path)]

    return out_files


def create_descrete_label(out_dir, files: list[str], cfg: DictConfig) -> list[str]:
    assert cfg.kmeans_model is not None, (
        "kmeans_model must be specified to create labels."
    )
    kmeans_model = torch.load(cfg.kmeans_model, map_location="cpu", weights_only=False)
    centers = kmeans_model["cluster_centers_"]
    n_clusters = centers.shape[0]
    out_files = []
    for in_path in tqdm(files, total=len(files)):
        features = np.load(in_path)
        labels = pairwise_distances_argmin(features, centers)
        out_path = file_io.path_replace(in_path, cfg.in_dir, cfg.out_dir, ext=".npy")
        np.save(out_path, labels)
        out_files += [str(out_path)]
    return out_files


@hydra.main(version_base=None, config_path="config", config_name="hubert_encode")
def main(cfg: DictConfig):
    if "kmean_train_audio" in cfg.get("data", {}):
        in_file_list = cfg.data.kmean_train_audio
    else:
        in_file_list = cfg.file_list
    in_file_list = Path(to_absolute_path(in_file_list))
    files = file_io.load_files(in_file_list)

    out_dir = Path(to_absolute_path(cfg.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.encode_mode == "feature":
        logger.info("Extracting hubert features")
        out_files = extract_hubert_feature(out_dir, files, cfg)
        suffix = "hubert"
    elif cfg.encode_mode == "label":
        logger.info("Creating descrete labels from hubert features")
        out_files = create_descrete_label(out_dir, files, cfg)
        suffix = "hubert.descrete"
    else:
        raise ValueError(f"Unsupported encode_mode: {cfg.encode_mode}")

    out_file_list = in_file_list.with_suffix(f".{suffix}.list")
    with open(out_file_list, "w") as f:
        f.write("\n".join(out_files))
    logger.info(f"List of encoded features are saved to {out_file_list}")


if __name__ == "__main__":
    main()

from logging import getLogger
from pathlib import Path
from typing import Optional

import hydra
import torch
import numpy as np

from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from joblib import Parallel, delayed
import wespeaker
from torchyin import yin
import librosa

from streamvc.utils import audio_io, file_io, fixed
from streamvc.models.encoder.hubert import HuBERTDiscreteEncoder


logger = getLogger(__name__)

# Global variables for multiprocessing
_hubert_model: Optional[HuBERTDiscreteEncoder] = None
_spk_extractor: Optional[object] = None
_config: Optional[DictConfig] = None


def path_create(
    file_list: list[str],
    inputpath: str,
    outputpath: str,
    ext: Optional[str] = None,
):
    for filepath in file_list:
        path_replace(filepath, inputpath, outputpath, ext)


def path_replace(
    filepath: str, inputpath: str, outputpath: str, ext: Optional[str] = None
) -> str:
    filepath = str.replace(filepath, inputpath, outputpath)
    fpath = Path(filepath)
    if ext is not None:
        fpath = fpath.with_suffix(ext)
    fpath.parent.mkdir(parents=True, exist_ok=True)

    return str(fpath)


# from https://github.com/brentspell/torch-yin/blob/main/torchyin/yin.py
def yin_process(
    signal: torch.Tensor,
    sample_rate: float,
    pitch_min: float = 20,
    pitch_max: float = 20000,
    frame_stride: float = 0.01,
    threshold: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    tau_min = int(sample_rate / pitch_max)
    tau_max = int(sample_rate / pitch_min)
    frame_length = 2 * tau_max
    frame_stride = int(frame_stride * sample_rate)

    # Pad so YIN framing aligns with 20ms hop count (e.g., HuBERT)
    pad = max(0, (frame_length - frame_stride) // 2)
    if pad > 0:
        signal = torch.nn.functional.pad(signal, (pad, pad))

    # compute the fundamental periods
    frames = yin._frame(signal, frame_length, frame_stride)
    cmdf = yin._diff(frames, tau_max)[..., tau_min:]
    tau = yin._search(cmdf, tau_max, threshold)
    f0 = torch.where(
        tau > 0,
        sample_rate / (tau + tau_min + 1).type(signal.dtype),
        torch.tensor(0, device=tau.device).type(signal.dtype),
    )
    # gather で各フレームの推定周期でのCMDF値を取得
    cmdf_at_tau = torch.gather(cmdf, dim=-1, index=tau.unsqueeze(-1)).squeeze(-1)
    vuv = torch.where(
        tau > 0, torch.tensor(1, device=tau.device), torch.tensor(0, device=tau.device)
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
    return f0.squeeze(0).cpu().numpy(), f0_estimates.squeeze(0).cpu().numpy()


def _init_worker(config_dict: dict):
    """Initialize worker process with models.

    Args:
        config_dict: Serialized configuration dictionary.
    """
    global _hubert_model, _spk_extractor, _config

    # Reconstruct config from dict
    _config = OmegaConf.create(config_dict)

    # Load models in worker process
    _hubert_model = HuBERTDiscreteEncoder(**_config.hubert)
    _hubert_model = _hubert_model.to(_config.device)

    _spk_extractor = wespeaker.load_model(_config.wespeaker_model)
    _spk_extractor.set_device(_config.device)


def _process_file_worker(filepath: str):
    """Process file using global models (worker function).

    Args:
        filepath: Audio file path to process.
    """
    global _hubert_model, _spk_extractor, _config
    assert _config is not None, "Config not initialized in worker"
    assert _hubert_model is not None, "HuBERT model not initialized in worker"
    assert _spk_extractor is not None, "Speaker extractor not initialized in worker"
    return process(filepath, _config, _hubert_model, _spk_extractor)


def calc_energy(wav: torch.Tensor, sr: int, frame_stride: float = 0.02) -> torch.Tensor:
    hop_length = int(sr * frame_stride)
    frames = yin._frame(wav, hop_length, hop_length)
    energy = torch.var(frames, dim=-1)

    return energy


def process(
    filepath: str,
    config: DictConfig,
    hubert: HuBERTDiscreteEncoder,
    spk_extractor,
):
    """Process a single audio file.

    Args:
        filepath: Audio file path to process.
        config: Configuration object.
    """

    # wav, sr = audio_io.load_wav(filepath, target_sr=config.sample_rate)
    wav, sr = audio_io.load_wav(filepath, target_sr=16000)

    _wav = torch.from_numpy(wav).to(config.device).unsqueeze(0)
    f0s, f0_estimates_results = [], []
    for threshold in config.yin.thresholds:
        # NOTE: 20ms same as HuBERT's hop size.
        f0, f0_estimates = yin_process(
            _wav,
            16000,
            pitch_min=config.yin.minf0,
            pitch_max=config.yin.maxf0,
            frame_stride=0.02,
            threshold=threshold,
        )
        f0s.append(f0.reshape(-1, 1))  # (T, 1)
        f0_estimates_results.append(f0_estimates)

    discrete_label, hubert_fea = hubert(_wav, return_hubert=True)
    discrete_label = discrete_label.squeeze(0).cpu().numpy()  # (T, )
    hubert_fea = hubert_fea.squeeze(0).cpu().numpy()  # (T, D)

    spk_emb = spk_extractor.extract_embedding_from_pcm(_wav, sr)
    spk_emb = spk_emb.unsqueeze(0).cpu().numpy()  # (1, D)

    f0 = np.concatenate(f0s, axis=1)
    f0_estimates = np.concatenate(f0_estimates_results, axis=1)

    energy = calc_energy(_wav, sr).cpu().numpy().T  # (T, 1)

    assert fixed.check_length((discrete_label, hubert_fea, f0, f0_estimates, energy)), (
        "but frame length is different among features. Please check the input audio or adjust the YIN thresholds."
    )
    # assert fixed.check_length((f0, f0_estimates, energy)), (
    #     "but frame length is different among features. Please check the input audio or adjust the YIN thresholds."
    # )

    # Create output filename from input filename
    fname = Path(filepath).stem
    spk = Path(filepath).parent.name
    output_dir = Path(config.output_dir).joinpath(spk)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir.joinpath(f"{fname}.hdf5")

    file_io.write_hdf5(output_file, "/discrete_label", discrete_label)
    file_io.write_hdf5(output_file, "/hubert_fea", hubert_fea)
    file_io.write_hdf5(output_file, "/f0", f0)
    file_io.write_hdf5(output_file, "/f0_estimates", f0_estimates)
    file_io.write_hdf5(output_file, "/energy", energy)
    file_io.write_hdf5(output_file, "/spk_emb", spk_emb)


@hydra.main(version_base=None, config_path="config", config_name="preprocess")
def main(config: DictConfig):
    # show default argument
    logger.info(OmegaConf.to_yaml(config))
    file_list = file_io.read_txt(to_absolute_path(config.file_list))
    logger.info(f"number of utterances = {len(file_list)}")

    # Determine number of parallel jobs
    # Recommend: num_gpus or limited by memory
    n_jobs = getattr(config, "n_jobs", 1)

    # Process files in parallel
    if n_jobs > 1:
        logger.info(
            f"Processing {len(file_list)} files with {n_jobs} parallel jobs "
            "(multiprocessing backend)"
        )
        # Convert config to dict for serialization
        config_dict = OmegaConf.to_container(config, resolve=True)

        # Use multiprocessing backend with worker initialization
        Parallel(
            n_jobs=n_jobs,
            backend="loky",
            initializer=_init_worker,
            initargs=(config_dict,),
        )(
            delayed(_process_file_worker)(filepath)
            for filepath in tqdm(
                file_list, total=len(file_list), desc="Processing files"
            )
        )
    else:
        logger.info(f"Processing {len(file_list)} files sequentially")
        # init feature extractors
        hubert = HuBERTDiscreteEncoder(**config.hubert)
        spk_extractor = wespeaker.load_model(config.wespeaker_model)
        hubert = hubert.to(config.device)
        spk_extractor.set_device(config.device)

        for filepath in tqdm(file_list, total=len(file_list), desc="Processing files"):
            process(filepath, config, hubert, spk_extractor)

    logger.info("Preprocessing completed.")

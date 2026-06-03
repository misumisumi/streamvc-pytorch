import os
from logging import getLogger
from pathlib import Path
from time import time

import hydra
import numpy as np
import soundfile as sf
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

import streamvc.models
from streamvc.datasets import AudioFeatDataset
from streamvc.utils import file_io

from torchyin import yin

# A logger for this file
logger = getLogger(__name__)


def extract_f0(
    signal: torch.Tensor,
    sample_rate: float,
    pitch_min: float = 20,
    pitch_max: float = 20000,
    frame_stride: float = 0.01,
    threshold: float = 0.1,
    whitening: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tau_min = int(sample_rate / pitch_max)
    tau_max = int(sample_rate / pitch_min)
    frame_length = 2 * tau_max
    frame_stride = int(frame_stride * sample_rate)

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
    whitening_f0 = (f0 - f0[f0 > 0].mean()) / f0[f0 > 0].std()
    whitening_f0 = torch.where(
        f0 > 0, whitening_f0, torch.tensor(0, device=f0.device).type(f0.dtype)
    )

    # f0, V/UV, 累積平均差分関数
    return (
        f0.cpu().numpy(),
        whitening_f0.cpu().numpy(),
        vuv.cpu().numpy(),
        cmdf_at_tau.cpu().numpy(),
    )


@hydra.main(version_base=None, config_path="config", config_name="decode")
def main(config: DictConfig) -> None:
    """Run decoding process."""

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # set device
    if config.device != "":
        device = torch.device(config.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Decode on {device}.")

    # load pre-trained model from checkpoint file
    out_dir = Path(config.out_dir)
    if config.checkpoint_path is None:
        checkpoint_path = out_dir.joinpath(
            "checkpoints",
            f"checkpoint-{config.checkpoint_steps}steps.pkl",
        )
    else:
        checkpoint_path = config.checkpoint_path
    state_dict = torch.load(
        to_absolute_path(checkpoint_path), map_location="cpu", weights_only=False
    )
    logger.info(f"Loaded model parameters from {checkpoint_path}.")
    # model = hydra.utils.instantiate(config.generator)
    model = streamvc.models.StreamVC(config.generator)
    model.load_state_dict(state_dict["model"]["generator"])
    if hasattr(model, "remove_weight_norm"):
        model.remove_weight_norm()
    model.eval().to(device)
    param = sum(p.numel() for p in model.parameters())
    logger.info(f"Number of generator parameters: {param / 10**6:.1f} M")

    # check directory existence
    out_dir = Path(
        to_absolute_path(out_dir.joinpath("wav", str(config.checkpoint_steps)))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    for f0_factor in config.f0_factors:
        dataset = AudioFeatDataset(
            audio_list=config.data.test_audio,
            feat_list=config.data.test_feat,
            return_filename=True,
            sample_rate=config.data.sample_rate,
            hop_size=config.data.hop_size,
            aux_feats=config.data.aux_feats,
            use_spk_emb=config.data.use_spk_emb,
        )
        logger.info(f"The number of features to be decoded = {len(dataset)}.")

        with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
            total_rtf = 1.0
            for idx, items in enumerate(pbar, 1):
                fpath, wav, c, s = items

                # save output signal as PCM 16 bit wav file
                fpath = Path(fpath)
                utt_id = fpath.stem
                spk_id = fpath.parents[config.spkidx].name
                save_dir = out_dir.joinpath(spk_id)
                save_dir.mkdir(parents=True, exist_ok=True)
                save_path = save_dir.joinpath(f"{utt_id}_f{f0_factor:.2f}.wav")

                wav = torch.FloatTensor(wav).reshape(1, 1, -1).to(device)
                c = torch.FloatTensor(c).unsqueeze(0).transpose(2, 1).to(device)
                # if s is not None:
                # s = file_io.read_hdf5(
                #     "/localtmp2/sumiharu.kobayashi.r4/datasets/features/VCTK/train/p231/p231_021.hdf5",
                #     "/spk_emb",
                # )
                s = file_io.read_hdf5(
                    "/localtmp2/sumiharu.kobayashi.r4/datasets/features/VCTK/train/p232/p232_021.hdf5",
                    "/spk_emb",
                )
                # s = file_io.read_hdf5(
                #     "/localtmp2/kobayashi.sumiharu.r4/VCTK/hubert/p232/p232_002.h5",
                #     "/wespeaker",
                # )
                # if spk_id == "jvs001":
                #     s = file_io.read_hdf5(
                #         "data/hubert/JVS001_002/jvs002/parallel100/wav24kHz16bit/VOICEACTRESS100_006.h5",
                #         "/wespeaker",
                #     )
                # elif spk_id == "jvs002":
                #     s = file_io.read_hdf5(
                #         "data/hubert/JVS001_002/jvs001/parallel100/wav24kHz16bit/VOICEACTRESS100_006.h5",
                #         "/wespeaker",
                #     )
                # else:
                #     raise ValueError(f"Unsupported speaker ID: {spk_id}")
                s = torch.FloatTensor(s).unsqueeze(0).to(device)

                # perform decoding
                start = time()
                y = model(wav, c, s=s)
                rtf = (time() - start) / (y.size(-1) / config.data.sample_rate)
                pbar.set_postfix({"RTF": rtf})
                total_rtf += rtf

                y = y.view(-1).cpu().numpy()
                sf.write(save_path, y, config.data.sample_rate, "PCM_16")

                # save source signal as PCM 16 bit wav file
                if config.save_source:
                    save_path = save_path.replace(".wav", "_s.wav")
                    s = outs[1].view(-1).cpu().numpy()
                    s = s / np.max(np.abs(s))  # normalize
                    sf.write(save_path, s, config.data.sample_rate, "PCM_16")

            # report average RTF
            mean_rtf = total_rtf / len(dataset)
            logger.info(
                f"Finished generation of {idx} utterances (RTF: {mean_rtf:.6f}, ×{1 / mean_rtf:.3f})."
            )


if __name__ == "__main__":
    main()

import numpy as np
from scipy.signal import firwin, lfilter


def low_cut_filter(x: np.ndarray, sr: int, cutoff: float = 70) -> np.ndarray:
    """Low cut filter

    Args:
        x (ndarray): Waveform sequence
        sr (int): Sampling frequency
        cutoff (float): Cutoff frequency of low cut filter

    Return:
        (ndarray): Low cut filtered waveform sequence

    """
    nyquist = sr // 2
    norm_cutoff = cutoff / nyquist
    fil = firwin(255, norm_cutoff, pass_zero=False)
    lcf_x = lfilter(fil, 1, x)

    return lcf_x


def low_pass_filter(x: np.ndarray, sr: int, cutoff: float = 70, padding: bool = True) -> np.ndarray:
    """Low pass filter

    Args:
        x (ndarray): Waveform sequence
        sr (int): Sampling frequency
        cutoff (float): Cutoff frequency of low pass filter

    Return:
        (ndarray): Low pass filtered waveform sequence

    """
    nyquist = sr // 2
    norm_cutoff = cutoff / nyquist
    numtaps = 255
    fil = firwin(numtaps, norm_cutoff)
    x_pad = np.pad(x, (numtaps, numtaps), "edge")
    lpf_x = lfilter(fil, 1, x_pad)
    lpf_x = lpf_x[numtaps + numtaps // 2 : -numtaps // 2]

    return lpf_x

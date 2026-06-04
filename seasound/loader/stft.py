"""Utility for calculating STFT power from audio data."""

import numpy as np
from scipy import signal


def compute_stft_power(
    audio_pa: np.ndarray,
    sample_rate: int,
    nfft: int,
    win_length: int,
    hop_length: int,
    window: str = "hann",
    fmin_hz: float = 10.0,
    fmax_hz: float = 50000.0,
):
    """Compute STFT power for a single audio segment."""
    noverlap = win_length - hop_length
    freqs, times, Zxx = signal.stft( #pylint: disable=invalid-name
        audio_pa,
        fs=sample_rate,
        window=window,
        nperseg=win_length,
        noverlap=noverlap,
        nfft=nfft,
        boundary=None, # pyright: ignore[reportArgumentType]
        padded=False,
    )
    power = np.abs(Zxx) ** 2
    mask = (freqs >= fmin_hz) & (freqs <= fmax_hz)
    return freqs[mask], times, power[mask, :]

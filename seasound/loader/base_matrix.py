"""
seasound/loader/base_matrix.py

Compute the 1-second resolution third-octave band SPL matrix
following the JOMOPANS standard.

This is the most computationally expensive part of the pipeline.
It runs once per WAV file; all subsequent analyses read from the
cached result.

Algorithm (per 1-second window):
    1. Apply Hann window
    2. FFT with 4x zero-padding (JOMOPANS Section 3.3.2)
    3. Compute one-sided PSD (Pa²/Hz)
    4. For each TOB band, sum PSD within IEC 61260 band edges
    5. Convert band power to dB re reference pressure

The 4x zero-padding doesn't improve frequency resolution (which is
fixed at 1/window_length Hz) but it interpolates the spectral bins
more finely. This means that narrow TOB bands at low frequencies
(e.g. the 10 Hz band spans only ~2 Hz) get more FFT bins falling
inside them, producing smoother and more stable estimates.
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import signal

from seasound.core.config import ProcessingConfig
from seasound.utils.spectral import (
    tob_centre_frequencies,
    tob_band_edges,
    freq_column_names,
)

logger = logging.getLogger(__name__)


def compute_base_matrix(
    audio_pa: np.ndarray,
    sample_rate: int,
    config: ProcessingConfig,
) -> pd.DataFrame:
    """
    Compute 1-second resolution TOB SPL matrix (JOMOPANS standard).

    Parameters
    ----------
    audio_pa : np.ndarray
        Calibrated audio in Pascals. 1-D array (mono).
    sample_rate : int
        Sample rate in Hz.
    config : ProcessingConfig
        Processing parameters (max_freq_hz, min_freq_hz,
        base_resolution_s, reference_pressure_pa, missing_band_strategy).

    Returns
    -------
    pd.DataFrame
        Index: integer seconds (0, 1, 2, ...)
        Columns: TOB centre frequency strings like '63.0Hz'
        Values: SPL in dB re config.reference_pressure_pa

    Notes
    -----
    Memory management: for long files (hours of audio at 96 kHz+), are
    processed in chunks of 300 seconds (5 minutes) to avoid holding the
    full spectrogram in memory. Each chunk processes independently
    and results are written into a pre-allocated output array.
    """
    resolution_s = config.base_resolution_s
    ref_pressure = config.reference_pressure_pa

    # --- Determine which TOB bands are reachable ---
    nyquist = sample_rate / 2.0
    max_freq = min(config.max_freq_hz, nyquist)

    if max_freq < config.max_freq_hz:
        msg = (
            f"Sample rate {sample_rate} Hz cannot reach "
            f"max_freq_hz={config.max_freq_hz} Hz "
            f"(Nyquist={nyquist} Hz)"
        )
        strategy = config.missing_band_strategy
        if strategy == "error":
            raise ValueError(msg)
        elif strategy == "clip":
            logger.info(f"{msg} — clipping to {max_freq} Hz")
        else:  # "nan" — handled after computation
            logger.info(f"{msg} — unreachable bands will be NaN")

    # Get TOB centres and band edges
    all_centres = tob_centre_frequencies(config.min_freq_hz, config.max_freq_hz)
    reachable_centres = tob_centre_frequencies(config.min_freq_hz, max_freq)
    reachable_edges = tob_band_edges(reachable_centres)

    # --- Segment audio into time bins ---
    bin_samples = int(resolution_s * sample_rate)
    n_bins = len(audio_pa) // bin_samples

    if n_bins == 0:
        logger.warning(
            f"Audio shorter than {resolution_s}s — cannot compute base matrix"
        )
        return pd.DataFrame(columns=freq_column_names(all_centres))
    
    # Trim to exact number of bins
    audio_trimmed = audio_pa[: n_bins * bin_samples]

    # --- FFT parameters ---
    nperseg = bin_samples
    nfft = 4 * nperseg  # 4× zero-padding per JOMOPANS

    # --- Chunked processing for memory efficiency ---
    chunk_duration_s = 300  # 5 minutes
    bins_per_chunk = max(1, chunk_duration_s // resolution_s)
    n_chunks = int(np.ceil(n_bins / bins_per_chunk))

    # Pre-allocate output: (n_reachable_bands, n_bins)
    ltsa = np.full((len(reachable_centres), n_bins), np.nan)

    logger.info(
        f"Computing base matrix: {n_bins} × {resolution_s}s bins, "
        f"{len(reachable_centres)} TOB bands, "
        f"{n_chunks} chunk(s)"
    )

    for chunk_idx in range(n_chunks):
        bin_start = chunk_idx * bins_per_chunk
        bin_end = min(bin_start + bins_per_chunk, n_bins)

        # Slice audio for this chunk
        sample_start = bin_start * bin_samples
        sample_end = bin_end * bin_samples
        audio_chunk = audio_trimmed[sample_start:sample_end]

        # scipy.signal.spectrogram computes the STFT efficiently
        # - window='hann' applies the Hann window per segment
        # - noverlap=0 means non-overlapping segments (one per time bin)
        # - scaling='density' gives PSD in Pa²/Hz
        freqs, _, Sxx = signal.spectrogram(
            audio_chunk,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=0,
            nfft=nfft,
            scaling="density",
            mode="psd",
        )

        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0

        # --- Integrate PSD into TOB bands ---
        for b_idx, (f_lower, f_upper) in enumerate(reachable_edges):
            # Boolean mask: which FFT bins fall within this TOB band
            mask = (freqs >= f_lower) & (freqs <= f_upper)

            if np.any(mask):
                # Sum PSD × df across the band → band power in Pa²
                band_power = np.sum(Sxx[mask, :], axis=0) * df
            else:
                # No FFT bins in this band (can happen for very narrow
                # low-frequency bands). Use a floor value.
                band_power = np.full(bin_end - bin_start, 1e-30)

            # Floor to avoid log10(0)
            band_power = np.maximum(band_power, 1e-30)

            # Convert to dB re reference pressure
            # SPL = 20 × log10(RMS / ref) = 10 × log10(power / ref²)
            ltsa[b_idx, bin_start:bin_end] = (
                10.0 * np.log10(band_power / (ref_pressure ** 2))
            )

    # --- Build DataFrame ---
    # If max_freq was clipped, we only computed reachable bands.
    # For "nan" strategy, add NaN columns for unreachable bands.
    if config.missing_band_strategy == "nan" and len(all_centres) > len(reachable_centres):
        full_ltsa = np.full((len(all_centres), n_bins), np.nan)
        # Map reachable bands into the full array
        for i, fc in enumerate(reachable_centres):
            idx = np.argmin(np.abs(all_centres - fc))
            full_ltsa[idx, :] = ltsa[i, :]
        columns = freq_column_names(all_centres)
        data = full_ltsa.T  # transpose to (n_bins, n_bands)
    elif config.missing_band_strategy == "clip":
        columns = freq_column_names(reachable_centres)
        data = ltsa.T
    else:
        columns = freq_column_names(all_centres)
        if len(all_centres) == len(reachable_centres):
            data = ltsa.T
        else:
            # nan strategy with unreachable bands
            full_ltsa = np.full((len(all_centres), n_bins), np.nan)
            for i, fc in enumerate(reachable_centres):
                idx = np.argmin(np.abs(all_centres - fc))
                full_ltsa[idx, :] = ltsa[i, :]
            data = full_ltsa.T

    df_out = pd.DataFrame(data, columns=columns)
    df_out.index.name = "time_bin"

    return df_out

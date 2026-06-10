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

The base matrix is produced by streaming (_band_levels is the numerical
kernel):

- BaseMatrixAccumulator: push() whole-bin blocks as they are read,
  finalise() to assemble the DataFrame. Because noverlap=0 over
  independent 1-second bins, per-block output is bit-identical to
  processing the whole array at once (refactor plan D4; gated by the
  streaming identity tests).
"""

import logging
from datetime import datetime, timedelta  # noqa: F401  (timedelta kept for API parity)
from typing import Optional

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


# ---------------------------------------------------------------------------
# Shared numerical kernel
# ---------------------------------------------------------------------------


def _band_setup(
    sample_rate: int,
    config: ProcessingConfig,
) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Resolve the TOB band set for this sample rate, applying the
    configured missing-band strategy.

    Returns
    -------
    (all_centres, reachable_centres, reachable_edges)

    Raises
    ------
    ValueError
        When the strategy is "error" and the sample rate cannot reach
        config.max_freq_hz.
    """
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
        if strategy == "clip":
            logger.info("%s — clipping to %s Hz", msg, max_freq)
        else:  # "nan" — handled after computation
            logger.info("%s — unreachable bands will be NaN", msg)

    all_centres = tob_centre_frequencies(config.min_freq_hz, config.max_freq_hz)
    reachable_centres = tob_centre_frequencies(config.min_freq_hz, max_freq)
    reachable_edges = tob_band_edges(reachable_centres)
    return all_centres, reachable_centres, reachable_edges #type: ignore


def _band_levels(
    audio_chunk: np.ndarray,
    sample_rate: int,
    nperseg: int,
    nfft: int,
    reachable_edges: list,
    ref_pressure: float,
) -> np.ndarray:
    """
    The numerical kernel: TOB band SPL for one whole-bin chunk of
    calibrated audio. The streaming accumulator is its only caller, so
    the band numerics have a single implementation.

    Parameters
    ----------
    audio_chunk : np.ndarray
        Calibrated audio, length an exact multiple of nperseg, already
        cast to the configured Sxx dtype.

    Returns
    -------
    np.ndarray
        (n_reachable_bands, n_bins_in_chunk) float64 SPL values.
    """
    n_bins_chunk = len(audio_chunk) // nperseg

    # scipy.signal.spectrogram computes the STFT efficiently
    # - window='hann' applies the Hann window per segment
    # - noverlap=0 means non-overlapping segments (one per time bin)
    # - scaling='density' gives PSD in Pa²/Hz
    freqs, _, Sxx = signal.spectrogram( #pylint: disable=invalid-name
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

    levels = np.full((len(reachable_edges), n_bins_chunk), np.nan)

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
            band_power = np.full(n_bins_chunk, 1e-30)

        # Floor to avoid log10(0)
        band_power = np.maximum(band_power, 1e-30)

        # Convert to dB re reference pressure
        # SPL = 20 × log10(RMS / ref) = 10 × log10(power / ref²)
        levels[b_idx, :] = 10.0 * np.log10(band_power / (ref_pressure ** 2))

    return levels


def _assemble_frame(
    ltsa: np.ndarray,
    n_bins: int,
    all_centres: np.ndarray,
    reachable_centres: np.ndarray,
    strategy: str,
) -> pd.DataFrame:
    """
    Build the output DataFrame from the filled (n_bands, n_bins) array,
    applying the configured missing-band strategy.
    """
    # If max_freq was clipped, we only computed reachable bands.
    # For "nan" strategy, add NaN columns for unreachable bands.
    if strategy == "nan" and len(all_centres) > len(reachable_centres):
        full_ltsa = np.full((len(all_centres), n_bins), np.nan)
        # Map reachable bands into the full array
        for i, fc in enumerate(reachable_centres):
            idx = np.argmin(np.abs(all_centres - fc))
            full_ltsa[idx, :] = ltsa[i, :]
        columns = freq_column_names(all_centres)
        data = full_ltsa.T  # transpose to (n_bins, n_bands)
    elif strategy == "clip":
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


# ---------------------------------------------------------------------------
# Streaming API
# ---------------------------------------------------------------------------


class BaseMatrixAccumulator:
    """
    Streaming base-matrix builder over whole-bin blocks.

    Construct once per file/channel with the total bin count (known
    up front from the file header), push() calibrated whole-bin blocks
    in order as they are read, then finalise() to obtain a DataFrame
    bit-identical to the legacy whole-array computation (plan D4: with
    noverlap=0 and block boundaries on bin boundaries, no information
    crosses block seams).

    Memory: holds only the (n_bands, n_bins) output array — a few MB
    per hour of audio — never the audio itself.

    Parameters
    ----------
    sample_rate : int
        Sample rate in Hz.
    n_bins : int
        Total number of whole bins the file will produce
        (usable_samples // bin_samples). May be 0 for audio shorter
        than one bin; finalise() then reproduces the legacy empty
        frame.
    config : ProcessingConfig
    """

    def __init__(
        self,
        sample_rate: int,
        n_bins: int,
        config: ProcessingConfig,
    ):
        self.sample_rate = sample_rate
        self.n_bins = n_bins
        self._config = config
        self._ref_pressure = config.reference_pressure_pa

        self._all_centres, self._reachable_centres, self._reachable_edges = (
            _band_setup(sample_rate, config)
        )

        self._nperseg = int(config.base_resolution_s * sample_rate)
        self._nfft = config.nfft_padding_factor * self._nperseg
        self._sxx_np_dtype = (
            np.float64 if config.sxx_dtype == "float64" else np.float32
        )

        self._ltsa = np.full((len(self._reachable_centres), n_bins), np.nan)
        self._cursor = 0  # bins filled so far
        self._anchor_t0: Optional[datetime] = None  # enables the t0 guard

        if n_bins == 0:
            logger.warning(
                "Audio shorter than %ss — cannot compute base matrix",
                config.base_resolution_s
            )
        else:
            logger.info(
                "Computing base matrix (streaming): %s × %ss bins, %s TOB bands",
                n_bins,
                config.base_resolution_s,
                len(self._reachable_centres),
            )

    @property
    def bins_filled(self) -> int:
        return self._cursor

    @property
    def _expected_t0(self) -> Optional[datetime]:
        if self._anchor_t0 is None:
            return None
        return self._anchor_t0 + timedelta(
            seconds=self._cursor * self._config.base_resolution_s
        )

    def set_anchor(self, datetime_start: Optional[datetime]) -> None:
        """Set the file start time enabling the per-push t0 guard."""
        self._anchor_t0 = datetime_start

    def push(self, block_pa: np.ndarray, t0: Optional[datetime] = None) -> None:
        """
        Process one calibrated audio block.

        Parameters
        ----------
        block_pa : np.ndarray
            Calibrated 1-D audio whose length is an exact multiple of
            bin_samples (the AudioBlockReader guarantees this).
        t0 : datetime, optional
            Block start time. When provided it is checked against the
            expected position — a cheap guard against reader
            sequencing bugs.
        """
        if len(block_pa) % self._nperseg != 0:
            raise ValueError(
                f"Block length {len(block_pa)} is not a whole multiple of "
                f"bin_samples={self._nperseg}"
            )
        k = len(block_pa) // self._nperseg
        if self._cursor + k > self.n_bins:
            raise ValueError(
                f"Block overruns the expected {self.n_bins} bins "
                f"(cursor={self._cursor}, block bins={k})"
            )
        if t0 is not None and self._expected_t0 is not None:
            if t0 != self._expected_t0:
                raise ValueError(
                    f"Block start {t0} does not match expected "
                    f"{self._expected_t0} — blocks must arrive in order "
                    f"with no gaps"
                )

        chunk = np.asarray(block_pa, dtype=self._sxx_np_dtype)
        self._ltsa[:, self._cursor:self._cursor + k] = _band_levels(
            chunk,
            self.sample_rate,
            self._nperseg,
            self._nfft,
            self._reachable_edges,
            self._ref_pressure,
        )
        self._cursor += k

    def finalise(
        self,
        datetime_start: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Assemble the output DataFrame.

        With datetime_start given, applies the same DatetimeIndex
        conversion the pipeline applies to the legacy per-file matrix
        (hardcoded freq="1s", reproducing current behaviour — see the
        known base_resolution_s caveat recorded in the plan amendments).

        Raises
        ------
        ValueError
            If fewer bins were pushed than promised at construction —
            a silent short-fill must never finalise.
        """
        if self._cursor != self.n_bins:
            raise ValueError(
                f"Accumulator finalised with {self._cursor}/{self.n_bins} "
                f"bins filled"
            )

        if self.n_bins == 0:
            # Reproduce the legacy empty frame exactly (all-centre
            # columns, default index, no index name).
            frame = pd.DataFrame(
                columns=freq_column_names(self._all_centres)
            )
        else:
            frame = _assemble_frame(
                self._ltsa,
                self.n_bins,
                self._all_centres,
                self._reachable_centres,
                self._config.missing_band_strategy,
            )

        if (
            datetime_start is not None
            and not isinstance(frame.index, pd.DatetimeIndex)
        ):
            dt_index = pd.date_range(
                start=datetime_start,
                periods=len(frame),
                freq="1s",
            )
            frame = frame.copy()
            frame.index = dt_index
            frame.index.name = "datetime"

        return frame

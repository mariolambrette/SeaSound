"""
Unified method for computing/getting STFT power for a WAV file in analysis
modules.

The module provides cache-first STFT retrieval with an on-demand calculation
fallback.

get_stft_for_file() is the main entry point. It will check if a cache file 
exists for the given WAV and channel, and if so load it. If not, it will compute
the STFT power from the calibrated audio data, and save to cache if enabled.
"""

from __future__ import annotations

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any

from seasound.core.config import PipelineConfig
from seasound.loader.reader import read_audio
from seasound.loader.filename_parsers import get_parser
from seasound.loader.calibration import load_calibration, apply_calibration
from seasound.loader.stft import compute_stft_power
from seasound.loader.cache import load_stft_npz, save_stft_npz

logger = logging.getLogger(__name__)


def _stft_cache_path(wav_path: str, channel: int, cache_dir: str) -> str:
    """Derive cache path for STFT power from WAV path and channel."""
    base = os.path.splitext(os.path.basename(wav_path))[0]
    return os.path.join(cache_dir, f"{base}_ch{channel}_stft.npz")


def get_stft_for_file(
    wav_path: str,
    config: PipelineConfig,
    cache_dir: str,
) -> list[dict[str, Any]]:
    """
    Return STFT power arrays for all output segments/channels from one WAV file.

    Behaviour:
        1) If STFT cache is enabled and file exists, load cache
        2) Else compute STFT on demand from calibrated audio
        3) If STFT cache is enables, persist computed STFT.

    Returns
    -------
    list[dict[str, Any]]
        One entry per channel with keys:
            - channel
            - serial
            - datetime_start
            - freqs_hz
            - times_s
            - power
    """

    parser = get_parser(config.input)
    segments = read_audio(wav_path, config.input, parser=parser)
    cal_df = load_calibration(config.calibration)

    out: list[dict[str, Any]] = []
    cache_enabled = bool(config.pipeline.stft_cache_enabled and cache_dir)


    for seg in segments:
        cache_path = (
            _stft_cache_path(seg.source_file, seg.channel, cache_dir)
            if cache_enabled
            else None
        )

        need_compute = True
        if cache_enabled and cache_path and os.path.isfile(cache_path):
            try:
                with load_stft_npz(cache_path) as z:
                    freqs_hz = np.asarray(z["freqs_hz"])
                    times_s = np.asarray(z["times_s"])
                    power = np.asarray(z["power"])
                need_compute = False
            except Exception as exc: # pylint: disable=broad-except
                logger.warning(
                    "Could not read STFT cache '%s' (%s); recomputing.",
                    cache_path,
                    exc,
                )
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        if need_compute:
            audio_pa, _ = apply_calibration(seg, cal_df, config.calibration)
            freqs_hz, times_s, power = compute_stft_power(
                audio_pa=audio_pa,
                sample_rate=seg.sample_rate,
                nfft=config.pipeline.stft_nfft,
                win_length=config.pipeline.stft_win_length,
                hop_length=config.pipeline.stft_hop_length,
                window=config.pipeline.stft_window,
                fmin_hz=config.pipeline.stft_fmin_hz,
                fmax_hz=config.pipeline.stft_fmax_hz,
            )

            if config.pipeline.stft_dtype == "float16":
                power = power.astype(np.float16)
            else:
                power = power.astype(np.float32)

            if cache_enabled:
                save_stft_npz(freqs_hz, times_s, power, seg, cache_dir)

        out.append({
            "channel": seg.channel,
            "serial": seg.serial,
            "datetime_start": seg.datetime_start,
            "freqs_hz": freqs_hz, # type: ignore
            "times_s": times_s, # type: ignore
            "power": power, # type: ignore
        })

    return out


def build_stft_matrix(
    runtime_context: dict,
    *,
    time_bins: int | None = 12000,
) -> tuple[pd.DataFrame | None, list[str]]:
    """
    STFT matrix construction.

    Builds a time-frequency DataFrame (DateTimeIndex × Hz columns) from STFT
    frames stored in the pipeline runtime context. Both SpectrogramAnalysis
    and EventDetectionAnalysis (when producing annotated spectrograms) call
    this helper.

    Parameters
    ----------
    runtime_context : dict
        Pipeline runtime context. Expected keys: ``pipeline_config``
        (PipelineConfig), ``cache_dir`` (str), ``input_files`` (list[str]).
    time_bins : int or None
        If positive, the returned matrix is downsampled (mean) so it has
        approximately this many time bins. Useful for visual outputs to
        keep memory bounded. ``None`` disables downsampling.

    Returns
    -------
    (matrix, warnings) :
        - ``matrix``: DataFrame with DateTimeIndex (named 'datetime') and
          Hz-named columns ('<freq>Hz'), values in dB re reference pressure.
          ``None`` if STFT data isn't available from cache or on-demand
          computation.
        - ``warnings``: human-readable warning strings to surface to the
          caller.
    """
    warnings: list[str] = []
    cfg_obj = runtime_context.get("pipeline_config")
    cache_dir = runtime_context.get("cache_dir")
    input_files = runtime_context.get("input_files")

    if not isinstance(cfg_obj, PipelineConfig):
        warnings.append(
            "Pipeline config not available in runtime context; "
            "cannot build STFT matrix."
        )
        return None, warnings
    if not isinstance(cache_dir, str):
        warnings.append(
            "Cache directory not available in runtime context; "
            "cannot build STFT matrix."
        )
        return None, warnings
    if not isinstance(input_files, list) or not input_files:
        warnings.append(
            "Input files not available in runtime context; "
            "cannot build STFT matrix."
        )
        return None, warnings

    ref_pressure = float(cfg_obj.pipeline.reference_pressure_pa)
    if ref_pressure <= 0:
        warnings.append(
            f"Invalid reference pressure {ref_pressure} Pa; must be > 0. "
            "Cannot compute STFT-derived SPL values."
        )
        return None, warnings

    # Visual downsampling is applied at the matrix level after assembly.
    # The note here is informational; the .npz cache stays at native resolution.
    if time_bins is not None and time_bins > 0:
        warnings.append(
            f"STFT matrix assembly targets approximately {time_bins} time "
            f"bins; STFT .npz cache remains at native resolution."
        )

    frames: list[pd.DataFrame] = []
    for wav_path in input_files:
        try:
            entries = get_stft_for_file(wav_path, cfg_obj, cache_dir)
        except Exception as exc: #pylint: disable=broad-except
            warnings.append(
                f"Could not load/compute STFT for "
                f"{os.path.basename(wav_path)}: {exc}."
            )
            continue

        for entry in entries:
            dt_start = entry.get("datetime_start")
            if dt_start is None or not isinstance(
                dt_start, (pd.Timestamp, datetime)
            ):
                continue

            freqs_hz = np.asarray(entry.get("freqs_hz"))
            times_s = np.asarray(entry.get("times_s"))
            power = np.asarray(entry.get("power"))
            if power.ndim != 2:
                continue

            safe_power = np.maximum(
                power.astype(np.float32), np.finfo(np.float32).tiny,
            )
            power_db = (
                10.0 * np.log10(safe_power / np.float32(ref_pressure ** 2))
            ).astype(np.float32)

            abs_times = (
                pd.Timestamp(dt_start) + pd.to_timedelta(times_s, unit="s")
            )
            cols = [f"{float(f):.2f}Hz" for f in freqs_hz]
            frame = pd.DataFrame(power_db.T, index=abs_times, columns=cols)
            frame.index.name = "datetime"
            frames.append(frame)

    if not frames:
        warnings.append(
            "No valid STFT frames could be loaded or computed."
        )
        return None, warnings

    matrix = pd.concat(frames).sort_index()
    matrix = matrix[~matrix.index.duplicated(keep="first")]

    # Apply matrix-level downsampling for visual outputs.
    if time_bins is not None and time_bins > 0 and len(matrix) > 1:
        duration = matrix.index.max() - matrix.index.min()
        if isinstance(duration, pd.Timedelta) and duration > pd.Timedelta(0):
            step_seconds = max(1.0, duration.total_seconds() / time_bins)
            downsample_step = pd.Timedelta(seconds=step_seconds)

            diffs = matrix.index.to_series().diff().dropna()
            positive_diffs = diffs[diffs > pd.Timedelta(0)]
            native_step = (
                positive_diffs.median()
                if len(positive_diffs) > 0
                else pd.Timedelta(seconds=0)
            )
            if not isinstance(native_step, pd.Timedelta):
                native_step = pd.Timedelta(seconds=0)

            if downsample_step > native_step:
                matrix = matrix.resample(downsample_step).mean()
                matrix = matrix.dropna(how="all")

    return matrix, warnings

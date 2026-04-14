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
import numpy as np
from typing import Any

from seasound.core.config import PipelineConfig
from seasound.loader.reader import read_audio
from seasound.loader.filename_parsers import get_parser
from seasound.loader.calibration import load_calibration, apply_calibration
from seasound.loader.stft import compute_stft_power
from seasound.loader.cache import load_stft_npz, save_stft_npz



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

        if cache_enabled and cache_path and os.path.isfile(cache_path):
            z = load_stft_npz(cache_path)
            freqs_hz, times_s, power = z["freqs_hz"], z["times_s"], z["power"]
        
        else:
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
            "freqs_hz": freqs_hz,
            "times_s": times_s,
            "power": power,
        })

    return out

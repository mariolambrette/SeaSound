"""
seasound/core/stft.py

Windowed STFT matrix construction over the chunked shard store
(refactor plan §8).

``iter_stft_windows`` is the bounded, per-``time_chunk`` primitive the
spectrogram and annotated-spectrogram consumers iterate: each window
reads only its own native frames from the store, downsamples on the
*global* grid, and yields a dB DataFrame identical to slicing the
legacy deployment-wide matrix by ``time_chunk``. ``build_stft_matrix``
is the whole-extent convenience built on the same primitive (one
window spanning the full extent), preserved for callers that want the
single matrix.

Both produce dB re reference pressure; the store holds linear power, so
the dB conversion and the ``<freq>Hz`` column naming live here, exactly
as the previous ``build_stft_matrix`` did.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd

from seasound.core.config import PipelineConfig
from seasound.loader.stft_store import StftStore

logger = logging.getLogger(__name__)


def _resolve_context(
    runtime_context: dict, warnings: list[str],
) -> tuple[PipelineConfig | None, str | None]:
    """Pull (pipeline_config, cache_dir) from the runtime context,
    appending a warning and returning Nones if either is unusable."""
    cfg_obj = runtime_context.get("pipeline_config")
    cache_dir = runtime_context.get("cache_dir")

    if not isinstance(cfg_obj, PipelineConfig):
        warnings.append(
            "Pipeline config not available in runtime context; "
            "cannot build STFT matrix."
        )
        return None, None
    if not isinstance(cache_dir, str):
        warnings.append(
            "Cache directory not available in runtime context; "
            "cannot build STFT matrix."
        )
        return None, None
    if float(cfg_obj.pipeline.reference_pressure_pa) <= 0:
        warnings.append(
            f"Invalid reference pressure "
            f"{float(cfg_obj.pipeline.reference_pressure_pa)} Pa; must be "
            f"> 0. Cannot compute STFT-derived SPL values."
        )
        return None, None
    return cfg_obj, cache_dir


def _frames_to_db_df(
    freqs_hz: np.ndarray,
    times: pd.DatetimeIndex,
    power: np.ndarray,
    ref_pressure: float,
) -> pd.DataFrame:
    """Convert one native power window to a dB DataFrame, using the
    exact formula and column naming of the former build_stft_matrix."""
    safe_power = np.maximum(
        power.astype(np.float32), np.finfo(np.float32).tiny,
    )
    power_db = (
        10.0 * np.log10(safe_power / np.float32(ref_pressure ** 2))
    ).astype(np.float32)
    cols = [f"{float(f):.2f}Hz" for f in freqs_hz]
    df = pd.DataFrame(power_db.T, index=times, columns=cols)
    df.index.name = "datetime"
    return df


def iter_stft_windows(
    runtime_context: dict,
    *,
    time_chunk: str | None,
    time_bins: int | None = 12000,
    channel: int = 0,
    warnings: list[str] | None = None,
) -> Iterator[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    """
    Yield ``(window_start, window_end, df)`` per ``time_chunk`` window on
    the global downsampling grid (§8).

    Each ``df`` is a dB DataFrame (DatetimeIndex × ``'<freq>Hz'`` columns)
    bit-identical to slicing the legacy globally-downsampled
    ``build_stft_matrix`` output by ``time_chunk`` — but only the
    window's native frames are read, so per-window memory is bounded by
    the window, never the deployment. With ``time_chunk=None`` a single
    window spanning the full extent is produced.

    Yields nothing (and appends a warning) when no STFT shards are
    available for the channel — mirroring the old "STFT not available"
    path that consumers already handle.
    """
    warnings = warnings if warnings is not None else []
    cfg_obj, cache_dir = _resolve_context(runtime_context, warnings)
    if cfg_obj is None:
        return
    ref_pressure = float(cfg_obj.pipeline.reference_pressure_pa)

    store = StftStore(cache_dir, channel=channel)
    grid = store.render_grid(time_bins)
    if len(grid.labels) == 0:
        warnings.append(
            "No STFT frames available from the shard store; STFT data "
            "could not be assembled."
        )
        return

    if time_chunk is None:
        label_groups: list[pd.DatetimeIndex] = [grid.labels]
    else:
        marker = pd.Series(0, index=grid.labels)
        label_groups = [ #type: ignore
            grp.index
            for _, grp in marker.resample(time_chunk)
            if len(grp) > 0
        ]

    for labels in label_groups:
        # Read every native frame whose global bin falls in this window.
        if grid.do_downsample:
            read_t0, read_t1 = labels[0], labels[-1] + grid.step
        else:
            read_t0, read_t1 = labels[0], labels[-1]

        freqs_hz, times, power = store.read(read_t0, read_t1)
        df = _frames_to_db_df(freqs_hz, times, power, ref_pressure)

        if grid.do_downsample:
            # dB-space mean per global bin, anchored to the global
            # origin so labels are a subset of the global grid; drop
            # empty bins, then keep exactly this window's labels.
            df = df.resample(grid.step, origin=grid.origin).mean()
            df = df.dropna(how="all")
        df = df.loc[df.index.isin(labels)]

        yield labels[0], labels[-1], df


def build_stft_matrix(
    runtime_context: dict,
    *,
    time_bins: int | None = 12000,
) -> tuple[pd.DataFrame | None, list[str]]:
    """
    Whole-extent convenience: the single deployment-wide STFT matrix
    (DatetimeIndex × ``'<freq>Hz'`` dB columns) assembled from the shard
    store on the global grid (channel 0). Store-backed replacement for
    the former npz-based assembly.

    Returns ``(matrix, warnings)``; ``matrix`` is ``None`` when no STFT
    shards are available.

    NOTE: this materialises the whole matrix. Consumers that render per
    ``time_chunk`` should iterate ``iter_stft_windows`` to stay
    memory-bounded.
    """
    warnings: list[str] = []
    windows = list(
        iter_stft_windows(
            runtime_context,
            time_chunk=None,
            time_bins=time_bins,
            channel=0,
            warnings=warnings,
        )
    )
    if not windows or windows[0][2].empty:
        if not warnings:
            warnings.append("No valid STFT frames could be loaded.")
        return None, warnings
    return windows[0][2], warnings

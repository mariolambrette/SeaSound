"""
Tests for build_stft_matrix (now store-backed; refactor plan §8/Stage 4).

build_stft_matrix is the whole-extent convenience over the STFT shard
store. It reads shards from ``cache_dir/stft`` rather than computing npz
on the fly, so these tests seed a shard directly and assert the
assembled dB matrix. The deeper pixel-level equivalence with the legacy
global-resample-then-slice path is gated in test_streaming_stft_render.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest #pylint: disable=unused-import

from seasound.core.stft import build_stft_matrix
from seasound.loader.stft_store import (
    StftShardWriter, stft_dir_for, shard_name,
)

SR, WIN, HOP, NFREQ = 96000, 2048, 2048, 64
FREQS = np.linspace(10.0, 50000.0, NFREQ)


def _seed_shard(cache_dir, basename, dt_start, n_frames, *, seed=0):
    """Write one complete shard of random positive power into the store."""
    rng = np.random.default_rng(seed)
    power = rng.uniform(1e-9, 5.0, (NFREQ, n_frames)).astype(np.float32)
    path = os.path.join(stft_dir_for(cache_dir), shard_name(basename, 0))
    w = StftShardWriter(
        path, FREQS, SR, HOP, WIN, dt_start, channel=0, serial="9999",
    )
    w.append(power)
    w.finalise()
    return power


def test_missing_pipeline_config_returns_none(tmp_path):
    """No pipeline_config → returns None + warning."""
    ctx = {"cache_dir": str(tmp_path)}
    matrix, warnings = build_stft_matrix(ctx)
    assert matrix is None
    assert any("Pipeline config" in w for w in warnings)


def test_missing_cache_dir_returns_none(test_config):
    """No cache_dir → returns None + warning."""
    ctx = {"pipeline_config": test_config}
    matrix, warnings = build_stft_matrix(ctx)
    assert matrix is None
    assert any("Cache directory" in w for w in warnings)


def test_no_shards_returns_none(test_config, tmp_path):
    """Empty store → returns None + STFT-unavailable warning."""
    ctx = {"pipeline_config": test_config, "cache_dir": str(tmp_path)}
    matrix, warnings = build_stft_matrix(ctx)
    assert matrix is None
    assert any("STFT" in w for w in warnings)


def test_happy_path_returns_dataframe(test_config, tmp_path):
    """Seeded shard → DataFrame with DatetimeIndex and Hz columns."""
    _seed_shard(tmp_path, "9999.260101120000.wav", datetime(2026, 1, 1, 12, 0, 0), 2000)
    ctx = {"pipeline_config": test_config, "cache_dir": str(tmp_path)}
    matrix, _ = build_stft_matrix(ctx, time_bins=1000)
    assert matrix is not None
    assert isinstance(matrix.index, pd.DatetimeIndex)
    assert all(c.endswith("Hz") for c in matrix.columns)
    assert len(matrix.columns) == NFREQ


def test_time_bins_downsamples(test_config, tmp_path):
    """Smaller time_bins produces no more rows than no downsampling."""
    _seed_shard(tmp_path, "9999.260101120000.wav", datetime(2026, 1, 1, 12, 0, 0), 4000)
    ctx = {"pipeline_config": test_config, "cache_dir": str(tmp_path)}
    big, _ = build_stft_matrix(ctx, time_bins=None)
    small, _ = build_stft_matrix(ctx, time_bins=10)
    assert big is not None and small is not None
    assert len(small) <= len(big)

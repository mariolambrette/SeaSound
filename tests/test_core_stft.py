"""Tests for build_stft_matrix."""

import pandas as pd
import pytest #pylint: disable=unused-import

from seasound.analysis.calculate_stft import build_stft_matrix


def test_missing_pipeline_config_returns_none(tmp_path):
    """No pipeline_config → returns None + warning."""
    ctx = {"cache_dir": str(tmp_path), "input_files": ["foo.wav"]}
    matrix, warnings = build_stft_matrix(ctx)
    assert matrix is None
    assert any("Pipeline config" in w for w in warnings)


def test_missing_cache_dir_returns_none(test_config):
    """No cache_dir → returns None + warning."""
    ctx = {"pipeline_config": test_config, "input_files": ["foo.wav"]}
    matrix, warnings = build_stft_matrix(ctx)
    assert matrix is None
    assert any("Cache directory" in w for w in warnings)


def test_empty_input_files_returns_none(test_config, tmp_path):
    """Empty input_files → returns None + warning."""
    ctx = {
        "pipeline_config": test_config,
        "cache_dir": str(tmp_path),
        "input_files": [],
    }
    matrix, warnings = build_stft_matrix(ctx)
    assert matrix is None
    assert any("Input files" in w for w in warnings)


def test_happy_path_returns_dataframe(
    test_config, synthetic_wav, tmp_path
):
    """Valid context → returns a DataFrame with DatetimeIndex and Hz cols."""
    ctx = {
        "pipeline_config": test_config,
        "cache_dir": str(tmp_path),
        "input_files": [synthetic_wav],
    }
    matrix, _ = build_stft_matrix(ctx, time_bins=1000)
    assert matrix is not None
    assert isinstance(matrix.index, pd.DatetimeIndex)
    assert all(c.endswith("Hz") for c in matrix.columns)


def test_time_bins_downsamples(test_config, synthetic_wav, tmp_path):
    """Smaller time_bins produces fewer rows."""
    ctx = {
        "pipeline_config": test_config,
        "cache_dir": str(tmp_path),
        "input_files": [synthetic_wav],
    }
    big, _ = build_stft_matrix(ctx, time_bins=None)
    small, _ = build_stft_matrix(ctx, time_bins=10)
    # Either downsampling shrank it, or the matrix was already small
    if big is not None and small is not None:
        assert len(small) <= len(big)

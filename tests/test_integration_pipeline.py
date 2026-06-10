"""Integration tests for loading pipeline contracts."""

import os
import json
from datetime import timedelta

import pandas as pd
import pytest #pylint: disable=unused-import

from seasound.core.pipeline import run_loading, get_clip_bounds, run_pipeline
from seasound.core.config import( #pylint: disable=unused-import
    PipelineConfig,
    InputConfig,
    DeploymentConfig,
    DeploymentBufferConfig
)
from seasound.loader.cache import is_cached


def test_end_to_end_ingestion_to_cache(test_config, synthetic_wav):
    """..."""
    # Arrange
    test_config.input.path = os.path.dirname(synthetic_wav)
    test_config.pipeline.resume = False

    # Act
    matrix = run_loading(test_config)

    # Assert
    assert isinstance(matrix, pd.DataFrame)
    assert len(matrix) > 0
    cache_dir = os.path.join(test_config.output.directory, "cache")
    assert is_cached(synthetic_wav, 0, cache_dir)


def test_resume_partial_multichannel_processes_missing_channel(test_config, synthetic_stereo_wav):
    """..."""
    # Arrange: auto channel strategy should create two channel caches
    test_config.input.path = os.path.dirname(synthetic_stereo_wav)
    test_config.input.channel_strategy = "auto"
    test_config.pipeline.resume = False

    # Initial run creates ch0 and ch1
    run_loading(test_config)
    cache_dir = os.path.join(test_config.output.directory, "cache")
    ch0 = is_cached(synthetic_stereo_wav, 0, cache_dir)
    ch1 = is_cached(synthetic_stereo_wav, 1, cache_dir)
    assert ch0 and ch1

    # Remove one channel cache to simulate partial cache
    base = os.path.splitext(os.path.basename(synthetic_stereo_wav))[0]
    ch1_path = os.path.join(cache_dir, f"{base}_ch1.parquet")
    os.remove(ch1_path)
    assert not os.path.exists(ch1_path)

    # Resume run should recreate missing channel
    test_config.pipeline.resume = True
    run_loading(test_config)
    assert os.path.exists(ch1_path)


def test_clip_none_uses_full_extent(test_config, synthetic_base_matrix):
    """..."""
    test_config.deployment = DeploymentConfig(
        enabled=True,
        clip_method="none",
        buffer_hours=DeploymentBufferConfig(start=0.0, end=0.0),
    )

    start, end = get_clip_bounds(test_config, synthetic_base_matrix)
    assert start == synthetic_base_matrix.index.min()
    assert end == synthetic_base_matrix.index.max()


def test_clip_manual_applies_shared_buffer(test_config, synthetic_base_matrix):
    """..."""
    raw_start = synthetic_base_matrix.index.min()
    raw_end = synthetic_base_matrix.index.max()

    test_config.deployment = DeploymentConfig(
        enabled=True,
        clip_method="manual",
        start_utc=str(raw_start),
        end_utc=str(raw_end),
        buffer_hours=DeploymentBufferConfig(start=0.1, end=0.2),
    )

    start, end = get_clip_bounds(test_config, synthetic_base_matrix)
    assert start == raw_start + timedelta(hours=0.1)
    assert end == raw_end - timedelta(hours=0.2)


def test_manifest_written_after_pipeline_run(test_config, synthetic_wav):
    """..."""
    test_config.input.path = os.path.dirname(synthetic_wav)
    test_config.load_only = True

    run_pipeline(test_config)

    manifest = os.path.join(test_config.output.directory, "run_manifest.json")
    assert os.path.isfile(manifest)
    with open(manifest, "r") as f: #pylint: disable=unspecified-encoding
        payload = json.load(f)
    assert "seasound_version" in payload
    assert "config_summary" in payload

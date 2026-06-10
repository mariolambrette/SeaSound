"""
§9 test 8 — resume correctness (refactor plan §12).

The resume rule is conjunctive and product-aware over this run's
resolved producer set: a file is skipped only when every required
product is complete for every channel, and a partially cached file is
reprocessed for the missing product alone. Cases covered:

(a) a fully cached file (all required products) is skipped;
(b) a base-only cached file with STFT newly required is reprocessed for
    STFT only — the base parquet is untouched and still reaches the merge;
(c) a shard lacking the ``complete`` flag is treated as absent and rebuilt;
(d) a deleted manifest is rebuilt from shard attributes.
"""

import copy
import os
import time

import zarr

from seasound.core.pipeline import (
    _cached_products,
    _is_fully_cached,
    _process_one_file_streaming,
)
from seasound.core.substrates import BASE_MATRIX, STFT, subtract_cached
from seasound.loader import stft_store
from seasound.loader.cache import base_matrix_cache_path
from seasound.loader.calibration import load_calibration
from seasound.loader.filename_parsers import get_parser
from seasound.loader.reader import probe_output_channels


def _streaming_stft_cfg(test_config):
    cfg = copy.deepcopy(test_config)
    cfg.pipeline.stft_enabled = True  # resolver force-on → resolved = {base, stft}
    return cfg


def _seed(wav_path, cfg, cache_dir, to_produce):
    cal_df = load_calibration(cfg.calibration)
    return _process_one_file_streaming(
        wav_path, cfg, cal_df, cache_dir, to_produce, get_parser(cfg.input)
    )


def test_fully_cached_file_is_skipped(synthetic_wav, test_config, tmp_path):
    cfg = _streaming_stft_cfg(test_config)
    cache_dir = str(tmp_path)
    _seed(synthetic_wav, cfg, cache_dir, {BASE_MATRIX, STFT})

    resolved = {BASE_MATRIX, STFT}
    assert _cached_products(synthetic_wav, cfg, cache_dir) == {BASE_MATRIX, STFT}
    assert _is_fully_cached(synthetic_wav, cfg, cache_dir, resolved) is True


def test_base_only_cached_reprocesses_stft_only(synthetic_wav, test_config, tmp_path):
    cfg = _streaming_stft_cfg(test_config)
    cache_dir = str(tmp_path)
    _seed(synthetic_wav, cfg, cache_dir, {BASE_MATRIX})  # base only

    resolved = {BASE_MATRIX, STFT}
    assert _cached_products(synthetic_wav, cfg, cache_dir) == {BASE_MATRIX}
    assert _is_fully_cached(synthetic_wav, cfg, cache_dir, resolved) is False
    assert subtract_cached(
        resolved, _cached_products(synthetic_wav, cfg, cache_dir)
    ) == {STFT}

    channels = probe_output_channels(synthetic_wav, cfg.input)
    base_paths = [
        base_matrix_cache_path(synthetic_wav, ch, cache_dir) for ch in channels
    ]
    mtimes = {p: os.path.getmtime(p) for p in base_paths}

    time.sleep(0.05)  # any rewrite would advance mtime past this
    artifacts = _seed(synthetic_wav, cfg, cache_dir, {STFT})  # STFT only

    # STFT now present; base parquet untouched; merge still sees the base.
    assert _cached_products(synthetic_wav, cfg, cache_dir) == {BASE_MATRIX, STFT}
    for p in base_paths:
        assert os.path.getmtime(p) == mtimes[p], "base parquet was rewritten"
    assert artifacts and all(not a.base_matrix.empty for a in artifacts)


def test_incomplete_shard_treated_as_absent_and_rebuilt(
    synthetic_wav, test_config, tmp_path
):
    cfg = _streaming_stft_cfg(test_config)
    cache_dir = str(tmp_path)
    _seed(synthetic_wav, cfg, cache_dir, {BASE_MATRIX, STFT})
    assert STFT in _cached_products(synthetic_wav, cfg, cache_dir)

    # Simulate a crash artifact: clear the complete flag on every shard.
    stft_dir = stft_store.stft_dir_for(cache_dir)
    channels = probe_output_channels(synthetic_wav, cfg.input)
    for ch in channels:
        path = os.path.join(stft_dir, stft_store.shard_name(synthetic_wav, ch))
        group = zarr.open_group(path, mode="a")
        group.attrs["complete"] = False

    # The flagless shard counts as absent → file is not fully cached.
    assert STFT not in _cached_products(synthetic_wav, cfg, cache_dir)
    assert _is_fully_cached(synthetic_wav, cfg, cache_dir, {BASE_MATRIX, STFT}) is False

    # Reprocessing for STFT rebuilds a complete shard (writer truncates).
    _seed(synthetic_wav, cfg, cache_dir, {STFT})
    assert STFT in _cached_products(synthetic_wav, cfg, cache_dir)


def test_manifest_rebuilds_from_shard_attributes(
    synthetic_wav, test_config, tmp_path
):
    cfg = _streaming_stft_cfg(test_config)
    cache_dir = str(tmp_path)
    _seed(synthetic_wav, cfg, cache_dir, {BASE_MATRIX, STFT})

    stft_dir = stft_store.stft_dir_for(cache_dir)
    rows_before = stft_store.rebuild_manifest_rows(stft_dir)
    assert rows_before
    manifest_path = stft_store.write_manifest(rows_before, stft_dir)
    assert os.path.isfile(manifest_path)

    os.remove(manifest_path)
    rows_after = stft_store.rebuild_manifest_rows(stft_dir)
    assert rows_after == rows_before

"""
tests/golden.py

Streaming-path reference helpers and bit-identity assertions for the suite.

The legacy non-streaming path has been removed, so the live legacy oracle is
gone; the absolute-value baseline now lives in committed snapshots (see
test_golden_baseline.py / tests/golden_io.py). What remains here is the
*streamed* per-file reference used to seed and check those snapshots, plus the
bit-identity assertion helpers shared across the suite:

- ``streamed_base_matrix_artifacts`` / ``streamed_stft_entries``: run the real
  streaming pipeline path on one file and return its per-channel output.
- ``assert_*_identical``: bit-identity comparisons (np.array_equal, NaN-aware)
  over base matrices and STFT entries.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from seasound.core.config import PipelineConfig, ProcessingConfig
from seasound.loader.calibration import load_calibration


# ---------------------------------------------------------------------------
# Streaming reference functions
# ---------------------------------------------------------------------------


def streamed_base_matrix_artifacts(
    wav_path: str,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """
    Run the real streaming pipeline path (``_process_one_file_streaming``) on
    one file and return per-channel artifacts: dicts with ``channel``,
    ``serial``, ``datetime_start``, ``calibrated``, ``base_matrix``.

    Producing the base matrix also caches it (the resolver path has no
    produce-without-cache mode), so a throwaway temp cache absorbs the write.
    """
    import tempfile

    # Imported lazily so golden.py stays importable without the pipeline's
    # orchestration dependencies.
    from seasound.core.pipeline import _process_one_file_streaming
    from seasound.core.substrates import BASE_MATRIX

    cfg = copy.deepcopy(config)
    cal_df = load_calibration(cfg.calibration)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        artifacts = _process_one_file_streaming(
            wav_path, cfg, cal_df, cache_dir, {BASE_MATRIX}
        )
        return [
            {
                "channel": a.channel,
                "serial": a.serial,
                "datetime_start": a.datetime_start,
                "calibrated": a.calibrated,
                "base_matrix": a.base_matrix,
            }
            for a in artifacts
        ]


def streamed_stft_entries(
    wav_path: str,
    config: PipelineConfig,
    block_seconds: int | None = None,
) -> list[dict[str, Any]]:
    """
    Run the real streaming path with STFT shard production on, then read the
    shard store back into per-channel entries: dicts with ``channel``,
    ``serial``, ``datetime_start``, ``freqs_hz``, ``times_s``, ``power``.

    ``times_s`` is reconstructed via the store's single D8 implementation
    (``frame_times_s``), so this exercises the whole chain: carry-buffered
    compute → shard write → windowed read → timestamp convention.
    """
    import tempfile

    from seasound.core.pipeline import _process_one_file_streaming
    from seasound.core.substrates import BASE_MATRIX, STFT
    from seasound.loader.stft_store import StftStore, frame_times_s

    cfg = copy.deepcopy(config)
    if block_seconds is not None:
        cfg.pipeline.streaming_block_seconds = block_seconds
    cal_df = load_calibration(cfg.calibration)

    entries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as cache_dir:
        artifacts = _process_one_file_streaming(
            wav_path, cfg, cal_df, cache_dir, {BASE_MATRIX, STFT}
        )
        for a in artifacts:
            store = StftStore(cache_dir, channel=a.channel)
            freqs_hz, _times, power = store.read()
            shard = store.shards.iloc[0]
            entries.append({
                "channel": a.channel,
                "serial": a.serial,
                "datetime_start": a.datetime_start,
                "freqs_hz": freqs_hz,
                "times_s": frame_times_s(
                    int(shard.n_frames), int(shard.win),
                    int(shard.hop), int(shard.sample_rate),
                ),
                "power": power,
            })
    return entries


def base_matrix_from_array(
    audio_pa: np.ndarray,
    sample_rate: int,
    config: ProcessingConfig,
) -> pd.DataFrame:
    """
    Whole-array base matrix via the streaming accumulator — the only
    base-matrix API after the Stage-6 cleanup. Pushes the whole signal
    as a single block and finalises (integer-second index, no datetime),
    so DSP-correctness and numeric-knob tests that hold a raw Pascal
    array can assert against it directly. Because noverlap=0 over
    independent 1-second bins, one push is bit-identical to per-block
    streaming.
    """
    from seasound.loader.base_matrix import BaseMatrixAccumulator

    bin_samples = int(config.base_resolution_s * sample_rate)
    n_bins = len(audio_pa) // bin_samples if bin_samples else 0
    acc = BaseMatrixAccumulator(sample_rate, n_bins, config)
    if n_bins > 0:
        acc.push(np.asarray(audio_pa)[: n_bins * bin_samples])
    return acc.finalise()


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_base_matrix_identical(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    *,
    context: str = "",
) -> None:
    """
    Assert two base matrices are bit-identical: same columns, same index, same
    values (NaN positions included).

    ``np.testing.assert_array_equal`` treats aligned NaNs as equal and reports
    the first mismatching positions on failure, which is what we want for
    unreachable-band columns under the "nan" strategy.
    """
    prefix = f"[{context}] " if context else ""
    assert list(expected.columns) == list(actual.columns), (
        f"{prefix}Base matrix columns differ"
    )
    assert expected.index.equals(actual.index), (
        f"{prefix}Base matrix index differs: "
        f"expected {expected.index[:3]}..., got {actual.index[:3]}..."
    )
    np.testing.assert_array_equal(
        expected.to_numpy(),
        actual.to_numpy(),
        err_msg=f"{prefix}Base matrix values differ (bit-identity required)",
    )


def assert_artifacts_identical(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    context: str = "",
) -> None:
    """
    Assert two per-file artifact lists (as returned by
    ``streamed_base_matrix_artifacts`` or a snapshot loader with the same
    shape) are identical: metadata and matrices.
    """
    prefix = f"[{context}] " if context else ""
    assert len(expected) == len(actual), (
        f"{prefix}Artifact count differs: {len(expected)} vs {len(actual)}"
    )
    for exp, act in zip(expected, actual):
        for key in ("channel", "serial", "datetime_start", "calibrated"):
            assert exp[key] == act[key], (
                f"{prefix}Artifact metadata '{key}' differs: "
                f"{exp[key]!r} vs {act[key]!r}"
            )
        assert_base_matrix_identical(
            exp["base_matrix"],
            act["base_matrix"],
            context=f"{context} channel {exp['channel']}".strip(),
        )


def assert_stft_entries_identical(
    expected: list[dict[str, Any]],
    actual: list[dict[str, Any]],
    *,
    context: str = "",
) -> None:
    """
    Assert two per-file STFT entry lists are identical frame-for-frame: same
    channels, same frequency axis, same frame times, same power values, no
    missing or duplicated frames.
    """
    prefix = f"[{context}] " if context else ""
    assert len(expected) == len(actual), (
        f"{prefix}STFT entry count differs: {len(expected)} vs {len(actual)}"
    )
    for exp, act in zip(expected, actual):
        for key in ("channel", "serial", "datetime_start"):
            assert exp[key] == act[key], (
                f"{prefix}STFT metadata '{key}' differs: "
                f"{exp[key]!r} vs {act[key]!r}"
            )
        np.testing.assert_array_equal(
            np.asarray(exp["freqs_hz"]), np.asarray(act["freqs_hz"]),
            err_msg=f"{prefix}STFT frequency axis differs",
        )
        exp_times = np.asarray(exp["times_s"])
        act_times = np.asarray(act["times_s"])
        assert exp_times.shape == act_times.shape, (
            f"{prefix}STFT frame count differs: "
            f"{exp_times.shape} vs {act_times.shape} "
            f"(missing or duplicated frames)"
        )
        np.testing.assert_array_equal(
            exp_times, act_times,
            err_msg=f"{prefix}STFT frame times differ",
        )
        np.testing.assert_array_equal(
            np.asarray(exp["power"]), np.asarray(act["power"]),
            err_msg=f"{prefix}STFT power values differ",
        )

"""
tests/golden.py

Golden-baseline helpers for the streaming refactor.

The float32 codebase at the branch point (tag: golden-float32-baseline)
is the reference implementation for the whole refactor. These helpers
expose its per-file outputs through two reference functions, plus
assertion helpers that gate every later stage:

- ``legacy_base_matrix_artifacts``: the per-file base-matrix output,
  exactly as ``_process_one_file`` produces it (read -> calibrate ->
  compute -> DatetimeIndex). Stage 2's ``BaseMatrixAccumulator`` must be
  *bit-identical* to this (refactor plan section 9, tests 1-2).
- ``legacy_stft_entries``: the per-file STFT output of
  ``get_stft_for_file`` with caching forced off. Stage 3's
  ``StftAccumulator`` must match it frame-for-frame (plan tests 3-4).

The comparisons stay live (both sides computed at test time) until the
legacy paths are deleted in Stage 6, at which point the reference side
is replaced by committed snapshot fixtures generated from the last
dual-path commit.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np
import pandas as pd

from seasound.core.config import PipelineConfig
from seasound.loader.reader import read_audio
from seasound.loader.filename_parsers import get_parser
from seasound.loader.calibration import load_calibration, apply_calibration
from seasound.loader.base_matrix import compute_base_matrix
from seasound.analysis.calculate_stft import get_stft_for_file


# ---------------------------------------------------------------------------
# Reference functions (legacy paths)
# ---------------------------------------------------------------------------


def legacy_base_matrix_artifacts(
    wav_path: str,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """
    Run one file through the legacy per-file base-matrix path.

    Mirrors ``seasound.core.pipeline._process_one_file`` step for step
    (read -> calibrate -> compute -> DatetimeIndex conversion), without
    cache writes. Note the DatetimeIndex conversion reproduces the
    pipeline's hardcoded ``freq="1s"`` — faithful to current behaviour
    even though it ignores ``base_resolution_s``.

    Parameters
    ----------
    wav_path : str
        Path to the WAV file.
    config : PipelineConfig
        Full pipeline configuration.

    Returns
    -------
    list[dict[str, Any]]
        One entry per output channel with keys: ``channel``, ``serial``,
        ``datetime_start``, ``calibrated``, ``base_matrix``.
    """
    parser = get_parser(config.input)
    segments = read_audio(wav_path, config.input, parser=parser)
    cal_df = load_calibration(config.calibration)

    artifacts: list[dict[str, Any]] = []
    for segment in segments:
        audio_pa, calibrated = apply_calibration(
            segment, cal_df, config.calibration
        )
        matrix = compute_base_matrix(audio_pa, segment.sample_rate, config.pipeline)

        if (
            segment.datetime_start is not None
            and not isinstance(matrix.index, pd.DatetimeIndex)
        ):
            dt_index = pd.date_range(
                start=segment.datetime_start,
                periods=len(matrix),
                freq="1s",
            )
            matrix = matrix.copy()
            matrix.index = dt_index
            matrix.index.name = "datetime"

        artifacts.append({
            "channel": segment.channel,
            "serial": segment.serial,
            "datetime_start": segment.datetime_start,
            "calibrated": calibrated,
            "base_matrix": matrix,
        })

    return artifacts


def legacy_stft_entries(
    wav_path: str,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """
    Run one file through the legacy per-file STFT path, bypassing the
    npz cache so the result is always a fresh computation.

    Parameters
    ----------
    wav_path : str
        Path to the WAV file.
    config : PipelineConfig
        Full pipeline configuration. Not mutated; a deep copy with
        ``stft_cache_enabled = False`` is used internally.

    Returns
    -------
    list[dict[str, Any]]
        ``get_stft_for_file`` entries: one per channel with keys
        ``channel``, ``serial``, ``datetime_start``, ``freqs_hz``,
        ``times_s``, ``power``.
    """
    cfg = copy.deepcopy(config)
    cfg.pipeline.stft_cache_enabled = False
    return get_stft_for_file(wav_path, cfg, cache_dir="")


def streamed_base_matrix_artifacts(
    wav_path: str,
    config: PipelineConfig,
) -> list[dict[str, Any]]:
    """
    Candidate function for the Stage 2 gates: run the real streaming
    pipeline path (_process_one_file_streaming) on one file with cache
    writes disabled, returning artifacts in the same dict shape as
    legacy_base_matrix_artifacts.
    """
    # Imported lazily so golden.py stays importable without the
    # pipeline's orchestration dependencies.
    from seasound.core.pipeline import _process_one_file_streaming

    cfg = copy.deepcopy(config)
    cfg.pipeline.streaming_enabled = True
    cfg.pipeline.cache_base_matrix = False
    cal_df = load_calibration(cfg.calibration)

    artifacts = _process_one_file_streaming(wav_path, cfg, cal_df, cache_dir="")
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
    Assert two base matrices are bit-identical: same columns, same
    index, same values (NaN positions included).

    ``np.testing.assert_array_equal`` treats aligned NaNs as equal and
    reports the first mismatching positions on failure, which is what we
    want for unreachable-band columns under the "nan" strategy.
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
    ``legacy_base_matrix_artifacts`` or a candidate implementation with
    the same shape) are identical: metadata and matrices.
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
    Assert two per-file STFT entry lists are identical frame-for-frame:
    same channels, same frequency axis, same frame times, same power
    values, no missing or duplicated frames.
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


def assert_candidate_matches_legacy_base_matrix(
    candidate_fn: Callable[[str, PipelineConfig], list[dict[str, Any]]],
    wav_path: str,
    config: PipelineConfig,
    *,
    context: str = "",
) -> None:
    """
    Plug-in gate for later stages: run ``candidate_fn`` (e.g. the
    streamed Stage 2 path) and the legacy path on the same file and
    config, and require identical artifacts.
    """
    expected = legacy_base_matrix_artifacts(wav_path, config)
    actual = candidate_fn(wav_path, config)
    assert_artifacts_identical(expected, actual, context=context)


def assert_candidate_matches_legacy_stft(
    candidate_fn: Callable[[str, PipelineConfig], list[dict[str, Any]]],
    wav_path: str,
    config: PipelineConfig,
    *,
    context: str = "",
) -> None:
    """
    Plug-in gate for later stages: run ``candidate_fn`` (e.g. the
    streamed Stage 3 ``StftAccumulator`` path) and the legacy per-file
    STFT on the same file and config, and require frame-for-frame
    identical entries.
    """
    expected = legacy_stft_entries(wav_path, config)
    actual = candidate_fn(wav_path, config)
    assert_stft_entries_identical(expected, actual, context=context)

"""
tests/test_streaming_invariance.py

Streaming-vs-streaming structural properties, with no legacy oracle, so they
survive the Stage-6 cleanup that deletes the legacy gates:

- base-matrix block independence (identical output across block_seconds);
- duty-cycle gap indexing across a merged pair, with no off-by-one;
- two per-file edge behaviours (uncalibrated fallback propagation; a file
  shorter than the start trim yields no rows) rescued from the deleted
  streamed-vs-legacy identity class.

(STFT seam independence is covered by
test_streaming_stft.py::TestAccumulatorContract, which pins the accumulator
against a single compute_stft_power over many push geometries.)
"""

import copy
from datetime import datetime

import pandas as pd
import pytest

from seasound.core.pipeline import _merge_base_matrices
from tests.conftest import _write_soundtrap_wav
from tests.golden import (
    assert_base_matrix_identical,
    streamed_base_matrix_artifacts,
)


class TestBlockIndependence:
    """Block boundaries must not change a single bit (plan D4): the streamed
    base matrix is identical across block_seconds. Reference block is 300 s
    (one block over the whole file); the others leave a trailing partial
    block on a 67 s file (67 = 9·7+4 = 2·30+7 = 60+7)."""

    @pytest.mark.parametrize("block_seconds", [7, 30, 60])
    def test_block_size_matches_single_block_reference(
        self, tmp_path, golden_config, block_seconds
    ):
        wav = _write_soundtrap_wav(
            tmp_path / "invariance", "9999",
            datetime(2026, 1, 1, 12, 0, 0), 67.0, 8000, seed=611,
        )

        ref_cfg = copy.deepcopy(golden_config)
        ref_cfg.pipeline.streaming_block_seconds = 300
        cfg = copy.deepcopy(golden_config)
        cfg.pipeline.streaming_block_seconds = block_seconds

        reference = streamed_base_matrix_artifacts(wav, ref_cfg)
        candidate = streamed_base_matrix_artifacts(wav, cfg)

        assert_base_matrix_identical(
            reference[0]["base_matrix"],
            candidate[0]["base_matrix"],
            context=f"block_seconds {block_seconds} vs 300",
        )


class TestGapIndexing:
    """Duty-cycle gap: two recordings 47 s apart must index correctly across
    the gap and merge with no off-by-one and no duplicates — verified on the
    streamed output directly."""

    def test_gap_merge_indexing(self, gapped_wav_pair, golden_config):
        _, (first, second) = gapped_wav_pair  # 13 s @12:00:00, 10 s @12:01:00

        matrices = [
            streamed_base_matrix_artifacts(p, golden_config)[0]["base_matrix"]
            for p in (first, second)
        ]

        # Per-file boundaries: no off-by-one at the gap.
        assert matrices[0].index[0] == pd.Timestamp("2026-01-01 12:00:00")
        assert matrices[0].index[-1] == pd.Timestamp("2026-01-01 12:00:12")
        assert matrices[1].index[0] == pd.Timestamp("2026-01-01 12:01:00")
        assert matrices[1].index[-1] == pd.Timestamp("2026-01-01 12:01:09")

        merged = _merge_base_matrices(matrices)
        # 13 + 10 disjoint rows across the gap.
        assert len(merged) == 23
        assert merged.index.is_monotonic_increasing
        assert not merged.index.has_duplicates


class TestStreamingEdges:
    """Per-file edge behaviours rescued from the deleted identity class —
    asserted structurally on the streamed output, no oracle needed."""

    def test_uncalibrated_serial_propagates(self, awkward_wav, golden_config):
        """An unknown serial under non-strict calibration propagates
        calibrated=False through to the artifact."""
        cfg = copy.deepcopy(golden_config)
        cfg.input.serial_override = "123456"  # not in the fixture table
        cfg.calibration.strict = False

        artifacts = streamed_base_matrix_artifacts(awkward_wav, cfg)
        assert artifacts[0]["calibrated"] is False

    def test_file_shorter_than_trim_yields_no_rows(self, tmp_path, test_config):
        """When the start trim exceeds the file duration, the streamed path
        produces no base-matrix rows (test_config trims 3 s; the file is 2 s)."""
        wav = _write_soundtrap_wav(
            tmp_path / "tiny", "9999",
            datetime(2026, 1, 1, 12, 0, 0), 2.0, 96000, seed=501,
        )
        artifacts = streamed_base_matrix_artifacts(wav, test_config)
        total_rows = sum(len(a["base_matrix"]) for a in artifacts)
        assert total_rows == 0, (
            "a file shorter than the start trim must yield no base-matrix rows"
        )

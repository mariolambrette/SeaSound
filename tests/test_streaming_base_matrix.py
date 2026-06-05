"""
tests/test_streaming_base_matrix.py

Stage 2 gates for the streaming base-matrix path (refactor plan §9
tests 1, 2, 9, plus reader/probe equivalence and accumulator guards).

Every identity assertion is bit-identity (np.array_equal via the
golden helpers): with noverlap=0 over whole-second bins, block
boundaries on bin boundaries must not change a single bit (plan D4).
"""

import copy
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

try:
    import soundfile as sf
except ImportError:
    sf = None

from seasound.core.config import validate
from seasound.core.exceptions import ConfigError
from seasound.core.pipeline import _merge_base_matrices, run_loading
from seasound.loader.base_matrix import BaseMatrixAccumulator
from seasound.loader.reader import (
    AudioBlockReader,
    probe_output_channels,
    read_audio,
)

from tests.conftest import _write_soundtrap_wav
from tests.golden import (
    assert_base_matrix_identical,
    assert_candidate_matches_legacy_base_matrix,
    legacy_base_matrix_artifacts,
    streamed_base_matrix_artifacts,
)


def _with_strategy(config, strategy, selected_channel=0):
    cfg = copy.deepcopy(config)
    cfg.input.channel_strategy = strategy
    cfg.input.selected_channel = selected_channel
    return cfg


class TestStreamedIdentity:
    """§9 test 1: streamed base matrix == legacy, bit for bit."""

    @pytest.mark.parametrize(
        "wav_fixture", ["synthetic_wav", "awkward_wav", "fractional_wav"]
    )
    def test_single_file_identity(self, wav_fixture, golden_config, request):
        wav = request.getfixturevalue(wav_fixture)
        assert_candidate_matches_legacy_base_matrix(
            streamed_base_matrix_artifacts, wav, golden_config,
            context=f"streamed {wav_fixture}",
        )

    def test_identity_under_default_start_trim(self, awkward_wav, test_config):
        """The seek-based trim must reproduce the slice-based trim,
        including the rounded-sample datetime shift."""
        assert_candidate_matches_legacy_base_matrix(
            streamed_base_matrix_artifacts, awkward_wav, test_config,
            context="streamed with 3s trim",
        )

    @pytest.mark.parametrize(
        "strategy,selected", [("mono", 0), ("select", 1), ("auto", 0)]
    )
    def test_stereo_strategies_identity(
        self, synthetic_stereo_wav, golden_config, strategy, selected
    ):
        cfg = _with_strategy(golden_config, strategy, selected)
        assert_candidate_matches_legacy_base_matrix(
            streamed_base_matrix_artifacts, synthetic_stereo_wav, cfg,
            context=f"streamed stereo {strategy}",
        )

    def test_uncalibrated_fallback_identity(self, awkward_wav, golden_config):
        cfg = copy.deepcopy(golden_config)
        cfg.input.serial_override = "123456"  # not in the fixture table
        cfg.calibration.strict = False
        assert_candidate_matches_legacy_base_matrix(
            streamed_base_matrix_artifacts, awkward_wav, cfg,
            context="streamed uncalibrated",
        )
        streamed = streamed_base_matrix_artifacts(awkward_wav, cfg)
        assert streamed[0]["calibrated"] is False

    def test_file_shorter_than_trim_identity(self, tmp_path, test_config):
        """Both paths must produce the same empty per-file frame when
        the start trim exceeds the file duration."""
        wav = _write_soundtrap_wav(
            tmp_path / "tiny", "9999",
            datetime(2026, 1, 1, 12, 0, 0), 2.0, 96000, seed=501,
        )
        assert_candidate_matches_legacy_base_matrix(
            streamed_base_matrix_artifacts, wav, test_config,
            context="streamed shorter-than-trim",
        )


class TestBlockBoundaryInvariance:
    """§9 test 2: several block_seconds values, identical output."""

    @pytest.mark.parametrize("block_seconds", [7, 30, 60, 300])
    def test_block_sizes_match_legacy(self, tmp_path, golden_config, block_seconds):
        # 67 s at 8 kHz: every tested block size leaves a remainder
        # block (67 = 9*7+4 = 2*30+7 = 60+7 < 300), so trailing-partial
        # -block handling is exercised at each size.
        wav = _write_soundtrap_wav(
            tmp_path / "invariance", "9999",
            datetime(2026, 1, 1, 12, 0, 0), 67.0, 8000, seed=502,
        )
        cfg = copy.deepcopy(golden_config)
        cfg.pipeline.streaming_block_seconds = block_seconds

        assert_candidate_matches_legacy_base_matrix(
            streamed_base_matrix_artifacts, wav, cfg,
            context=f"block_seconds={block_seconds}",
        )


class TestGapsAndMultiFile:
    """§9 test 9: non-contiguous recordings index correctly, and the
    merged deployment matrix is identical between paths."""

    @pytest.mark.parametrize(
        "pair_fixture",
        ["gapped_wav_pair", "contiguous_wav_pair", "overlapping_wav_pair"],
    )
    def test_per_file_identity_across_pairs(
        self, pair_fixture, golden_config, request
    ):
        _, paths = request.getfixturevalue(pair_fixture)
        for path in paths:
            assert_candidate_matches_legacy_base_matrix(
                streamed_base_matrix_artifacts, path, golden_config,
                context=f"{pair_fixture}: {path}",
            )

    def test_gap_indexing_and_merge(self, gapped_wav_pair, golden_config):
        _, (first, second) = gapped_wav_pair

        streamed = [
            streamed_base_matrix_artifacts(p, golden_config)[0]["base_matrix"]
            for p in (first, second)
        ]
        legacy = [
            legacy_base_matrix_artifacts(p, golden_config)[0]["base_matrix"]
            for p in (first, second)
        ]

        # Datetime indexing across the gap: no off-by-one.
        assert streamed[1].index[0] == pd.Timestamp("2026-01-01 12:01:00")
        assert streamed[0].index[-1] == pd.Timestamp("2026-01-01 12:00:12")

        assert_base_matrix_identical(
            _merge_base_matrices(legacy),
            _merge_base_matrices(streamed),
            context="merged gapped deployment",
        )

    def test_run_loading_end_to_end_identity(self, gapped_wav_pair, golden_config):
        """The full orchestrator (serial path, cache writes on) must
        return a bit-identical merged matrix with streaming on vs off."""
        directory, _ = gapped_wav_pair

        results = {}
        for streaming in (False, True):
            cfg = copy.deepcopy(golden_config)
            cfg.input.path = directory
            cfg.pipeline.streaming_enabled = streaming
            cfg.pipeline.resume = False
            cfg.pipeline.workers = 1
            cfg.output.directory = str(
                pd.io.common.os.path.join(directory, f"out_{streaming}") #type: ignore
            )
            results[streaming] = run_loading(cfg)

        assert_base_matrix_identical(
            results[False], results[True], context="run_loading dual path"
        )


class TestReaderEquivalence:
    """The block reader must decode exactly what sf.read decodes."""

    @pytest.mark.parametrize("trim_s", [0.0, 3.0])
    def test_blockwise_read_matches_whole_read(
        self, awkward_wav, golden_config, trim_s
    ):
        if sf is None:
            pytest.skip("soundfile not installed")

        cfg = copy.deepcopy(golden_config)
        cfg.input.per_file_trim_start_s = trim_s

        whole, sr = sf.read(awkward_wav, dtype="float32")
        n_trim = int(round(trim_s * sr)) if trim_s > 0 else 0
        whole = whole[n_trim:]
        n_bins = len(whole) // sr
        expected = whole[: n_bins * sr]

        chunks, t0s = [], []
        with AudioBlockReader(awkward_wav, cfg.input) as reader:
            assert reader.n_bins == n_bins
            for block, t0 in reader.blocks(7):
                chunks.append(block)
                t0s.append(t0)

        np.testing.assert_array_equal(np.concatenate(chunks), expected)
        # t0 sequence: file start (+trim) stepping by 7 s per full block
        assert t0s[0] == pd.Timestamp("2026-01-01 12:00:00") + pd.Timedelta(
            seconds=n_trim / sr
        )

    def test_probe_matches_read_audio_channels(
        self, awkward_wav, synthetic_stereo_wav, golden_config
    ):
        cases = [
            (awkward_wav, "mono", 0),
            (synthetic_stereo_wav, "mono", 0),
            (synthetic_stereo_wav, "select", 1),
            (synthetic_stereo_wav, "auto", 0),
        ]
        for wav, strategy, selected in cases:
            cfg = _with_strategy(golden_config, strategy, selected)
            expected = [
                seg.channel for seg in read_audio(wav, cfg.input)
            ]
            assert probe_output_channels(wav, cfg.input) == expected, (
                f"probe != read_audio for {strategy}"
            )


class TestAccumulatorGuards:
    """Defensive contract: silent short-fills and misordered blocks
    must raise, never finalise."""

    @staticmethod
    def _acc(n_bins=4, sr=8000):
        from seasound.core.config import ProcessingConfig
        return BaseMatrixAccumulator(sr, n_bins, ProcessingConfig()), sr

    def test_partial_bin_block_raises(self):
        acc, sr = self._acc()
        with pytest.raises(ValueError):
            acc.push(np.zeros(sr + 1, dtype=np.float32))

    def test_overrun_raises(self):
        acc, sr = self._acc(n_bins=1)
        with pytest.raises(ValueError):
            acc.push(np.zeros(2 * sr, dtype=np.float32))

    def test_short_finalise_raises(self):
        acc, sr = self._acc(n_bins=4)
        acc.push(np.zeros(2 * sr, dtype=np.float32))
        with pytest.raises(ValueError):
            acc.finalise()

    def test_out_of_order_t0_raises(self):
        acc, sr = self._acc(n_bins=4)
        start = datetime(2026, 1, 1, 12, 0, 0)
        acc.set_anchor(start)
        acc.push(np.zeros(2 * sr, dtype=np.float32), t0=start)
        with pytest.raises(ValueError):
            acc.push(np.zeros(2 * sr, dtype=np.float32), t0=start)  # repeated t0


class TestConfigValidation:

    @staticmethod
    def _raw(block_seconds, resolution=1):
        return {
            "analyse_only": True,  # skip path existence checks
            "pipeline": {
                "streaming_block_seconds": block_seconds,
                "base_resolution_s": resolution,
            },
        }

    def test_default_block_seconds_valid(self):
        cfg = validate(self._raw(60))
        assert cfg.pipeline.streaming_block_seconds == 60
        # Streaming is the default since the Stage 2 gates passed;
        # False remains available as the legacy escape hatch until
        # Stage 6 removes both.
        assert cfg.pipeline.streaming_enabled is True

    def test_indivisible_block_seconds_rejected(self):
        with pytest.raises(ConfigError):
            validate(self._raw(7, resolution=2))

    def test_nonpositive_block_seconds_rejected(self):
        with pytest.raises(ConfigError):
            validate(self._raw(0))

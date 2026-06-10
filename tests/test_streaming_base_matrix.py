"""
tests/test_streaming_base_matrix.py

Unit behaviours of the streaming base-matrix path that stand alone, without
a legacy comparison: the block reader decodes exactly what sf.read decodes,
the accumulator enforces its block contract, and config validation guards
streaming_block_seconds.

The streamed-vs-legacy identity gates that used to live here moved with the
Stage-6 cleanup: representative absolute values are pinned by
test_golden_baseline.py, and block independence / gap indexing by
test_streaming_invariance.py.
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
from seasound.loader.base_matrix import BaseMatrixAccumulator
from seasound.loader.reader import (
    AudioBlockReader,
    probe_output_channels,
    read_audio,
)


def _with_strategy(config, strategy, selected_channel=0):
    cfg = copy.deepcopy(config)
    cfg.input.channel_strategy = strategy
    cfg.input.selected_channel = selected_channel
    return cfg


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
    """Defensive contract: silent short-fills and misordered blocks must
    raise, never finalise."""

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

    def test_indivisible_block_seconds_rejected(self):
        with pytest.raises(ConfigError):
            validate(self._raw(7, resolution=2))

    def test_nonpositive_block_seconds_rejected(self):
        with pytest.raises(ConfigError):
            validate(self._raw(0))

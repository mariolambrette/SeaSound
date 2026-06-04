"""
tests/test_characterisation_legacy.py

Characterisation tests for the golden float32 baseline.

These tests pin the *current* behaviour of the legacy Stage-1 paths
before the streaming refactor touches them. They are the reference half
of the regression strategy in the refactor plan (section 9): every
later stage's identity gate compares a new implementation against the
behaviours pinned here.

Two tests encode assumptions stated in the plan rather than behaviour
verified from this module's source (they are marked in their
docstrings). If either fails against the real codebase, that is a
finding about the plan — amend the plan (D8 / section 10), not the test,
before any production code is written.
"""

import numpy as np
import pandas as pd
import pytest #pylint: disable=unused-import

from seasound.core.config import ProcessingConfig
from seasound.loader.base_matrix import compute_base_matrix
from seasound.loader.reader import read_audio
from seasound.loader.filename_parsers import get_parser
from seasound.loader.calibration import load_calibration, apply_calibration

from tests.golden import ( #pylint: disable=unused-import
    legacy_base_matrix_artifacts,
    legacy_stft_entries,
    assert_base_matrix_identical,
    assert_candidate_matches_legacy_base_matrix,
)


class TestBaseMatrixLegacyBehaviour:
    """Pin per-file base-matrix behaviour of the golden baseline."""

    def test_datetime_index_from_filename(self, awkward_wav, golden_config):
        """With trim disabled, the per-file matrix has one row per whole
        second, indexed from the filename datetime at 1 s spacing."""
        artifacts = legacy_base_matrix_artifacts(awkward_wav, golden_config)

        assert len(artifacts) == 1
        matrix = artifacts[0]["base_matrix"]

        assert isinstance(matrix.index, pd.DatetimeIndex)
        assert matrix.index.name == "datetime"
        assert len(matrix) == 13
        assert matrix.index[0] == pd.Timestamp("2026-01-01 12:00:00")
        assert matrix.index[-1] == pd.Timestamp("2026-01-01 12:00:12")
        steps = np.unique(np.diff(matrix.index.to_numpy()))
        assert steps.tolist() == [np.timedelta64(1, "s")]

    def test_trailing_partial_second_dropped(self, fractional_wav, golden_config):
        """A 13.5 s file yields exactly 13 rows: the trailing partial
        second is dropped at end-of-file. The streamed path must drop it
        in the same place — at end-of-file, never at a block edge."""
        artifacts = legacy_base_matrix_artifacts(fractional_wav, golden_config)
        matrix = artifacts[0]["base_matrix"]

        assert len(matrix) == 13
        assert matrix.index[-1] == pd.Timestamp("2026-01-01 12:00:12")

    def test_default_start_trim_shifts_output(self, awkward_wav, test_config):
        """Pin the per_file_trim_start_s=3.0 default: 3 s of audio are
        removed and the datetime index starts 3 s after the filename
        time. Streaming must reproduce this as a seek plus the same
        datetime shift."""
        artifacts = legacy_base_matrix_artifacts(awkward_wav, test_config)
        matrix = artifacts[0]["base_matrix"]

        assert len(matrix) == 10  # 13 s - 3 s trim
        assert matrix.index[0] == pd.Timestamp("2026-01-01 12:00:03")

    def test_golden_gate_plumbing(self, awkward_wav, golden_config):
        """Smoke-test the plug-in gate with the legacy path as its own
        candidate: the harness later stages rely on must pass trivially
        against itself."""
        assert_candidate_matches_legacy_base_matrix(
            legacy_base_matrix_artifacts,
            awkward_wav,
            golden_config,
            context="legacy-vs-legacy",
        )

    def test_bin_independence_across_splits(self):
        """Foundational invariant for the streaming refactor (plan D4):
        with noverlap=0 and whole-second bins, computing the base matrix
        on the whole array equals computing it on whole-second splits
        and concatenating the rows — including a split that does *not*
        coincide with the internal 300 s chunk boundary.

        If this fails, blocks are not independent and the bit-identity
        goal of Stage 2 is unachievable as designed.
        """
        sr = 8000
        duration_s = 615  # > 2 internal chunks of 300 s, not a multiple
        rng = np.random.default_rng(7)
        audio = (0.1 * rng.standard_normal(duration_s * sr)).astype(np.float32)

        config = ProcessingConfig()  # defaults; band set is irrelevant here

        full = compute_base_matrix(audio, sr, config)
        assert len(full) == duration_s

        for split_s in (300, 137):  # chunk-aligned and deliberately not
            head = compute_base_matrix(audio[: split_s * sr], sr, config)
            tail = compute_base_matrix(audio[split_s * sr:], sr, config)

            joined = np.vstack([head.to_numpy(), tail.to_numpy()])
            np.testing.assert_array_equal(
                full.to_numpy(),
                joined,
                err_msg=(
                    f"Splitting at {split_s}s changed base-matrix values — "
                    f"bins are not independent"
                ),
            )


class TestStftLegacyBehaviour:
    """Pin per-file STFT conventions of the golden baseline."""

    def test_window_centre_timestamps(self, awkward_wav, golden_config):
        """PLAN ASSUMPTION (D8): frame k is stamped at window-centre time
        (win_length/2 + k * hop_length) / sample_rate, and absolute frame
        datetimes are datetime_start + times_s. If this fails against the
        real compute_stft_power, amend plan D8 before Stage 3."""
        entries = legacy_stft_entries(awkward_wav, golden_config)
        assert len(entries) == 1
        entry = entries[0]

        sr = 96000
        win = golden_config.pipeline.stft_win_length
        hop = golden_config.pipeline.stft_hop_length
        times = np.asarray(entry["times_s"], dtype=np.float64)

        expected = (win / 2 + hop * np.arange(len(times))) / sr
        np.testing.assert_allclose(
            times, expected, rtol=1e-12, atol=0,
            err_msg="Frame times are not window-centre as plan D8 assumes",
        )

        abs_times = pd.Timestamp(entry["datetime_start"]) + pd.to_timedelta(
            times, unit="s"
        )
        first_expected = pd.Timestamp("2026-01-01 12:00:00") + pd.to_timedelta(
            win / 2 / sr, unit="s"
        )
        assert abs_times[0] == first_expected

    def test_frame_count_formula(self, awkward_wav, golden_config):
        """PLAN ASSUMPTION (D8: boundary=None, padded=False): the frame
        count for n samples is 1 + (n - win_length) // hop_length, i.e.
        no padded or boundary frames are fabricated."""
        entries = legacy_stft_entries(awkward_wav, golden_config)
        entry = entries[0]

        n_samples = 13 * 96000
        win = golden_config.pipeline.stft_win_length
        hop = golden_config.pipeline.stft_hop_length
        expected_frames = 1 + (n_samples - win) // hop

        assert entry["power"].shape[1] == expected_frames
        assert len(entry["times_s"]) == expected_frames

    def test_frequency_axis_respects_limits(self, awkward_wav, golden_config):
        """The returned frequency axis is already trimmed to
        [stft_fmin_hz, stft_fmax_hz] and matches the power array's
        frequency dimension."""
        entries = legacy_stft_entries(awkward_wav, golden_config)
        entry = entries[0]
        freqs = np.asarray(entry["freqs_hz"])

        assert np.all(freqs >= golden_config.pipeline.stft_fmin_hz)
        assert np.all(freqs <= golden_config.pipeline.stft_fmax_hz)
        assert entry["power"].shape[0] == len(freqs)


class TestCalibrationLegacyBehaviour:
    """Pin calibration semantics the Stage-1 resolve_gain extraction
    must preserve exactly."""

    def test_soundtrap_calibration_is_scalar_gain(self, awkward_wav, golden_config):
        """PLAN ASSUMPTION (section 10): the soundtrap method reduces to
        one scalar gain, 10^(sens_db/20) * 1e-6, applied multiplicatively.
        Pinned with a tight tolerance here; the exact floating-point
        operation order is pinned bit-exactly in PR 1 once
        calibration_methods.py is in front of us."""
        parser = get_parser(golden_config.input)
        segment = read_audio(awkward_wav, golden_config.input, parser=parser)[0]
        cal_df = load_calibration(golden_config.calibration)

        audio_pa, calibrated = apply_calibration(
            segment, cal_df, golden_config.calibration
        )

        assert calibrated is True
        gain = 10.0 ** (-176.0 / 20.0) * 1e-6  # serial 9999 in the fixture table
        np.testing.assert_allclose(
            audio_pa, segment.data * gain, rtol=1e-6, atol=0,
            err_msg="soundtrap calibration is not the plan section 10 scalar",
        )

    def test_unknown_serial_returns_uncalibrated(self, awkward_wav, golden_config):
        """Non-strict fallback: a serial absent from the calibration
        table returns the input data unchanged with calibrated=False.
        resolve_gain must reproduce this as (1.0, False)."""
        import copy

        config = copy.deepcopy(golden_config)
        config.input.serial_override = "123456"  # not in the fixture table
        config.calibration.strict = False

        parser = get_parser(config.input)
        segment = read_audio(awkward_wav, config.input, parser=parser)[0]
        cal_df = load_calibration(config.calibration)

        audio_pa, calibrated = apply_calibration(
            segment, cal_df, config.calibration
        )

        assert calibrated is False
        np.testing.assert_array_equal(audio_pa, segment.data)

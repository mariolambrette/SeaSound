"""
tests/test_characterisation.py

Characterisation tests pinning the per-file conventions of the Stage-1 path:
whole-second base-matrix indexing, the start-trim shift, STFT frame-count and
window-centre-timestamp conventions, and scalar calibration semantics.

Previously these pinned the legacy path before the refactor touched it; with
the legacy path removed they pin the *streaming* path against the same
conventions. The block-independence invariant they used to assert directly
(via compute_base_matrix) now lives in test_streaming_invariance.py, and the
golden-gate plumbing smoke test is gone with the gate.
"""

import numpy as np
import pandas as pd

from seasound.loader.reader import read_audio
from seasound.loader.filename_parsers import get_parser
from seasound.loader.calibration import load_calibration, apply_calibration

from tests.golden import (
    streamed_base_matrix_artifacts,
    streamed_stft_entries,
)


class TestBaseMatrixConventions:
    """Pin per-file base-matrix behaviour of the streaming path."""

    def test_datetime_index_from_filename(self, awkward_wav, golden_config):
        """With trim disabled, the per-file matrix has one row per whole
        second, a DatetimeIndex from the filename datetime at 1 s spacing."""
        artifacts = streamed_base_matrix_artifacts(awkward_wav, golden_config)

        assert len(artifacts) == 1
        matrix = artifacts[0]["base_matrix"]

        assert isinstance(matrix.index, pd.DatetimeIndex)
        assert len(matrix) == 13
        assert matrix.index[0] == pd.Timestamp("2026-01-01 12:00:00")
        assert matrix.index[-1] == pd.Timestamp("2026-01-01 12:00:12")
        steps = np.unique(np.diff(matrix.index.to_numpy()))
        assert steps.tolist() == [np.timedelta64(1, "s")]

    def test_trailing_partial_second_dropped(self, fractional_wav, golden_config):
        """A 13.5 s file yields exactly 13 rows: the trailing partial second
        is dropped at end-of-file (never at a block edge)."""
        artifacts = streamed_base_matrix_artifacts(fractional_wav, golden_config)
        matrix = artifacts[0]["base_matrix"]

        assert len(matrix) == 13
        assert matrix.index[-1] == pd.Timestamp("2026-01-01 12:00:12")

    def test_default_start_trim_shifts_output(self, awkward_wav, test_config):
        """per_file_trim_start_s=3.0: 3 s of audio are removed (a seek) and
        the datetime index starts 3 s after the filename time."""
        artifacts = streamed_base_matrix_artifacts(awkward_wav, test_config)
        matrix = artifacts[0]["base_matrix"]

        assert len(matrix) == 10  # 13 s - 3 s trim
        assert matrix.index[0] == pd.Timestamp("2026-01-01 12:00:03")


class TestStftConventions:
    """Pin per-file STFT conventions of the streaming store."""

    def test_window_centre_timestamps(self, awkward_wav, golden_config):
        """Frame k is stamped at the window-centre time
        (win_length/2 + k * hop_length) / sample_rate (D8), and absolute frame
        datetimes are datetime_start + times_s."""
        entries = streamed_stft_entries(awkward_wav, golden_config, block_seconds=7)
        assert len(entries) == 1
        entry = entries[0]

        sr = 96000
        win = golden_config.pipeline.stft_win_length
        hop = golden_config.pipeline.stft_hop_length
        times = np.asarray(entry["times_s"], dtype=np.float64)

        expected = (win / 2 + hop * np.arange(len(times))) / sr
        np.testing.assert_allclose(
            times, expected, rtol=1e-12, atol=0,
            err_msg="Frame times are not window-centre (D8)",
        )

        abs_times = pd.Timestamp(entry["datetime_start"]) + pd.to_timedelta(
            times, unit="s"
        )
        first_expected = pd.Timestamp("2026-01-01 12:00:00") + pd.to_timedelta(
            win / 2 / sr, unit="s"
        )
        assert abs_times[0] == first_expected

    def test_frame_count_formula(self, awkward_wav, golden_config):
        """For n samples the frame count is 1 + (n - win_length) // hop_length
        (D8: boundary=None, padded=False) — no fabricated padded frames."""
        entries = streamed_stft_entries(awkward_wav, golden_config, block_seconds=7)
        entry = entries[0]

        n_samples = 13 * 96000
        win = golden_config.pipeline.stft_win_length
        hop = golden_config.pipeline.stft_hop_length
        expected_frames = 1 + (n_samples - win) // hop

        assert entry["power"].shape[1] == expected_frames
        assert len(entry["times_s"]) == expected_frames

    def test_frequency_axis_respects_limits(self, awkward_wav, golden_config):
        """The frequency axis is trimmed to [stft_fmin_hz, stft_fmax_hz] and
        matches the power array's frequency dimension."""
        entries = streamed_stft_entries(awkward_wav, golden_config, block_seconds=7)
        entry = entries[0]
        freqs = np.asarray(entry["freqs_hz"])

        assert np.all(freqs >= golden_config.pipeline.stft_fmin_hz)
        assert np.all(freqs <= golden_config.pipeline.stft_fmax_hz)
        assert entry["power"].shape[0] == len(freqs)


class TestCalibrationBehaviour:
    """Pin calibration semantics the Stage-1 resolve_gain extraction
    preserves. apply_calibration is retained (resolve_gain + multiply) for
    these tests."""

    def test_soundtrap_calibration_is_scalar_gain(self, awkward_wav, golden_config):
        """The soundtrap method reduces to one scalar gain,
        10^(sens_db/20) * 1e-6, applied multiplicatively."""
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
            err_msg="soundtrap calibration is not the expected scalar",
        )

    def test_unknown_serial_returns_uncalibrated(self, awkward_wav, golden_config):
        """Non-strict fallback: a serial absent from the calibration table
        returns the input unchanged with calibrated=False."""
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

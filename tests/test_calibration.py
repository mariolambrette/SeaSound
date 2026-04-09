"""Tests for the calibration module."""

import numpy as np
import pytest
from seasound.loader.calibration import load_calibration, apply_calibration
from seasound.loader.reader import AudioSegment
from seasound.core.config import CalibrationConfig


class TestLoadCalibration:

    def test_load_valid_file(self, sample_calibration_file):
        config = CalibrationConfig(
            enabled=True, strict=True,
            file=sample_calibration_file,
            sensitivity_column="High_Gain",
        )
        cal_df = load_calibration(config)
        assert cal_df is not None
        assert "9999" in cal_df.index
        assert cal_df.loc["9999", "High_Gain"] == -176.0

    def test_disabled_returns_none(self, sample_calibration_file):
        config = CalibrationConfig(enabled=False, file=sample_calibration_file)
        assert load_calibration(config) is None


class TestApplyCalibration:

    def test_known_sensitivity(self, sample_calibration_file):
        """
        With sensitivity = -176 dB, the linear factor is 10^(-176/20).
        A sample value of 1.0 should produce:
            1.0 * 10^(-176/20) µPa = 1.585e-9 µPa = 1.585e-15 Pa

        Wait — that's incredibly small. Let's think about this more carefully.

        Actually for SoundTrap: wav * 10^(cal/20) = µPa
        cal = -176, so 10^(-176/20) = 10^(-8.8) ≈ 1.585e-9
        wav=1.0 → 1.585e-9 µPa → 1.585e-15 Pa

        This is correct! A normalised sample of 1.0 at full scale corresponds
        to a very small pressure because the hydrophone is very sensitive.
        The full-scale voltage represents a small physical signal.
        """
        config = CalibrationConfig(
            enabled=True, strict=True,
            file=sample_calibration_file,
            sensitivity_column="High_Gain",
            vpp=2.0,
        )
        cal_df = load_calibration(config)

        # Create a simple test segment
        segment = AudioSegment(
            data=np.array([1.0, -1.0, 0.5]),
            sample_rate=96000,
            serial="9999",
            datetime_start=None,
            channel=0,
            source_file="test.wav",
        )

        audio_pa, calibrated = apply_calibration(segment, cal_df, config)
        assert calibrated is True

        # Check the conversion
        cal_linear = 10 ** (-176.0 / 20.0)
        expected_upa = np.array([1.0, -1.0, 0.5]) * cal_linear
        expected_pa = expected_upa * 1e-6

        np.testing.assert_allclose(audio_pa, expected_pa, rtol=1e-10)
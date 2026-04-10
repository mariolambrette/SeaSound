"""Tests for the calibration module."""

import numpy as np
import pandas as pd
import pytest
from seasound.loader.calibration import load_calibration, apply_calibration
from seasound.loader.reader import AudioSegment
from seasound.core.config import CalibrationConfig
from seasound.loader.calibration_methods import (
    SoundTrapMethod,
    StandardMethod,
    get_calibration_method,
)


class TestLoadCalibration:

    def test_load_valid_file(self, sample_calibration_file):
        config = CalibrationConfig(
            enabled=True, strict=True,
            file=sample_calibration_file,
            serial_column="Serial",
            sensitivity_column="High_Gain",
            method="soundtrap",
        )
        cal_df = load_calibration(config)
        assert cal_df is not None
        assert "9999" in cal_df.index
        assert cal_df.loc["9999", "High_Gain"] == -176.0

    def test_disabled_returns_none(self, sample_calibration_file):
        config = CalibrationConfig(enabled=False, file=sample_calibration_file)
        assert load_calibration(config) is None


class TestApplyCalibration:

    def test_known_sensitivity_standard(self, sample_calibration_file):
        """
        Standard method: samples → volts → µPa → Pa.

        With sensitivity = -176 dB, vpp = 2.0:
            volts = 1.0 × (2.0 / 2) = 1.0
            sensitivity_linear = 10^(-176/20) ≈ 1.585e-9
            µPa = 1.0 / 1.585e-9 ≈ 6.31e8
            Pa = 6.31e8 × 1e-6 = 630.96
        """
        config = CalibrationConfig(
            enabled=True, strict=True,
            file=sample_calibration_file,
            serial_column="Serial",
            sensitivity_column="High_Gain",
            method="standard",
            vpp=2.0,
        )
        cal_df = load_calibration(config)

        segment = AudioSegment(
            data=np.array([1.0]),
            sample_rate=96000,
            serial="9999",
            datetime_start=None,
            channel=0,
            source_file="test.wav",
        )

        audio_pa, calibrated = apply_calibration(segment, cal_df, config)
        assert calibrated is True

        # Manual calculation
        volts = 1.0 * (2.0 / 2.0)
        sens_linear = 10 ** (-176.0 / 20.0)
        expected_pa = (volts / sens_linear) * 1e-6

        np.testing.assert_allclose(audio_pa, [expected_pa], rtol=1e-10)

    def test_missing_serial_strict_raises(self, sample_calibration_file):
        config = CalibrationConfig(
            enabled=True,
            strict=True,
            file=sample_calibration_file,
            serial_column="Serial",
            sensitivity_column="High_Gain",
            method="soundtrap",
            vpp=2.0,
        )
        cal_df = load_calibration(config)

        segment = AudioSegment(
            data=np.array([0.1]),
            sample_rate=96000,
            serial=None,
            datetime_start=None,
            channel=0,
            source_file="test.wav",
        )

        with pytest.raises(Exception):
            apply_calibration(segment, cal_df, config)

    def test_missing_serial_non_strict_returns_uncalibrated(self, sample_calibration_file):
        config = CalibrationConfig(
            enabled=True,
            strict=False,
            file=sample_calibration_file,
            serial_column="Serial",
            sensitivity_column="High_Gain",
            method="soundtrap",
            vpp=2.0,
        )
        cal_df = load_calibration(config)

        raw = np.array([0.1, -0.2])
        segment = AudioSegment(
            data=raw,
            sample_rate=96000,
            serial=None,
            datetime_start=None,
            channel=0,
            source_file="test.wav",
        )

        audio_pa, calibrated = apply_calibration(segment, cal_df, config)
        assert calibrated is False
        np.testing.assert_allclose(audio_pa, raw)

    def test_nan_sensitivity_strict_raises(self, tmp_path):
        df = pd.DataFrame({
            "Serial": ["9999"],
            "High_Gain": [np.nan],
        })
        cal_path = str(tmp_path / "cal_nan.xlsx")
        df.to_excel(cal_path, index=False)

        config = CalibrationConfig(
            enabled=True,
            strict=True,
            file=cal_path,
            serial_column="Serial",
            sensitivity_column="High_Gain",
            method="soundtrap",
            vpp=2.0,
        )
        cal_df = load_calibration(config)

        segment = AudioSegment(
            data=np.array([0.2]),
            sample_rate=96000,
            serial="9999",
            datetime_start=None,
            channel=0,
            source_file="test.wav",
        )

        with pytest.raises(Exception):
            apply_calibration(segment, cal_df, config)

    def test_nan_sensitivity_non_strict_returns_uncalibrated(self, tmp_path):
        df = pd.DataFrame({
            "Serial": ["9999"],
            "High_Gain": [np.nan],
        })
        cal_path = str(tmp_path / "cal_nan.xlsx")
        df.to_excel(cal_path, index=False)

        config = CalibrationConfig(
            enabled=True,
            strict=False,
            file=cal_path,
            serial_column="Serial",
            sensitivity_column="High_Gain",
            method="soundtrap",
            vpp=2.0,
        )
        cal_df = load_calibration(config)

        raw = np.array([0.2, -0.2])
        segment = AudioSegment(
            data=raw,
            sample_rate=96000,
            serial="9999",
            datetime_start=None,
            channel=0,
            source_file="test.wav",
        )

        audio_pa, calibrated = apply_calibration(segment, cal_df, config)
        assert calibrated is False
        np.testing.assert_allclose(audio_pa, raw)


class TestCalibrationMethods:
    """Test calibration method classes directly."""

    def test_soundtrap_method(self):
        method = SoundTrapMethod()
        samples = np.array([1.0, 0.0, -1.0])
        result = method.to_pascals(samples, -176.0, vpp=2.0)
        cal_lin = 10 ** (-176.0 / 20.0)
        expected = samples * cal_lin * 1e-6
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    def test_standard_method(self):
        method = StandardMethod()
        samples = np.array([1.0])
        result = method.to_pascals(samples, -176.0, vpp=2.0)
        volts = 1.0 * 1.0  # vpp/2
        sens_lin = 10 ** (-176.0 / 20.0)
        expected = (volts / sens_lin) * 1e-6
        np.testing.assert_allclose(result, [expected], rtol=1e-10)

    def test_soundtrap_ignores_vpp(self):
        """SoundTrap method should give same result regardless of vpp."""
        method = SoundTrapMethod()
        samples = np.array([0.5])
        result_a = method.to_pascals(samples, -176.0, vpp=2.0)
        result_b = method.to_pascals(samples, -176.0, vpp=5.0)
        np.testing.assert_allclose(result_a, result_b)

    def test_standard_depends_on_vpp(self):
        """Standard method output should scale with vpp."""
        method = StandardMethod()
        samples = np.array([1.0])
        result_2v = method.to_pascals(samples, -176.0, vpp=2.0)
        result_4v = method.to_pascals(samples, -176.0, vpp=4.0)
        np.testing.assert_allclose(result_4v, result_2v * 2.0, rtol=1e-10)

    def test_get_calibration_method_unknown(self):
        with pytest.raises(ValueError):
            get_calibration_method("nonexistent")
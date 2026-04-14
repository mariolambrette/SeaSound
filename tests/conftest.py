"""
tests/conftest.py

Shared pytest fixtures for SeaSound tests.

These fixtures generate synthetic test data with known properties,
so we can verify that the pipeline produces correct results.
"""

import os
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

try:
    import soundfile as sf
except ImportError:
    sf = None


@pytest.fixture
def tmp_dir(tmp_path):
    """A temporary directory for test outputs."""
    return str(tmp_path)


@pytest.fixture
def synthetic_wav(tmp_path):
    """
    Generate a 10-second mono WAV file with a 1000 Hz sine wave.

    Properties:
    - Sample rate: 96000 Hz
    - Duration: 10 seconds
    - Signal: 1000 Hz sine, amplitude 0.5
    - Filename: 9999.260101120000.wav (SoundTrap format)
    - Expected datetime: 2026-01-01 12:00:00
    - Expected serial: "9999"

    Returns path to the WAV file.
    """
    if sf is None:
        pytest.skip("soundfile not installed")

    sr = 96000
    duration = 10
    freq = 1000
    amplitude = 0.5

    t = np.arange(sr * duration) / sr
    audio = amplitude * np.sin(2 * np.pi * freq * t)

    filepath = str(tmp_path / "9999.260101120000.wav")
    sf.write(filepath, audio, sr)

    return filepath


@pytest.fixture
def synthetic_base_matrix():
    """
    A 3600-row (1 hour) base matrix with known SPL values.

    First 1800 rows: 80 dB in all bands
    Last 1800 rows: 90 dB in all bands

    This allows testing of:
    - Median (should be 85 dB for full hour)
    - Mean (energy average: ~87 dB, not 85)
    - Percentiles (P25=80, P75=90)
    - Event detection (10 dB step)
    """
    from seasound.utils.spectral import tob_centre_frequencies, freq_column_names

    centres = tob_centre_frequencies(10, 50000)
    cols = freq_column_names(centres)

    n_rows = 3600
    data = np.full((n_rows, len(centres)), 80.0)
    data[1800:, :] = 90.0

    index = pd.date_range("2026-01-01 12:00:00", periods=n_rows, freq="1s")
    return pd.DataFrame(data, index=index, columns=cols)


@pytest.fixture
def sample_calibration_file(tmp_path):
    """
    Generate a small Excel calibration file.

    Contains:
    - Serial 9999 with High_Gain = -176.0
    - Serial 9471 with High_Gain = -174.5
    """
    df = pd.DataFrame({
        "Serial": ["9999", "9471"],
        "High_Gain": [-176.0, -174.5],
        "Low_Gain": [-170.0, -168.5],
    })
    path = str(tmp_path / "calibration.xlsx")
    df.to_excel(path, index=False)
    return path


@pytest.fixture
def test_config(tmp_path, sample_calibration_file):
    """
    A PipelineConfig pointing at temporary directories.
    """
    from seasound.core.config import PipelineConfig, InputConfig, CalibrationConfig, \
        ProcessingConfig, OutputConfig, DeploymentConfig

    return PipelineConfig(
        input=InputConfig(
            path=str(tmp_path),
            pattern="*.wav",
            recursive=False,
            filename_format="soundtrap",
            channel_strategy="mono",
        ),
        calibration=CalibrationConfig(
            enabled=True,
            strict=False,
            file=sample_calibration_file,
            serial_column="Serial",
            sensitivity_column="High_Gain",
            method="soundtrap",
            vpp=2.0,
        ),
        deployment=DeploymentConfig(enabled=False),
        output=OutputConfig(directory=str(tmp_path / "output")),
        pipeline=ProcessingConfig(
            resume=False,
            workers=1,
            max_freq_hz=50000,
            min_freq_hz=10,
            base_resolution_s=1,
            reference_pressure_pa=1e-6,
            missing_band_strategy="nan",
            cache_base_matrix=True,
            stft_cache_enabled=False,
            stft_nfft=2048,
            stft_win_length=2048,
            stft_hop_length=1024,
            stft_window="hann",
            stft_fmin_hz=10.0,
            stft_fmax_hz=50000.0,
            stft_dtype="float32",
        ),
    )


@pytest.fixture
def synthetic_stereo_wav(tmp_path):
    """
    Generate a 10-second stereo WAV file for resume/channel tests.
    Channel 0: 1000 Hz sine
    Channel 1: 2000 Hz sine
    """
    if sf is None:
        pytest.skip("soundfile not installed")

    sr = 96000
    duration = 10
    t = np.arange(sr * duration) / sr
    ch0 = 0.5 * np.sin(2 * np.pi * 1000 * t)
    ch1 = 0.5 * np.sin(2 * np.pi * 2000 * t)
    audio = np.column_stack([ch0, ch1])

    filepath = str(tmp_path / "9999.260101120000.wav")
    sf.write(filepath, audio, sr)
    return filepath
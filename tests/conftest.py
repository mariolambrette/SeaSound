"""
tests/conftest.py

Shared pytest fixtures for SeaSound tests.

These fixtures generate synthetic test data with known properties,
so we can verify that the pipeline produces correct results.
"""

import os           # pylint: disable=unused-import
import tempfile     # pylint: disable=unused-import
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
def test_config(tmp_path, sample_calibration_file): #pylint: disable=redefined-outer-name
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
    Generate a 10-second stereo WAV file for multi-channel testing.
    
    Returns
    -------
    str
        Path to synthetic stereo WAV file.
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

@pytest.fixture
def synthetic_long_base_matrix():
    """
    Deterministic 10-day synthetic base matrix for statistical/window tests.

    - 10 days at 1-second resolution (864,000 rows)
    - 5 TOB-style frequency columns
    - Values in realistic SPL range with deterministic pseudo-random noise
    - Includes a daily cycle + controlled burst events to exercise percentiles/windowing
    """
    n_seconds = 10 * 24 * 3600
    index = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=n_seconds,
        freq="1s",
        tz="UTC",
    )

    # Deterministic RNG for stable tests across environments
    rng = np.random.default_rng(42)

    # 5 representative TOB bands
    freqs_hz = [10.0, 100.0, 1000.0, 10000.0, 50000.0]
    cols = [f"{f}Hz" for f in freqs_hz]

    # Build signal components
    t = np.arange(n_seconds, dtype=float)

    # Daily cycle in dB-space (small amplitude; keeps realistic variation)
    daily = 3.0 * np.sin(2.0 * np.pi * t / 86400.0)

    # Low-amplitude random variation
    noise = rng.normal(loc=0.0, scale=1.2, size=(n_seconds, len(cols)))

    # Frequency-dependent baseline
    baselines = np.array([72.0, 75.0, 78.0, 81.0, 84.0], dtype=float)

    data = baselines + daily[:, None] + noise

    # Add deterministic burst events (for high-percentile behavior)
    burst_starts = [6 * 3600, 30 * 3600, 54 * 3600, 78 * 3600, 102 * 3600, 126 * 3600, 150 * 3600, 174 * 3600, 198 * 3600, 222 * 3600] #pylint: disable=line-too-long
    burst_len = 300  # 5 minutes
    for s in burst_starts:
        e = min(s + burst_len, n_seconds)
        data[s:e, :] += np.array([6.0, 8.0, 10.0, 7.0, 5.0], dtype=float)

    # Clip to realistic SPL bounds
    data = np.clip(data, 50.0, 120.0)

    out = pd.DataFrame(data, index=index, columns=cols)
    out.index.name = "datetime"
    return out


# ===========================================================================
# Streaming-refactor fixtures (PR 0)
#
# The fixtures above all describe "polite" inputs: exact whole-second
# durations, single files, contiguous time. The streaming refactor's
# failure modes live in awkward inputs, so these fixtures deliberately
# provide: durations indivisible by candidate block sizes, fractional
# trailing seconds, multi-file seams, duty-cycle gaps, and overlapping
# recordings.
#
# Audio content is seeded noise plus a tone so that every second of
# audio is statistically distinct. A seam bug (dropped, duplicated, or
# misaligned block) therefore changes output values; with a stationary
# pure sine it could hide behind identical-looking frames.
# ===========================================================================


def _write_soundtrap_wav(directory, serial, start, duration_s, sample_rate, seed):
    """
    Write a deterministic SoundTrap-named WAV file and return its path.

    Parameters
    ----------
    directory : pathlib.Path
        Directory to write into (created if absent).
    serial : str
        Hydrophone serial encoded in the filename.
    start : datetime
        Recording start time encoded in the filename.
    duration_s : float
        Duration in seconds (may be fractional).
    sample_rate : int
        Sample rate in Hz.
    seed : int
        RNG seed; different per file so no two fixtures share content.

    Returns
    -------
    str
        Path to the written WAV file.
    """
    if sf is None:
        pytest.skip("soundfile not installed")

    directory.mkdir(parents=True, exist_ok=True)
    n = int(round(duration_s * sample_rate))
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sample_rate
    audio = 0.3 * np.sin(2 * np.pi * 1000.0 * t) + 0.1 * rng.standard_normal(n)
    audio = np.clip(audio, -1.0, 1.0)

    name = f"{serial}.{start.strftime('%y%m%d%H%M%S')}.wav"
    filepath = str(directory / name)
    sf.write(filepath, audio, sample_rate)
    return filepath


@pytest.fixture
def golden_config(test_config): #pylint: disable=redefined-outer-name
    """
    ``test_config`` with the per-file start trim disabled.

    ``InputConfig.per_file_trim_start_s`` defaults to 3.0, so any fixture
    WAV processed under ``test_config`` silently loses its first three
    seconds and has its datetime shifted by the same amount. That default
    is pinned explicitly in the characterisation suite; everywhere else,
    golden identity comparisons must control it deliberately rather than
    inherit it, so duration and seam arithmetic in tests stays literal.
    """
    import copy

    config = copy.deepcopy(test_config)
    config.input.per_file_trim_start_s = 0.0
    return config


@pytest.fixture
def awkward_wav(tmp_path):
    """
    A 13-second mono WAV — deliberately indivisible by candidate
    streaming block sizes (7, 30, 60, 300 s), so every block-boundary
    invariance test exercises a trailing partial block.

    Properties:
    - Sample rate: 96000 Hz
    - Duration: 13 seconds
    - Filename: 9999.260101120000.wav (SoundTrap format)
    - Expected datetime: 2026-01-01 12:00:00
    """
    return _write_soundtrap_wav(
        tmp_path / "awkward", "9999",
        datetime(2026, 1, 1, 12, 0, 0), 13.0, 96000, seed=101,
    )


@pytest.fixture
def fractional_wav(tmp_path):
    """
    A 13.5-second mono WAV (96 kHz). The trailing half second does not
    fill a 1-second bin, pinning the legacy trailing-sample drop that
    the streamed path must reproduce at end-of-file (not end-of-block).
    """
    return _write_soundtrap_wav(
        tmp_path / "fractional", "9999",
        datetime(2026, 1, 1, 12, 0, 0), 13.5, 96000, seed=102,
    )


@pytest.fixture
def contiguous_wav_pair(tmp_path):
    """
    Two back-to-back recordings: 13 s starting 12:00:00 and 10 s starting
    12:00:13. Exercises the per-file STFT carry-buffer reset — the union
    of per-file frame sets must appear, never a fabricated cross-file frame.

    Returns
    -------
    tuple[str, list[str]]
        (directory, [first_path, second_path]) in chronological order.
    """
    directory = tmp_path / "contiguous"
    first = _write_soundtrap_wav(
        directory, "9999", datetime(2026, 1, 1, 12, 0, 0), 13.0, 96000, seed=201,
    )
    second = _write_soundtrap_wav(
        directory, "9999", datetime(2026, 1, 1, 12, 0, 13), 10.0, 96000, seed=202,
    )
    return str(directory), [first, second]


@pytest.fixture
def gapped_wav_pair(tmp_path):
    """
    Two duty-cycled recordings with a 47-second gap: 13 s starting
    12:00:00, then 10 s starting 12:01:00. Both products must index
    correctly across the gap with no off-by-one.

    Returns
    -------
    tuple[str, list[str]]
        (directory, [first_path, second_path]) in chronological order.
    """
    directory = tmp_path / "gapped"
    first = _write_soundtrap_wav(
        directory, "9999", datetime(2026, 1, 1, 12, 0, 0), 13.0, 96000, seed=301,
    )
    second = _write_soundtrap_wav(
        directory, "9999", datetime(2026, 1, 1, 12, 1, 0), 10.0, 96000, seed=302,
    )
    return str(directory), [first, second]


@pytest.fixture
def overlapping_wav_pair(tmp_path):
    """
    Two recordings overlapping by 5 seconds: 13 s starting 12:00:00 and
    10 s starting 12:00:08. The overlap window [12:00:08, 12:00:13) has
    different audio content in each file, so keep-first deduplication is
    observable in values, not just row counts.

    Returns
    -------
    tuple[str, list[str]]
        (directory, [first_path, second_path]) in chronological order.
    """
    directory = tmp_path / "overlapping"
    first = _write_soundtrap_wav(
        directory, "9999", datetime(2026, 1, 1, 12, 0, 0), 13.0, 96000, seed=401,
    )
    second = _write_soundtrap_wav(
        directory, "9999", datetime(2026, 1, 1, 12, 0, 8), 10.0, 96000, seed=402,
    )
    return str(directory), [first, second]

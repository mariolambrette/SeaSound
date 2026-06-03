"""Tests for read_audio's per-file start trim."""

from datetime import datetime

import numpy as np
import pytest
import soundfile as sf

from seasound.core.config import InputConfig
from seasound.loader.reader import read_audio


def _write_sine_wav(
    path: str, *, freq: float = 1000.0, duration_s: float = 5.0,
    sample_rate: int = 96000,
) -> None:
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    data = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    sf.write(path, data, sample_rate)


def test_trim_zero_is_noop(tmp_path):
    wav = str(tmp_path / "9999.250101120000.wav")
    _write_sine_wav(wav, duration_s=5.0, sample_rate=96000)
    config = InputConfig(
        filename_format="soundtrap", per_file_trim_start_s=0.0,
    )
    segs = read_audio(wav, config)
    assert len(segs[0].data) == 5 * 96000


def test_trim_default_three_seconds(tmp_path):
    wav = str(tmp_path / "9999.250101120000.wav")
    _write_sine_wav(wav, duration_s=5.0, sample_rate=96000)
    config = InputConfig(
        filename_format="soundtrap", per_file_trim_start_s=3.0,
    )
    segs = read_audio(wav, config)
    assert len(segs[0].data) == 2 * 96000


def test_trim_shifts_datetime_start(tmp_path):
    wav = str(tmp_path / "9999.250101120000.wav")
    _write_sine_wav(wav, duration_s=5.0, sample_rate=96000)
    config = InputConfig(
        filename_format="soundtrap", per_file_trim_start_s=3.0,
    )
    segs = read_audio(wav, config)
    assert segs[0].datetime_start == datetime(2025, 1, 1, 12, 0, 3)


def test_trim_exceeds_duration_returns_empty(tmp_path, caplog):
    wav = str(tmp_path / "9999.250101120000.wav")
    _write_sine_wav(wav, duration_s=1.0, sample_rate=96000)
    config = InputConfig(
        filename_format="soundtrap", per_file_trim_start_s=3.0,
    )
    with caplog.at_level("WARNING"):
        segs = read_audio(wav, config)
    assert len(segs) == 1
    assert len(segs[0].data) == 0
    assert any("exceeds file duration" in r.message for r in caplog.records)


def test_trim_audio_content_starts_after_trim(tmp_path):
    """The first sample of the trimmed audio should be the sample that
    was at t=trim in the original recording."""
    sample_rate = 96000
    wav = str(tmp_path / "9999.250101120000.wav")
    duration_s = 5.0
    t = np.arange(int(sample_rate * duration_s)) / sample_rate
    # Linear ramp 0 → 1 over duration_s
    data = (t / duration_s).astype(np.float32)
    sf.write(wav, data, sample_rate)
    config = InputConfig(
        filename_format="soundtrap", per_file_trim_start_s=3.0,
    )
    segs = read_audio(wav, config)
    # 3/5 of the way through ramp → 0.6
    assert segs[0].data[0] == pytest.approx(0.6, abs=1e-4)


def test_fractional_trim_shifts_by_actual_sample_count(tmp_path):
    """For a fractional trim, datetime shift uses the rounded sample
    count (n_trim / sample_rate), not the requested float."""
    sample_rate = 96000
    wav = str(tmp_path / "9999.250101120000.wav")
    _write_sine_wav(wav, duration_s=2.0, sample_rate=sample_rate)
    # Pick a value where round(trim * sr) != trim * sr exactly
    trim_s = 0.123456
    config = InputConfig(
        filename_format="soundtrap", per_file_trim_start_s=trim_s,
    )
    segs = read_audio(wav, config)
    n_trim = round(trim_s * sample_rate)
    actual = n_trim / sample_rate
    expected_offset_us = round(actual * 1_000_000)
    actual_offset_us = (
        (segs[0].datetime_start - datetime(2025, 1, 1, 12, 0, 0))
        .microseconds
    )
    # Allow 1 µs of slop for rounding inside timedelta
    assert abs(actual_offset_us - expected_offset_us) <= 1

"""Tests for the audio reader module."""

import pytest
from seasound.loader.reader import parse_filename, read_audio, AudioSegment
from seasound.core.config import InputConfig
from datetime import datetime


class TestFilenameParser:
    """Test filename metadata extraction."""

    def test_soundtrap_format(self):
        config = InputConfig(filename_format="soundtrap")
        serial, dt = parse_filename("9471.251011103045.wav", config)
        assert serial == "9471"
        assert dt == datetime(2025, 10, 11, 10, 30, 45)

    def test_soundtrap_with_path(self):
        config = InputConfig(filename_format="soundtrap")
        serial, dt = parse_filename("/data/raw/9471.251011103045.wav", config)
        assert serial == "9471"

    def test_wildlife_format(self):
        config = InputConfig(filename_format="wildlife")
        serial, dt = parse_filename("SM4_20251011_103045.wav", config)
        assert serial == "SM4"
        assert dt == datetime(2025, 10, 11, 10, 30, 45)

    def test_iclisten_format(self):
        config = InputConfig(filename_format="iclisten")
        serial, dt = parse_filename(
            "icListenHF_1234_20251011_103045.wav", config
        )
        assert serial == "1234"
        assert dt == datetime(2025, 10, 11, 10, 30, 45)

    def test_unparseable_returns_none(self):
        config = InputConfig(filename_format="soundtrap")
        serial, dt = parse_filename("weirdname.wav", config)
        assert serial is None


class TestReader:
    """Test audio file reading."""

    def test_read_synthetic_wav(self, synthetic_wav):
        config = InputConfig(
            filename_format="soundtrap",
            channel_strategy="mono",
        )
        segments = read_audio(synthetic_wav, config)
        assert len(segments) == 1

        seg = segments[0]
        assert seg.sample_rate == 96000
        assert seg.serial == "9999"
        assert seg.channel == 0
        assert len(seg.data) == 96000 * 10  # 10 seconds

    def test_read_missing_file(self):
        config = InputConfig()
        with pytest.raises(Exception):
            read_audio("/nonexistent/file.wav", config)
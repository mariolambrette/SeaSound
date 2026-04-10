"""Tests for the audio reader and filename parsers."""

import pytest
from datetime import datetime
from seasound.loader.reader import read_audio, AudioSegment
from seasound.loader.filename_parsers import (
    SoundTrapParser,
    WildlifeParser,
    IcListenParser,
    CustomParser,
    ManualMetadataParser,
    get_parser,
    FileMetadata,
)
from seasound.core.config import InputConfig


class TestFilenameParsers:
    """Test filename metadata extraction via parser classes."""

    def test_soundtrap_format(self):
        parser = SoundTrapParser()
        meta = parser.parse("9471.251011103045.wav")
        assert meta.serial == "9471"
        assert meta.datetime_start == datetime(2025, 10, 11, 10, 30, 45)

    def test_soundtrap_with_path(self):
        parser = SoundTrapParser()
        meta = parser.parse("/data/raw/9471.251011103045.wav")
        assert meta.serial == "9471"

    def test_wildlife_format(self):
        parser = WildlifeParser()
        meta = parser.parse("SM4_20251011_103045.wav")
        assert meta.serial == "SM4"
        assert meta.datetime_start == datetime(2025, 10, 11, 10, 30, 45)

    def test_iclisten_format(self):
        parser = IcListenParser()
        meta = parser.parse("icListenHF_1234_20251011_103045.wav")
        assert meta.serial == "1234"
        assert meta.datetime_start == datetime(2025, 10, 11, 10, 30, 45)

    def test_custom_parser(self):
        parser = CustomParser(
            regex=r"(?P<serial>\d+)\.(?P<datetime>\d{12})\.wav",
            datetime_format="%y%m%d%H%M%S",
        )
        meta = parser.parse("9471.251011103045.wav")
        assert meta.serial == "9471"
        assert meta.datetime_start == datetime(2025, 10, 11, 10, 30, 45)

    def test_manual_parser(self):
        parser = ManualMetadataParser(
            serial="9471",
            start_datetime="2025-10-11 12:00:00",
        )
        meta = parser.parse("any_filename.wav")
        assert meta.serial == "9471"
        assert meta.datetime_start == datetime(2025, 10, 11, 12, 0, 0)

    def test_manual_parser_no_datetime(self):
        parser = ManualMetadataParser(serial="9471", start_datetime=None)
        meta = parser.parse("any_filename.wav")
        assert meta.serial == "9471"
        assert meta.datetime_start is None

    def test_unparseable_returns_none_fields(self):
        parser = SoundTrapParser()
        meta = parser.parse("weirdname.wav")
        assert meta.serial is None

    def test_get_parser_from_config(self):
        config = InputConfig(filename_format="soundtrap")
        parser = get_parser(config)
        assert isinstance(parser, SoundTrapParser)

    def test_get_parser_unknown_raises(self):
        config = InputConfig(filename_format="nonexistent")
        with pytest.raises(Exception):
            get_parser(config)

    def test_all_parsers_return_file_metadata(self):
        """Every parser must return a FileMetadata instance."""
        parsers = [
            SoundTrapParser(),
            WildlifeParser(),
            IcListenParser(),
            ManualMetadataParser(serial=None, start_datetime=None),
        ]
        for parser in parsers:
            result = parser.parse("test.wav")
            assert isinstance(result, FileMetadata), (
                f"{parser.__class__.__name__}.parse() returned "
                f"{type(result)}, expected FileMetadata"
            )


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

    def test_read_with_explicit_parser(self, synthetic_wav):
        """Parser can be passed explicitly to avoid re-creation."""
        config = InputConfig(
            filename_format="soundtrap",
            channel_strategy="mono",
        )
        parser = SoundTrapParser()
        segments = read_audio(synthetic_wav, config, parser=parser)
        assert len(segments) == 1
        assert segments[0].serial == "9999"

    def test_read_missing_file(self):
        config = InputConfig()
        with pytest.raises(Exception):
            read_audio("/nonexistent/file.wav", config)
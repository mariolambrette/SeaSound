"""
seasound/loader/filename_parsers.py

Filename metadata extraction for different hydrophone manufacturers.

Each parser is a subclass of FilenameParser and is registered in
PARSER_REGISTRY. The reader module uses get_parser() to obtain
the configured parser.

To add a new parser:
    1. Create a subclass of FilenameParser
    2. Implement the parse() method returning FileMetadata
    3. Add it to PARSER_REGISTRY at the bottom of this file
"""

import os
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


from seasound.core.exceptions import ConfigError
from seasound.core.config import InputConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standardised output
# ---------------------------------------------------------------------------

@dataclass
class FileMetadata:
    """
    Standardised metadata extracted from an audio filename.

    Every parser must return this type, ensuring that downstream code
    (reader, calibration, cache) always receives the same structure
    regardless of which manufacturer's naming convention was used.

    Attributes
    ----------
    serial : str or None
        Hydrophone serial number. Used for calibration lookup.
        None if the parser could not extract it.
    datetime_start : datetime or None
        Recording start time. Used to build the DatetimeIndex.
        None if the parser could not extract it.
    """
    serial: Optional[str] = None
    datetime_start: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class FilenameParser(ABC):
    """
    Base class for filename parsers.

    Subclasses must implement parse() and set the `name` class attribute.
    The abstract method enforces that every parser returns FileMetadata —
    you cannot accidentally return a bare tuple or forget a field.

    The base class also provides a default __repr__ for logging.
    """
    name: str  # Must match the YAML config value

    @abstractmethod
    def parse(self, filepath: str) -> FileMetadata:
        """
        Extract metadata from a filename.

        Parameters
        ----------
        filepath : str
            Full or relative path to the audio file.
            Implementations should use os.path.basename() to
            extract just the filename.

        Returns
        -------
        FileMetadata
            Always returns a FileMetadata instance. Fields that
            cannot be parsed are set to None.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
    

# ---------------------------------------------------------------------------
# Implementations
# ---------------------------------------------------------------------------

class SoundTrapParser(FilenameParser):
    """
    Parse OceanInstruments SoundTrap filenames.

    Format: SERIAL.YYMMDDHHMMSS.wav
    Example: 9471.251011103045.wav
        → serial = "9471"
        → datetime = 2025-10-11 10:30:45

    SoundTrap devices (ST300, ST400, ST500, ST600) encode the serial
    number and recording start time directly in the filename using
    a 2-digit year format.
    """
    name = "soundtrap"

    def parse(self, filepath: str) -> FileMetadata:
        basename = os.path.basename(filepath)
        parts = basename.split(".")

        serial = None
        dt = None

        if len(parts) >= 3:
            if parts[0].isdigit():
                serial = parts[0]
            try:
                dt = datetime.strptime(parts[1], "%y%m%d%H%M%S")
            except (ValueError, IndexError):
                pass

        if serial is None:
            logger.warning(f"Could not extract serial from {basename}")
        if dt is None:
            logger.warning(f"Could not extract datetime from {basename}")

        return FileMetadata(serial=serial, datetime_start=dt)


class WildlifeParser(FilenameParser):
    """
    Parse Wildlife Acoustics (Song Meter) filenames.

    Format: PREFIX_YYYYMMDD_HHMMSS.wav
    Example: SM4_20251011_103045.wav
        → serial = "SM4"
        → datetime = 2025-10-11 10:30:45

    The prefix is treated as the serial/device identifier.
    The last two underscore-separated segments are date and time.
    """
    name = "wildlife"

    def parse(self, filepath: str) -> FileMetadata:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        parts = basename.split("_")

        serial = None
        dt = None

        if len(parts) >= 3:
            serial = parts[0]
            try:
                dt = datetime.strptime(
                    f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S"
                )
            except (ValueError, IndexError):
                pass

        if serial is None:
            logger.warning(f"Could not extract serial from {basename}")
        if dt is None:
            logger.warning(f"Could not extract datetime from {basename}")

        return FileMetadata(serial=serial, datetime_start=dt)


class IcListenParser(FilenameParser):
    """
    Parse Ocean Sonics icListen filenames.

    Format: icListenHF_SERIAL_YYYYMMDD_HHMMSS.wav
    Example: icListenHF_1234_20251011_103045.wav
        → serial = "1234"
        → datetime = 2025-10-11 10:30:45
    """
    name = "iclisten"

    def parse(self, filepath: str) -> FileMetadata:
        basename = os.path.splitext(os.path.basename(filepath))[0]
        parts = basename.split("_")

        serial = None
        dt = None

        if len(parts) >= 4:
            serial = parts[1]
            try:
                dt = datetime.strptime(
                    f"{parts[2]}_{parts[3]}", "%Y%m%d_%H%M%S"
                )
            except (ValueError, IndexError):
                pass

        if serial is None:
            logger.warning(f"Could not extract serial from {basename}")
        if dt is None:
            logger.warning(f"Could not extract datetime from {basename}")

        return FileMetadata(serial=serial, datetime_start=dt)


class CustomParser(FilenameParser):
    """
    User-supplied regex parser.

    The regex must contain named groups:
        (?P<serial>...)    — captures the serial number
        (?P<datetime>...)  — captures the datetime string

    The datetime string is parsed with the provided strptime format.

    Example config:
        filename_format: "custom"
        custom_regex: "(?P<serial>\\d+)\\.(?P<datetime>\\d{12})\\.wav"
        custom_datetime_format: "%y%m%d%H%M%S"
    """
    name = "custom"

    def __init__(self, regex: str, datetime_format: str):
        if not regex:
            raise ConfigError(
                "custom_regex is required when filename_format is 'custom'"
            )
        if not datetime_format:
            raise ConfigError(
                "custom_datetime_format is required when "
                "filename_format is 'custom'"
            )
        self.regex = regex
        self.datetime_format = datetime_format

    def parse(self, filepath: str) -> FileMetadata:
        basename = os.path.basename(filepath)
        match = re.match(self.regex, basename)

        if match is None:
            logger.warning(
                f"Custom regex did not match {basename}"
            )
            return FileMetadata()

        groups = match.groupdict()
        serial = groups.get("serial")
        dt = None

        dt_str = groups.get("datetime")
        if dt_str:
            try:
                dt = datetime.strptime(dt_str, self.datetime_format)
            except ValueError:
                logger.warning(
                    f"Could not parse datetime '{dt_str}' with "
                    f"format '{self.datetime_format}'"
                )

        return FileMetadata(serial=serial, datetime_start=dt)


class ManualMetadataParser(FilenameParser):
    """
    Bypass filename parsing entirely.

    Uses serial and datetime values supplied directly in the config.
    Useful when files have been renamed or use a non-standard naming
    convention, and the user knows the metadata.

    Config:
        filename_format: "manual"
        serial_override: "9471"
        start_datetime: "2025-10-11 12:00:00"
    """
    name = "manual"

    def __init__(self, serial: Optional[str], start_datetime: Optional[str]):
        self.serial = serial
        self.dt = None
        if start_datetime:
            try:
                self.dt = datetime.strptime(start_datetime, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    self.dt = datetime.fromisoformat(start_datetime)
                except ValueError:
                    raise ConfigError(
                        f"Could not parse start_datetime '{start_datetime}'. "
                        f"Use format: YYYY-MM-DD HH:MM:SS"
                    )

    def parse(self, filepath: str) -> FileMetadata:
        return FileMetadata(serial=self.serial, datetime_start=self.dt)
    

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

PARSER_REGISTRY: dict[str, type[FilenameParser]] = {
    "soundtrap": SoundTrapParser,
    "wildlife": WildlifeParser,
    "iclisten": IcListenParser,
    "custom": CustomParser,
    "manual": ManualMetadataParser,
}


def get_parser(config: InputConfig) -> FilenameParser:
    """
    Instantiate the configured filename parser.

    Parameters
    ----------
    config : InputConfig
        Input configuration containing filename_format and any
        format-specific settings.

    Returns
    -------
    FilenameParser
        Ready-to-use parser instance.

    Raises
    ------
    ConfigError
        If the filename_format is not recognised.
    """
    fmt = config.filename_format
    parser = PARSER_REGISTRY.get(fmt)

    if parser is None:
        available = ", ".join(sorted(PARSER_REGISTRY.keys()))
        raise ConfigError(
            f"Unknown filename_format '{fmt}'. "
            f"Available formats: {available}"
        )

    if fmt == "custom":
        return parser(
            regex=getattr(config, "custom_regex", None), # pyright: ignore[reportCallIssue]
            datetime_format=getattr(config, "custom_datetime_format", None), # pyright: ignore[reportCallIssue]
        )
    elif fmt == "manual":
        return parser(
            serial=getattr(config, "serial_override", None), # pyright: ignore[reportCallIssue]
            start_datetime=getattr(config, "start_datetime", None), # pyright: ignore[reportCallIssue]
        )
    else:
        return parser()
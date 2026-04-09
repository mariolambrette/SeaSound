"""
seasound/loader/reader.py

Audio file reading with filename metadata extraction.

Supports multiple hydrophone naming conventions and channel handling
strategies. Returns AudioSegment objects ready for calibration.
"""

import os
import re
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

from seasound.core.config import InputConfig
from seasound.core.exceptions import ReaderError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class AudioSegment:
    """
    One channel of one audio file, ready for calibration.

    This is the handoff object between the reader and the calibration module. It
    carries both the raw audio data and the metadata needed to look up
    calibration values and timestamp the output.

    Attributes
    ----------
    data : np.ndarray
        Audio samples as float64 in range [-1, 1]. This is the
        normalised format returned by soundfile — NOT in physical
        units yet. Calibration converts these to Pascals.
    sample_rate : int
        Sample rate in Hz (e.g. 96000, 144000, 384000).
    serial : str or None
        Hydrophone serial number extracted from the filename.
        Used to look up calibration sensitivity.
    datetime_start : datetime or None
        Recording start time extracted from the filename.
        Used to build the DatetimeIndex of the base matrix.
    channel : int
        0-indexed channel number. For mono files this is always 0.
    source_file : str
        Path to the original audio file (for logging and provenance).
    """
    data: np.ndarray
    sample_rate: int
    serial: Optional[str]
    datetime_start: Optional[datetime]
    channel: int
    source_file: str


# ---------------------------------------------------------------------------
# Filename parsers
# ---------------------------------------------------------------------------

def _parse_soundtrap(filename: str) -> tuple[Optional[str], Optional[datetime]]:
    """
    Parse SoundTrap filename: SERIAL.YYMMDDHHMMSS.wav
    Example: 9471.251011103045.wav → serial="9471", datetime=2025-10-11 10:30:45

    SoundTrap devices encode the serial number and recording start time
    directly in the filename. The date uses 2-digit year format.
    """
    basename = os.path.basename(filename)
    parts = basename.split(".")

    if len(parts) < 3:
        return None, None

    serial = parts[0] if parts[0].isdigit() else None

    try:
        dt = datetime.strptime(parts[1], "%y%m%d%H%M%S")
    except (ValueError, IndexError):
        dt = None

    return serial, dt


def _parse_wildlife(filename: str) -> tuple[Optional[str], Optional[datetime]]:
    """
    Parse Wildlife Acoustics filename: PREFIX_YYYYMMDD_HHMMSS.wav
    Example: SM4_20251011_103045.wav → serial="SM4", datetime=2025-10-11 10:30:45
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split("_")

    if len(parts) < 3:
        return None, None

    serial = parts[0]

    try:
        dt = datetime.strptime(f"{parts[-2]}_{parts[-1]}", "%Y%m%d_%H%M%S")
    except (ValueError, IndexError):
        dt = None

    return serial, dt


def _parse_iclisten(filename: str) -> tuple[Optional[str], Optional[datetime]]:
    """
    Parse icListen filename: icListenHF_SERIAL_YYYYMMDD_HHMMSS.wav
    Example: icListenHF_1234_20251011_103045.wav
    """
    basename = os.path.splitext(os.path.basename(filename))[0]
    parts = basename.split("_")

    if len(parts) < 4:
        return None, None

    serial = parts[1]

    try:
        dt = datetime.strptime(f"{parts[2]}_{parts[3]}", "%Y%m%d_%H%M%S")
    except (ValueError, IndexError):
        dt = None

    return serial, dt


def _parse_custom(
    filename: str,
    regex: str,
    datetime_format: str,
) -> tuple[Optional[str], Optional[datetime]]:
    """
    Parse filename using a user-supplied regex with named groups.

    The regex must contain:
        (?P<serial>...)    — captures the serial number
        (?P<datetime>...)  — captures the datetime string

    The datetime string is parsed with datetime_format (strptime syntax).
    """
    basename = os.path.basename(filename)
    match = re.match(regex, basename)

    if match is None:
        return None, None

    serial = match.group("serial") if "serial" in match.groupdict() else None

    try:
        dt_str = match.group("datetime")
        dt = datetime.strptime(dt_str, datetime_format)
    except (IndexError, ValueError, KeyError):
        dt = None

    return serial, dt


def parse_filename(
    filepath: str,
    config: InputConfig,
) -> tuple[Optional[str], Optional[datetime]]:
    """
    Dispatch to the appropriate filename parser based on config.

    Parameters
    ----------
    filepath : str
        Path to audio file.
    config : InputConfig
        Input configuration specifying filename_format.

    Returns
    -------
    tuple of (serial, datetime)
        Either or both may be None if parsing fails.
    """
    parsers = {
        "soundtrap": _parse_soundtrap,
        "wildlife": _parse_wildlife,
        "iclisten": _parse_iclisten,
    }

    fmt = config.filename_format

    if fmt == "custom":
        if not config.custom_regex or not config.custom_datetime_format:
            logger.warning(
                "filename_format is 'custom' but custom_regex or "
                "custom_datetime_format not set"
            )
            return None, None
        return _parse_custom(
            filepath, config.custom_regex, config.custom_datetime_format
        )

    parser = parsers.get(fmt)
    if parser is None:
        logger.warning(f"Unknown filename_format '{fmt}', cannot parse metadata")
        return None, None

    serial, dt = parser(filepath)

    if serial is None:
        logger.warning(f"Could not extract serial from {filepath}")
    if dt is None:
        logger.warning(f"Could not extract datetime from {filepath}")

    return serial, dt


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------


def read_audio(filepath: str, config: InputConfig) -> list[AudioSegment]:
    """
    Read an audio file and return one AudioSegment per output channel.

    This is the main entry point for the reader module. It:
    1. Opens the file with soundfile
    2. Extracts metadata from the filename
    3. Applies the configured channel strategy
    4. Returns a list of AudioSegment objects

    Parameters
    ----------
    filepath : str
        Path to the audio file.
    config : InputConfig
        Input configuration.

    Returns
    -------
    list[AudioSegment]
        Usually length 1 (mono strategy). Length > 1 for "auto" strategy
        on multi-channel files.

    Raises
    ------
    ReaderError
        If the file cannot be read.
    """
    if not SOUNDFILE_AVAILABLE:
        raise ReaderError(
            "soundfile library not installed. "
            "Install with: pip install soundfile"
        )

    if not os.path.isfile(filepath):
        raise ReaderError(f"File not found: {filepath}")
    
    # --- Read the file ---
    try:
        audio_data, sample_rate = sf.read(filepath, dtype="float64") # pyright: ignore[reportPossiblyUnboundVariable]
    except Exception as exc:
        raise ReaderError(f"Could not read {filepath}: {exc}")
    
    # --- Extract metadata from filename ---
    serial, dt_start = parse_filename(filepath, config)

    logger.info(
        f"Read {os.path.basename(filepath)}: "
        f"{sample_rate} Hz, "
        f"{'stereo' if audio_data.ndim > 1 else 'mono'}, "
        f"{len(audio_data) / sample_rate:.1f}s"
    )

    # --- Apply channel strategy ---
    strategy = config.channel_strategy

    if audio_data.ndim == 1:
        # Already mono — return as-is regardless of strategy
        return [AudioSegment(
            data=audio_data,
            sample_rate=sample_rate,
            serial=serial,
            datetime_start=dt_start,
            channel=0,
            source_file=filepath,
        )]

    n_channels = audio_data.shape[1]

    if strategy == "mono":
        # Average all channels
        mono = np.mean(audio_data, axis=1)
        return [AudioSegment(
            data=mono,
            sample_rate=sample_rate,
            serial=serial,
            datetime_start=dt_start,
            channel=0,
            source_file=filepath,
        )]

    elif strategy == "select":
        ch = config.selected_channel
        if ch >= n_channels:
            raise ReaderError(
                f"selected_channel={ch} but file only has {n_channels} channels"
            )
        return [AudioSegment(
            data=audio_data[:, ch],
            sample_rate=sample_rate,
            serial=serial,
            datetime_start=dt_start,
            channel=ch,
            source_file=filepath,
        )]

    elif strategy == "auto":
        # Return one segment per channel
        segments = []
        for ch in range(n_channels):
            segments.append(AudioSegment(
                data=audio_data[:, ch],
                sample_rate=sample_rate,
                serial=serial,
                datetime_start=dt_start,
                channel=ch,
                source_file=filepath,
            ))
        return segments

    elif strategy == "dual_gain":
        raise NotImplementedError(
            "dual_gain channel strategy is planned for a future release. "
            "Use 'mono' or 'select' for now."
        )
    
    else:
        raise ReaderError(f"Unknown channel_strategy: {strategy}")

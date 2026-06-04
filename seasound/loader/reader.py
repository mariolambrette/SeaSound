"""
seasound/loader/reader.py

Audio file reading with filename metadata extraction.

Supports multiple hydrophone naming conventions and channel handling
strategies. Returns AudioSegment objects ready for calibration.
"""

import os
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

from seasound.core.config import InputConfig
from seasound.core.exceptions import ReaderError
from seasound.loader.filename_parsers import FilenameParser, get_parser

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
        Audio samples as float32 in range [-1, 1]. This is the
        normalised format returned by soundfile (read with
        dtype="float32" — the Stage 0 memory optimisation and the
        golden-baseline numeric format) — NOT in physical
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
# Per-file start trim
# ---------------------------------------------------------------------------


def _apply_start_trim(
    audio_data: np.ndarray,
    sample_rate: int,
    datetime_start: Optional[datetime],
    trim_s: float,
    filepath: str,
) -> tuple[np.ndarray, Optional[datetime]]:
    """
    Trim ``trim_s`` seconds from the start of ``audio_data`` and shift
    ``datetime_start`` forward by the exact rounded sample count.

    Returns ``(trimmed_audio, shifted_datetime_start)``. If ``trim_s <= 0``
    the inputs are returned unchanged. If the file is shorter than the
    requested trim, an empty audio array is returned (with a warning) so
    the downstream channel-strategy branching still produces a
    well-formed segment.

    Slicing along axis 0 is used directly, so the helper is correct for
    both 1-D (mono) and 2-D (multi-channel) input arrays.
    """
    if trim_s <= 0:
        return audio_data, datetime_start

    n_trim = int(round(trim_s * sample_rate))
    if n_trim >= len(audio_data):
        logger.warning(
            "per_file_trim_start_s=%.3fs exceeds file duration for %s "
            "(%d samples available); returning empty audio.",
            trim_s, os.path.basename(filepath), len(audio_data),
        )
        return audio_data[:0], datetime_start

    trimmed = audio_data[n_trim:]
    actual_trim_s = n_trim / sample_rate
    if datetime_start is not None:
        datetime_start = datetime_start + timedelta(seconds=actual_trim_s)
    logger.debug(
        "Trimmed %.3fs from start of %s",
        actual_trim_s, os.path.basename(filepath),
    )
    return trimmed, datetime_start


# ---------------------------------------------------------------------------
# Main reader
# ---------------------------------------------------------------------------


def read_audio(
    filepath: str,
    config: InputConfig,
    parser: Optional[FilenameParser] = None,
) -> list[AudioSegment]:
    """
    Read an audio file and return one AudioSegment per output channel.

    Parameters
    ----------
    filepath : str
        Path to the audio file.
    config : InputConfig
        Input configuration.
    parser : FilenameParser, optional
        Filename parser to extract metadata. If None, one is created
        from the config (provided for efficiency when processing
        many files — avoids re-instantiating the parser each time).

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
        audio_data, sample_rate = sf.read(filepath, dtype="float32") # pyright: ignore[reportPossiblyUnboundVariable] #pylint: disable=line-too-long
    except Exception as exc:
        raise ReaderError(f"Could not read {filepath}: {exc}") from exc

    # --- Extract metadata from filename ---
    if parser is None:
        parser = get_parser(config)

    metadata = parser.parse(filepath)
    serial = metadata.serial
    dt_start = metadata.datetime_start

    # serial_override applies regardless of filename_format
    if config.serial_override:
        serial = config.serial_override

    logger.info(
        "Read %s: %s Hz, %s, %.1fs",
        os.path.basename(filepath),
        sample_rate,
        "stereo" if audio_data.ndim > 1 else "mono",
        len(audio_data) / sample_rate,
    )

    # --- Apply per-file start trim (before channel branching) ---
    audio_data, dt_start = _apply_start_trim(
        audio_data, sample_rate, dt_start,
        config.per_file_trim_start_s, filepath,
    )

    # --- Apply channel strategy ---
    strategy = config.channel_strategy

    if audio_data.ndim == 1:
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

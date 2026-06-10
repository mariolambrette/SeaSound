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
        audio_data, sample_rate = sf.read(filepath, dtype="float32") # type: ignore
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


# ---------------------------------------------------------------------------
# Streaming reader (refactor plan Stage 2)
# ---------------------------------------------------------------------------


def _output_channels(n_channels: int, config: InputConfig) -> list[int]:
    """
    Resolve output channel IDs for a file with n_channels, mirroring
    read_audio's channel-strategy branching exactly — including the
    legacy quirk that a mono file yields channel [0] before any
    strategy validation runs.
    """
    if n_channels == 1:
        return [0]

    strategy = config.channel_strategy

    if strategy == "mono":
        return [0]
    if strategy == "select":
        ch = config.selected_channel
        if ch >= n_channels:
            raise ReaderError(
                f"selected_channel={ch} but file only has {n_channels} channels"
            )
        return [ch]
    if strategy == "auto":
        return list(range(n_channels))
    if strategy == "dual_gain":
        raise NotImplementedError(
            "dual_gain channel strategy is planned for a future release. "
            "Use 'mono' or 'select' for now."
        )
    raise ReaderError(f"Unknown channel_strategy: {strategy}")


def probe_output_channels(filepath: str, config: InputConfig) -> list[int]:
    """
    Return the output channel IDs read_audio would produce for this
    file, from the file header alone — no sample data is read.

    Used by the resume check (_is_fully_cached), which previously read
    every file in full just to count channels.
    """
    if not SOUNDFILE_AVAILABLE:
        raise ReaderError(
            "soundfile library not installed. "
            "Install with: pip install soundfile"
        )
    if not os.path.isfile(filepath):
        raise ReaderError(f"File not found: {filepath}")
    try:
        info = sf.info(filepath) # pyright: ignore[reportPossiblyUnboundVariable]
    except Exception as exc:
        raise ReaderError(f"Could not read {filepath}: {exc}") from exc
    return _output_channels(info.channels, config)


def extract_channel_block(
    block: np.ndarray,
    strategy: str,
    channel: int,
) -> np.ndarray:
    """
    Derive one output channel's samples from a raw (possibly
    multi-channel) block, mirroring read_audio's per-strategy
    derivation. "mono" returns a fresh per-sample mean (bit-identical
    to averaging the whole file, since the mean is per-sample);
    "select"/"auto" return column views, exactly as the legacy
    segments are views of the full read.
    """
    if block.ndim == 1:
        return block
    if strategy == "mono":
        return np.mean(block, axis=1)
    return block[:, channel]


class AudioBlockReader:
    """
    Stream one audio file in whole-bin float32 blocks (refactor plan
    Stage 2 / D1): per-worker memory holds one block, never the file.

    Context manager. On entry it resolves everything read_audio
    resolves per file — filename metadata (serial, datetime_start),
    serial_override, per_file_trim_start_s (as a seek, with the same
    rounded-sample datetime shift), and the output channel set — and
    exposes them as attributes, so the streaming pipeline can resolve
    calibration and size accumulators before any samples are read.

    Attributes (after __enter__)
    ----------
    sample_rate : int
    n_channels : int
    n_bins : int
        Whole bins the file will produce after the start trim
        (trailing partial bin dropped at end-of-file, exactly as the
        legacy path trims to n_bins * bin_samples).
    channels : list[int]
        Output channel IDs under the configured strategy.
    serial : str or None
    datetime_start : datetime or None
        Filename datetime, shifted by the exact trimmed duration.
    source_file : str
        Duck-types as the segment argument of resolve_calibration.
    """

    def __init__(
        self,
        filepath: str,
        config: InputConfig,
        parser: Optional[FilenameParser] = None,
        bin_seconds: int = 1,
    ):
        self.source_file = filepath
        self._config = config
        self._parser = parser
        self._bin_seconds = bin_seconds
        self._sf = None

    def __enter__(self) -> "AudioBlockReader":
        if not SOUNDFILE_AVAILABLE:
            raise ReaderError(
                "soundfile library not installed. "
                "Install with: pip install soundfile"
            )
        if not os.path.isfile(self.source_file):
            raise ReaderError(f"File not found: {self.source_file}")

        try:
            self._sf = sf.SoundFile(self.source_file) #type: ignore
        except Exception as exc:
            raise ReaderError(
                f"Could not read {self.source_file}: {exc}"
            ) from exc

        self.sample_rate = int(self._sf.samplerate)  #pylint: disable=attribute-defined-outside-init
        self.n_channels = int(self._sf.channels)  #pylint: disable=attribute-defined-outside-init
        frames = int(self._sf.frames)

        # --- Extract metadata from filename ---
        parser = self._parser if self._parser is not None else get_parser(self._config)
        metadata = parser.parse(self.source_file)
        self.serial = metadata.serial  #pylint: disable=attribute-defined-outside-init
        self.datetime_start = metadata.datetime_start  #pylint: disable=attribute-defined-outside-init

        # serial_override applies regardless of filename_format
        if self._config.serial_override:
            self.serial = self._config.serial_override  #pylint: disable=attribute-defined-outside-init

        logger.info(
            "Read %s (streaming): %s Hz, %s, %.1fs",
            os.path.basename(self.source_file),
            self.sample_rate,
            "stereo" if self.n_channels > 1 else "mono",
            frames / self.sample_rate,
        )

        # --- Start trim as a seek (same arithmetic as _apply_start_trim) ---
        trim_s = self._config.per_file_trim_start_s
        usable = frames
        self._trim_frames = 0  #pylint: disable=attribute-defined-outside-init
        if trim_s > 0:
            n_trim = int(round(trim_s * self.sample_rate))
            if n_trim >= frames:
                logger.warning(
                    "per_file_trim_start_s=%.3fs exceeds file duration for %s "
                    "(%d samples available); returning empty audio.",
                    trim_s, os.path.basename(self.source_file), frames,
                )
                usable = 0
            else:
                self._sf.seek(n_trim)
                self._trim_frames = n_trim  #pylint: disable=attribute-defined-outside-init
                usable = frames - n_trim
                actual_trim_s = n_trim / self.sample_rate
                if self.datetime_start is not None:
                    self.datetime_start = self.datetime_start + timedelta(  #pylint: disable=attribute-defined-outside-init
                        seconds=actual_trim_s
                    )
                logger.debug(
                    "Trimmed %.3fs from start of %s",
                    actual_trim_s, os.path.basename(self.source_file),
                )

        self._usable_frames = usable  #pylint: disable=attribute-defined-outside-init
        self._bin_samples = int(self._bin_seconds * self.sample_rate)  #pylint: disable=attribute-defined-outside-init
        self.n_bins = usable // self._bin_samples  #pylint: disable=attribute-defined-outside-init

        # --- Output channel set under the configured strategy ---
        self.channels = _output_channels(self.n_channels, self._config) #pylint: disable=attribute-defined-outside-init

        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._sf is not None:
            self._sf.close()
            self._sf = None

    def blocks(self, block_seconds: int):
        """
        Yield (raw_block, t0) in file order until the last whole bin.

        raw_block is float32, 1-D (mono file) or (n, n_channels), with
        a length that is an exact multiple of bin_samples; the final
        block carries the remainder bins. t0 is the block's start
        datetime (datetime_start + bins_done seconds), or None when the
        filename carried no datetime.

        block_seconds must be a whole multiple of bin_seconds
        (config validation enforces the same for
        streaming_block_seconds vs base_resolution_s).
        """
        if self._sf is None:
            raise ReaderError(
                "AudioBlockReader.blocks() called outside the context manager"
            )
        if block_seconds < self._bin_seconds or block_seconds % self._bin_seconds:
            raise ValueError(
                f"block_seconds={block_seconds} must be a whole multiple of "
                f"bin_seconds={self._bin_seconds}"
            )

        bins_per_block = block_seconds // self._bin_seconds
        bins_done = 0
        while bins_done < self.n_bins:
            k = min(bins_per_block, self.n_bins - bins_done)
            n_samples = k * self._bin_samples
            data = self._sf.read(n_samples, dtype="float32")
            if len(data) != n_samples:
                raise ReaderError(
                    f"Short read from {self.source_file}: expected "
                    f"{n_samples} samples, got {len(data)}"
                )
            t0 = (
                self.datetime_start
                + timedelta(seconds=bins_done * self._bin_seconds)
                if self.datetime_start is not None
                else None
            )
            yield data, t0
            bins_done += k

    def read_tail(self):
        """
        Read the fractional tail: the samples past the last whole bin
        (always fewer than bin_samples), or None when the usable length
        divides evenly into bins.

        The base-matrix path must never consume these — it drops the
        partial bin (whole-bin resolution) — but the STFT must see the
        full trimmed file, so the streamed STFT producer reads the tail
        to stay frame-for-frame identical to a single whole-file STFT
        (refactor plan §9 test 3).

        Seeks explicitly, so the result does not depend on whether
        blocks() has run or how far it was consumed.
        """
        if self._sf is None:
            raise ReaderError(
                "AudioBlockReader.read_tail() called outside the "
                "context manager"
            )
        whole = self.n_bins * self._bin_samples
        tail_len = self._usable_frames - whole
        if tail_len <= 0:
            return None
        self._sf.seek(self._trim_frames + whole)
        data = self._sf.read(tail_len, dtype="float32")
        if len(data) != tail_len:
            raise ReaderError(
                f"Short tail read from {self.source_file}: expected "
                f"{tail_len} samples, got {len(data)}"
            )
        return data

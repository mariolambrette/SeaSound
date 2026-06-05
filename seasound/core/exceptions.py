"""
seasound/core/exceptions.py

Custom exception hierarchy for SeaSound.

All SeaSound exceptions inherit from SeaSoundError, so callers can
catch all pipeline-specific errors with a single except clause while
letting unexpected errors (bugs) propagate normally.
"""

class SeaSoundError(Exception):
    """Base exception for all SeaSound errors."""


class ConfigError(SeaSoundError):
    """Raised when configuration is invalid or incomplete."""


class CalibrationError(SeaSoundError):
    """Raised when calibration data is missing or invalid."""


class ReaderError(SeaSoundError):
    """Raised when an audio file cannot be read or parsed."""


class StftStoreError(SeaSoundError):
    """Raised when an STFT shard or the shard manifest is invalid."""

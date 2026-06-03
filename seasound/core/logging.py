"""
seasound/core/logging.py

Structured logging setup for the SeaSound pipeline.
All modules use: logger = logging.getLogger(__name__)
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """
    Configure the root 'seasound' logger.

    Call once at pipeline start. All modules that use
    logging.getLogger(__name__) will inherit this configuration.

    Parameters
    ----------
    level : str
        One of "DEBUG", "INFO", "WARNING", "ERROR".
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create the seasound logger (parent of all seasound.* loggers)
    logger = logging.getLogger("seasound")
    logger.setLevel(numeric_level)

    # Don't add handlers if they already exist (prevents duplicates
    # if setup_logging is called twice, e.g. in tests)
    if logger.handlers:
        return

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(numeric_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Prevent propagation to root logger (avoids duplicate messages)
    logger.propagate = False

"""
seasound/loader/cache.py

Parquet-based caching for base matrices.

Why Parquet instead of CSV or HDF5?
- Parquet is columnar: reading a subset of TOB bands is fast
- Parquet supports metadata: provenance travels with the file
- Parquet is compressed by default (~5-10x smaller than CSV)
- PyArrow reads Parquet into pandas very efficiently
- A 30-day deployment at 1s resolution (2.6M rows x 38 columns)
  is about 50 MB in Parquet vs 500+ MB in CSV

The cache enables the "compute once, analyse many" workflow.
After the first run, re-analysis reads from Parquet in seconds
instead of re-processing WAV files for minutes/hours.
"""

import os
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import seasound
from seasound.loader.reader import AudioSegment

logger = logging.getLogger(__name__)


def _cache_filename(wav_path: str, channel: int) -> str:
    """
    Derive cache filename from WAV path and channel.

    Example: 9471.251011103045.wav, channel 0
           → 9471.251011103045_ch0.parquet
    """
    base = os.path.splitext(os.path.basename(wav_path))[0]
    return f"{base}_ch{channel}.parquet"


def is_cached(wav_path: str, channel: int, cache_dir: str) -> bool:
    """Check if a base matrix Parquet exists for a given WAV file + channel."""
    fname = _cache_filename(wav_path, channel)
    return os.path.isfile(os.path.join(cache_dir, fname))


def save_base_matrix(
    matrix: pd.DataFrame,
    segment: AudioSegment,
    calibrated: bool,
    cache_dir: str,
) -> str:
    """
    Save a base matrix to Parquet with provenance metadata.

    The matrix gets a proper DatetimeIndex built from the segment's
    datetime_start and the row indices (which represent seconds).

    Parameters
    ----------
    matrix : pd.DataFrame
        Base matrix from compute_base_matrix(). Index is integer seconds.
    segment : AudioSegment
        Source segment (provides serial, datetime, channel, filepath).
    calibrated : bool
        Whether calibration was successfully applied.
    cache_dir : str
        Directory for cache files.

    Returns
    -------
    str
        Path to the saved Parquet file.
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Build DatetimeIndex from segment start time + second offsets
    if segment.datetime_start is not None:
        dt_index = pd.date_range(
            start=segment.datetime_start,
            periods=len(matrix),
            freq="1s",
        )
        matrix = matrix.copy()
        matrix.index = dt_index
        matrix.index.name = "datetime"
    else:
        logger.warning(
            f"No datetime for {segment.source_file}; "
            f"saving with integer index"
        )

    # Convert to PyArrow table
    table = pa.Table.from_pandas(matrix, preserve_index=True)

    # Attach metadata
    custom_meta = {
        b"seasound_version": seasound.__version__.encode(),
        b"serial": (segment.serial or "unknown").encode(),
        b"channel": str(segment.channel).encode(),
        b"sample_rate": str(segment.sample_rate).encode(),
        b"calibration_applied": str(calibrated).lower().encode(),
        b"source_file": os.path.basename(segment.source_file).encode(),
        b"datetime_start": (
            segment.datetime_start.isoformat()
            if segment.datetime_start else "unknown"
        ).encode(),
    }
    existing_meta = table.schema.metadata or {}
    table = table.replace_schema_metadata({**existing_meta, **custom_meta})

    # Write
    fname = _cache_filename(segment.source_file, segment.channel)
    path = os.path.join(cache_dir, fname)
    pq.write_table(table, path, compression="snappy")

    logger.debug(f"Cached base matrix: {fname} ({len(matrix)} rows)")
    return path


def load_base_matrix(parquet_path: str) -> pd.DataFrame:
    """Load a single cached base matrix."""
    df = pd.read_parquet(parquet_path)

    # Ensure datetime index
    if "datetime" in df.columns:
        df = df.set_index("datetime")
    if not isinstance(df.index, pd.DatetimeIndex):
        # Try to convert
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    return df


def load_all_cached(cache_dir: str) -> pd.DataFrame:
    """
    Load and concatenate all cached base matrices from a directory.

    Returns a single DataFrame sorted by datetime with duplicates removed.
    This is the primary input to Stage 2 (analysis).
    """
    import glob

    files = sorted(glob.glob(os.path.join(cache_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {cache_dir}")

    logger.info(f"Loading {len(files)} cached base matrices from {cache_dir}")

    frames = []
    for f in files:
        try:
            df = load_base_matrix(f)
            frames.append(df)
        except Exception as exc:
            logger.warning(f"Could not load {f}: {exc}")

    if not frames:
        raise FileNotFoundError("No valid Parquet files could be loaded")

    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="first")]

    logger.info(
        f"Merged matrix: {len(full):,} rows, "
        f"{full.index.min()} → {full.index.max()}"
    )
    return full
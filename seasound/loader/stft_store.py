"""
seasound/loader/stft_store.py

Chunked, windowed STFT power store (refactor plan §6, D3, D8).

Replaces the per-file ``.npz`` STFT cache with self-describing zarr
shards that can be read by datetime window without ever assembling the
deployment-wide matrix:

- One shard per file/channel (``<cache_dir>/stft/<basename>_ch<n>.zarr``),
  written by exactly one worker — parallel writes never contend (D3).
- Shards are internally chunked along the time axis, so a datetime
  window touches only the overlapping chunks (bounded memory).
- Values are **linear power** (as today's ``.npz``), not dB —
  dB conversion stays in the consumer, preserving current semantics.
- Every shard stores its full metadata in its own attributes, plus a
  ``complete: true`` attribute written as the **last** metadata act of
  ``finalise()``. A shard without the flag is a crash artifact and is
  treated as absent (D8).
- ``manifest.parquet`` (one row per shard) is a *derived cache*: the
  coordinator writes it serially from rows the workers return; any
  reader that finds it missing or inconsistent with the shards on disk
  regenerates the rows from shard attributes.

Frame timestamps follow the window-centre convention (D8): frame *k*
(0-based within a file) is stamped

    file_datetime_start + (win_length/2 + k * hop_length) / sample_rate

computed in float64 exactly as ``scipy.signal.stft`` returns
``times_s``, so store datetimes equal the legacy
``datetime_start + times_s`` to the nanosecond after pandas conversion.

The manifest layout is deliberately inside the ``stft/`` subdirectory:
``cache.load_all_cached`` globs ``<cache_dir>/*.parquet`` non-recursively
for *base matrices*, so the manifest must never live directly in
``cache_dir``.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd
import zarr

from seasound.core.exceptions import StftStoreError

logger = logging.getLogger(__name__)


def _clear_readonly_and_retry(func, path, _exc_info):
    """shutil.rmtree error hook: clear read-only attributes and retry
    (the standard treatment for Windows file removal)."""
    os.chmod(path, stat.S_IWRITE)
    func(path)


_LOCK_RETRY_ATTEMPTS = 8
_LOCK_RETRY_BASE_DELAY_S = 0.05  # exponential: ~6.4 s worst case total


def _retry_transient_locks(fn, *args, what: str = "", path: str = "", **kwargs):
    """
    Run one zarr mutation, retrying through transient PermissionError.

    zarr writes metadata atomically (``*.partial`` -> ``os.replace``);
    on Windows that replace fails with WinError 5 whenever *either*
    side holds a handle — most commonly an antivirus or indexer scan of
    the just-written ``.partial`` file, which clears within
    milliseconds. Every metadata-touching operation in the writer goes
    through this helper; retried operations here are idempotent (attrs
    updates rewrite the same content; a failed append never advanced
    the array shape, so re-appending rewrites the same chunk region).
    """
    for attempt in range(_LOCK_RETRY_ATTEMPTS):
        try:
            return fn(*args, **kwargs)
        except PermissionError as exc:
            if attempt == _LOCK_RETRY_ATTEMPTS - 1:
                raise StftStoreError(
                    f"Persistent file lock while {what or 'writing'} "
                    f"{path}: {exc}. Transient antivirus/indexer locks "
                    f"are retried automatically; a persistent failure "
                    f"means another process holds the shard open."
                ) from exc
            time.sleep(_LOCK_RETRY_BASE_DELAY_S * (2 ** attempt))


def _remove_existing_shard(path: str, attempts: int = 5) -> None:
    """
    Remove an existing shard directory before rewriting it.

    The writer owns this deletion instead of relying on zarr's
    ``mode="w"`` overwrite: on Windows, rewriting metadata in place
    (``*.partial`` -> ``os.replace`` -> ``zarr.json``) fails with
    WinError 5 whenever the destination name holds a transient handle
    (antivirus/indexer scans of just-touched files, NT pending-delete).
    Deleting the whole directory first — with read-only clearing and a
    short retry/backoff for those transient locks — and creating into a
    clean path sidesteps the overwrite entirely. This is the path the
    resume rule takes when rebuilding a flagless crash shard.
    """
    if not os.path.isdir(path):
        return
    # onexc (3.12+) supersedes the deprecated onerror; same handler works
    # for both since it ignores the third argument.
    rmtree_kwargs = (
        {"onexc": _clear_readonly_and_retry}
        if sys.version_info >= (3, 12)
        else {"onerror": _clear_readonly_and_retry}
    )
    for attempt in range(attempts):
        try:
            shutil.rmtree(path, **rmtree_kwargs)
            return
        except OSError as exc:
            if attempt == attempts - 1:
                raise StftStoreError(
                    f"Could not remove existing STFT shard at {path} after "
                    f"{attempts} attempts: {exc}"
                ) from exc
            time.sleep(0.2 * (attempt + 1))


STFT_SUBDIR = "stft"
MANIFEST_NAME = "manifest.parquet"
SHARD_SUFFIX = ".zarr"

#: One manifest row per shard; ``shard_path`` is the shard directory
#: name relative to the stft directory, so caches survive being moved.
MANIFEST_COLUMNS = [
    "shard_path",
    "channel",
    "serial",
    "datetime_start",
    "t_start",
    "t_end",
    "n_frames",
    "n_freq",
    "sample_rate",
    "hop",
    "win",
    "window",
    "dtype",
    "codec",
    "complete",
]


# ---------------------------------------------------------------------------
# Frame timestamp convention (D8) — single implementation
# ---------------------------------------------------------------------------


def frame_times_s(
    n_frames: int,
    win_length: int,
    hop_length: int,
    sample_rate: int,
) -> np.ndarray:
    """
    Window-centre frame times in seconds, float64, exactly as
    ``scipy.signal.stft(..., boundary=None, padded=False)`` returns them.
    """
    return (win_length / 2 + hop_length * np.arange(n_frames)) / sample_rate


def frame_datetimes(
    datetime_start: datetime,
    n_frames: int,
    win_length: int,
    hop_length: int,
    sample_rate: int,
) -> pd.DatetimeIndex:
    """
    Absolute frame datetimes: ``datetime_start + times_s`` via the same
    pandas conversion the legacy ``build_stft_matrix`` applies, so the
    result is nanosecond-equal to the current convention.
    """
    times = frame_times_s(n_frames, win_length, hop_length, sample_rate)
    return pd.DatetimeIndex(
        pd.Timestamp(datetime_start) + pd.to_timedelta(times, unit="s")
    )


def stft_dir_for(cache_dir: str) -> str:
    """The store directory for a given pipeline cache directory."""
    return os.path.join(cache_dir, STFT_SUBDIR)


def shard_name(source_file: str, channel: int) -> str:
    """Shard directory name for a source file and channel."""
    base = os.path.splitext(os.path.basename(source_file))[0]
    return f"{base}_ch{channel}{SHARD_SUFFIX}"


def shard_complete(stft_dir: str, source_file: str, channel: int) -> bool:
    """Whether a complete STFT shard exists for a source file + channel.

    Reads only the shard's ``complete`` attribute (metadata, no array
    data), the same source of truth ``rebuild_manifest_rows`` uses: a
    missing shard, or a flagless one (a crash artifact), counts as
    absent — so the resume rule rebuilds it (§12).
    """
    path = os.path.join(stft_dir, shard_name(source_file, channel))
    if not os.path.isdir(path):
        return False
    try:
        group = zarr.open_group(path, mode="r")
        return bool(group.attrs.get("complete", False))
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("Could not read STFT shard %s: %s", path, exc)
        return False


# ---------------------------------------------------------------------------
# Writer (incremental, one shard per worker)
# ---------------------------------------------------------------------------


class StftShardWriter:
    """
    Incremental writer for one STFT shard.

    Opens the shard with ``mode="w"`` — an existing shard at the same
    path (including a flagless crash artifact) is recreated from
    scratch, which is exactly the resume rule's rebuild behaviour.

    ``append`` extends the power array along the time axis and never
    holds more than the appended block. ``finalise`` records the frame
    count and time extent, then writes ``complete: true`` as the final
    metadata act and returns the shard's manifest row.

    Parameters
    ----------
    shard_path : str
        Full path of the shard directory (use ``shard_name`` to build
        the conventional name).
    freqs_hz : np.ndarray
        Frequency axis (already trimmed to [fmin, fmax], as
        compute_stft_power returns it). Stored as a float64 array
        inside the shard, not in JSON attributes.
    sample_rate, hop_length, win_length : int
        STFT parameters; together with ``datetime_start`` they define
        every frame datetime (D8).
    datetime_start : datetime
        File start time. Required: the store is datetime-addressed,
        so a shard cannot exist for a file without one (the legacy
        ``build_stft_matrix`` likewise skips such files).
    channel, serial :
        Provenance, recorded in attributes and the manifest row.
    """

    def __init__(
        self,
        shard_path: str,
        freqs_hz: np.ndarray,
        sample_rate: int,
        hop_length: int,
        win_length: int,
        datetime_start: datetime,
        *,
        channel: int,
        serial: Optional[str],
        window: str = "hann",
        dtype: str = "float32",
        time_chunk_frames: int = 8192,
        codec: str = "zstd",
        codec_level: int = 3,
    ):
        if datetime_start is None:
            raise StftStoreError(
                f"Cannot write STFT shard for {shard_path}: the store is "
                f"datetime-addressed and the file has no datetime_start"
            )
        if codec != "zstd":
            raise StftStoreError(
                f"Unsupported shard codec '{codec}' (only 'zstd' is "
                f"implemented; hdf5 fallback is a separate store format)"
            )

        self.shard_path = shard_path
        self._freqs = np.asarray(freqs_hz, dtype=np.float64)
        self._n_freq = len(self._freqs)
        self._sample_rate = int(sample_rate)
        self._hop = int(hop_length)
        self._win = int(win_length)
        self._datetime_start = pd.Timestamp(datetime_start)
        self._dtype = np.dtype(dtype)
        self._n_frames = 0
        self._finalised = False

        # Own the removal of any pre-existing shard (crash artifact or
        # rebuild) rather than relying on zarr's mode="w" overwrite —
        # see _remove_existing_shard for the Windows rationale.
        _remove_existing_shard(shard_path)

        # Provisional attributes go into the group-creation metadata
        # write (one write instead of two): everything except the
        # completion markers, so a crash mid-write leaves a
        # self-evidently incomplete shard.
        provisional_attrs = {
            "channel": int(channel),
            "serial": serial if serial is not None else "",
            "datetime_start": self._datetime_start.isoformat(),
            "sample_rate": self._sample_rate,
            "hop": self._hop,
            "win": self._win,
            "window": window,
            "dtype": self._dtype.name,
            "codec": codec,
        }
        self._group = _retry_transient_locks(
            zarr.open_group, shard_path, mode="w",
            attributes=provisional_attrs,
            what="creating shard", path=shard_path,
        )
        self._power = _retry_transient_locks(
            self._group.create_array,
            "power",
            shape=(self._n_freq, 0),
            chunks=(self._n_freq, int(time_chunk_frames)),
            dtype=self._dtype,
            compressors=zarr.codecs.ZstdCodec(level=codec_level),
            what="creating power array", path=shard_path,
        )
        freq_arr = _retry_transient_locks(
            self._group.create_array,
            "freqs_hz", shape=(self._n_freq,), dtype="float64",
            what="creating frequency axis", path=shard_path,
        )
        _retry_transient_locks(
            freq_arr.__setitem__, slice(None), self._freqs,
            what="writing frequency axis", path=shard_path,
        )

    def append(self, power_block: np.ndarray) -> None:
        """
        Append one block of frames: shape (n_freq, n_frames_in_block).
        """
        if self._finalised:
            raise StftStoreError(
                f"append() after finalise() on {self.shard_path}"
            )
        block = np.asarray(power_block, dtype=self._dtype)
        if block.ndim != 2 or block.shape[0] != self._n_freq:
            raise StftStoreError(
                f"Power block shape {block.shape} does not match "
                f"(n_freq={self._n_freq}, n_frames_in_block)"
            )
        if block.shape[1] == 0:
            return
        _retry_transient_locks(
            self._power.append, block, axis=1,
            what="appending frames to", path=self.shard_path,
        )
        self._n_frames += block.shape[1]

    def finalise(self) -> dict[str, Any]:
        """
        Record the frame count and time extent and mark the shard
        complete, then return its manifest row.

        Completion is one atomic metadata write whose new content
        includes ``complete: true`` — ``os.replace`` is atomic, so a
        crash anywhere before it leaves the previous, flagless metadata
        and every reader treats the shard as absent. (Identical crash
        semantics to a separate trailing write, with half the
        replace-over-existing operations, which matters on Windows —
        see _retry_transient_locks.)
        """
        if self._finalised:
            raise StftStoreError(
                f"finalise() called twice on {self.shard_path}"
            )

        if self._n_frames > 0:
            dts = frame_datetimes(
                self._datetime_start, self._n_frames,
                self._win, self._hop, self._sample_rate,
            )
            t_start, t_end = dts[0], dts[-1]
        else:
            t_start = t_end = None

        _retry_transient_locks(
            self._group.attrs.update,
            {
                "n_frames": self._n_frames,
                "t_start": t_start.isoformat() if t_start is not None else "",
                "t_end": t_end.isoformat() if t_end is not None else "",
                "complete": True,
            },
            what="finalising", path=self.shard_path,
        )
        self._finalised = True

        return _row_from_attrs(
            os.path.basename(self.shard_path),
            dict(self._group.attrs),
            self._n_freq,
        )


def _row_from_attrs(
    shard_basename: str,
    attrs: dict[str, Any],
    n_freq: int,
) -> dict[str, Any]:
    """Build one manifest row from a shard's attribute dict."""
    return {
        "shard_path": shard_basename,
        "channel": int(attrs["channel"]),
        "serial": attrs.get("serial", ""),
        "datetime_start": pd.Timestamp(attrs["datetime_start"]),
        "t_start": pd.Timestamp(attrs["t_start"]) if attrs.get("t_start") else pd.NaT,
        "t_end": pd.Timestamp(attrs["t_end"]) if attrs.get("t_end") else pd.NaT,
        "n_frames": int(attrs["n_frames"]),
        "n_freq": n_freq,
        "sample_rate": int(attrs["sample_rate"]),
        "hop": int(attrs["hop"]),
        "win": int(attrs["win"]),
        "window": attrs.get("window", ""),
        "dtype": attrs.get("dtype", ""),
        "codec": attrs.get("codec", ""),
        "complete": bool(attrs.get("complete", False)),
    }


# ---------------------------------------------------------------------------
# Manifest (derived cache; rebuildable from shard attributes)
# ---------------------------------------------------------------------------


def rebuild_manifest_rows(stft_dir: str) -> list[dict[str, Any]]:
    """
    Scan shard attributes (cheap: metadata only, no array data) and
    return manifest rows for every **complete** shard. Flagless shards
    are crash artifacts and are skipped with a warning.
    """
    rows: list[dict[str, Any]] = []
    if not os.path.isdir(stft_dir):
        return rows

    for name in sorted(os.listdir(stft_dir)):
        if not name.endswith(SHARD_SUFFIX):
            continue
        path = os.path.join(stft_dir, name)
        try:
            group = zarr.open_group(path, mode="r")
            attrs = dict(group.attrs)
            if not attrs.get("complete", False):
                logger.warning(
                    "Ignoring incomplete STFT shard (crash artifact): %s",
                    name,
                )
                continue
            rows.append(_row_from_attrs(name, attrs, group["freqs_hz"].shape[0]))
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.warning("Could not read STFT shard %s: %s", name, exc)
    return rows


def write_manifest(rows: list[dict[str, Any]], stft_dir: str) -> str:
    """
    Write ``manifest.parquet`` from worker-returned rows. Called by the
    coordinator only, serially — workers never touch this file (D3/D8).
    """
    os.makedirs(stft_dir, exist_ok=True)
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    path = os.path.join(stft_dir, MANIFEST_NAME)
    df.to_parquet(path)
    logger.debug("STFT manifest written: %s (%d shard(s))", path, len(df))
    return path


def load_manifest(stft_dir: str) -> Optional[pd.DataFrame]:
    """Load ``manifest.parquet`` if present; None otherwise."""
    path = os.path.join(stft_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Reader (windowed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RenderGrid:
    """
    The global downsampling grid for windowed rendering (refactor plan
    §8), computed from the manifest alone — no power arrays are touched.

    Reproduces exactly what the legacy ``build_stft_matrix`` did before
    chunking: one global ``resample(step).mean()`` over the whole
    (deduplicated) frame index, applied only when ``step`` exceeds the
    native frame spacing. ``labels`` are the surviving global bin
    left-edges (equivalently, the index of the legacy global matrix
    after ``dropna(how="all")``); when ``do_downsample`` is False they
    are the native frame datetimes themselves.

    Anchoring the grid here, from a fixed ``origin`` that is itself a
    global bin edge, is what makes a per-window resample produce labels
    that are a strict subset of the global labels — so chunked reading
    has no effect on pixel values.
    """
    do_downsample: bool
    labels: pd.DatetimeIndex
    origin: Optional[pd.Timestamp] = None
    step: Optional[pd.Timedelta] = None


class StftStore:
    """
    Datetime-windowed reader over the shards of one channel.

    The manifest is consulted first; if it is missing or inconsistent
    with the complete shards on disk, the rows are regenerated from
    shard attributes (in memory — writing the manifest stays the
    coordinator's job).

    Reads resolve a [t0, t1] window (inclusive at both ends, matching
    ``DataFrame.loc`` slicing semantics) against the manifest, open only
    the overlapping shards, slice each by datetime, concatenate, and
    drop duplicate timestamps keep-first in shard ``(t_start, path)``
    order — the earlier recording wins where files overlap, matching
    the current concat→sort→dedup behaviour.
    """

    def __init__(self, cache_dir: str, channel: int = 0):
        self.stft_dir = stft_dir_for(cache_dir)
        self.channel = channel

        rows = self._load_rows()
        df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
        df = df[df["channel"] == channel]
        df = df.sort_values(["t_start", "shard_path"], na_position="last")
        self._shards = df.reset_index(drop=True)
        self._freqs_cache: Optional[np.ndarray] = None

    # -- manifest handling ------------------------------------------------

    def _load_rows(self) -> list[dict[str, Any]]:
        on_disk = {
            row["shard_path"]: row for row in rebuild_manifest_rows(self.stft_dir)
        }
        manifest = load_manifest(self.stft_dir)
        if manifest is not None:
            manifest_paths = set(manifest["shard_path"].tolist())
            if manifest_paths == set(on_disk):
                return manifest.to_dict("records")
            logger.warning(
                "STFT manifest inconsistent with shards on disk "
                "(%d listed vs %d complete) — regenerating from shard "
                "attributes",
                len(manifest_paths), len(on_disk),
            )
        return list(on_disk.values())

    # -- public API ---------------------------------------------------------

    @property
    def shards(self) -> pd.DataFrame:
        """Manifest rows for this channel, sorted by (t_start, path)."""
        return self._shards.copy()

    def _freq_axis(self) -> np.ndarray:
        """The store's frequency axis (from the first shard), cached.
        Returned even for windows containing no frames, so consumers
        always know the frequency dimension of this channel."""
        if self._freqs_cache is None:
            if self._shards.empty:
                self._freqs_cache = np.empty(0, dtype=np.float64)
            else:
                group = zarr.open_group(
                    os.path.join(
                        self.stft_dir, self._shards.iloc[0]["shard_path"]
                    ),
                    mode="r",
                )
                self._freqs_cache = np.asarray(group["freqs_hz"][:])
        return self._freqs_cache

    def extent(self) -> tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
        """(global_t0, global_t1) across this channel's frames, from the
        manifest only — no array data is touched."""
        valid = self._shards.dropna(subset=["t_start", "t_end"])
        if valid.empty:
            return None, None
        return valid["t_start"].min(), valid["t_end"].max()

    def read(
        self,
        t0=None,
        t1=None,
    ) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray]:
        """
        Return ``(freqs_hz, times, power)`` for all frames with
        ``t0 <= frame_datetime <= t1``.

        ``t0``/``t1`` may be any user-supplied datetimes (nothing ties a
        read to file boundaries); None means the corresponding end of
        the store extent. ``power`` has shape (n_freq, n_frames) in the
        stored dtype; memory is bounded by the window, never the
        deployment.
        """
        ext0, ext1 = self.extent()
        if ext0 is None:
            freqs = self._freq_axis()
            return (
                freqs,
                pd.DatetimeIndex([]),
                np.empty((len(freqs), 0), dtype=np.float32),
            )
        t0 = pd.Timestamp(t0) if t0 is not None else ext0
        t1 = pd.Timestamp(t1) if t1 is not None else ext1

        freqs: Optional[np.ndarray] = None
        time_parts: list[np.ndarray] = []
        power_parts: list[np.ndarray] = []

        overlapping = self._shards.dropna(subset=["t_start", "t_end"])
        overlapping = overlapping[
            (overlapping["t_start"] <= t1) & (overlapping["t_end"] >= t0)
        ]

        for row in overlapping.itertuples():
            group = zarr.open_group(
                os.path.join(self.stft_dir, row.shard_path), mode="r"
            )
            shard_freqs = np.asarray(group["freqs_hz"][:])
            if freqs is None:
                freqs = shard_freqs
            elif not np.array_equal(freqs, shard_freqs):
                raise StftStoreError(
                    f"Frequency axis of shard {row.shard_path} does not "
                    f"match the other shards in this read — shards with "
                    f"different STFT parameters cannot be combined"
                )

            dts = frame_datetimes(
                row.datetime_start, row.n_frames,
                row.win, row.hop, row.sample_rate,
            )
            i0 = dts.searchsorted(t0, side="left")
            i1 = dts.searchsorted(t1, side="right")
            if i1 <= i0:
                continue
            time_parts.append(dts[i0:i1].asi8)
            power_parts.append(group["power"][:, i0:i1])

        if freqs is None or not time_parts:
            freqs = freqs if freqs is not None else self._freq_axis()
            return (
                freqs,
                pd.DatetimeIndex([]),
                np.empty((len(freqs), 0), dtype=np.float32),
            )

        times_ns = np.concatenate(time_parts)
        power = np.concatenate(power_parts, axis=1)

        # Stable sort, then keep-first on duplicates: with shards
        # pre-sorted by (t_start, path) and a stable sort, the first
        # occurrence of a duplicated timestamp is the earlier shard's
        # frame — files that overlap keep the earlier recording.
        order = np.argsort(times_ns, kind="stable")
        times_ns = times_ns[order]
        power = power[:, order]
        keep = np.ones(len(times_ns), dtype=bool)
        keep[1:] = times_ns[1:] != times_ns[:-1]

        return (
            freqs,
            pd.DatetimeIndex(times_ns[keep].view("datetime64[ns]")),
            power[:, keep],
        )

    def _global_frame_index(
        self, t0=None, t1=None,
    ) -> pd.DatetimeIndex:
        """
        Every frame datetime in ``[t0, t1]`` (deduplicated keep-first),
        derived from the manifest alone — no power arrays are opened.

        Matches ``read()``'s dedup exactly: shards are pre-sorted by
        (t_start, shard_path), frames concatenated in that order, then a
        stable sort + keep-first on duplicate timestamps, so overlapping
        files keep the earlier recording's frame.
        """
        ext0, ext1 = self.extent()
        if ext0 is None:
            return pd.DatetimeIndex([])
        t0 = pd.Timestamp(t0) if t0 is not None else ext0
        t1 = pd.Timestamp(t1) if t1 is not None else ext1

        overlapping = self._shards.dropna(subset=["t_start", "t_end"])
        overlapping = overlapping[
            (overlapping["t_start"] <= t1) & (overlapping["t_end"] >= t0)
        ]

        parts: list[np.ndarray] = []
        for row in overlapping.itertuples():
            dts = frame_datetimes(
                row.datetime_start, row.n_frames,
                row.win, row.hop, row.sample_rate,
            )
            i0 = dts.searchsorted(t0, side="left")
            i1 = dts.searchsorted(t1, side="right")
            if i1 > i0:
                parts.append(dts[i0:i1].asi8)

        if not parts:
            return pd.DatetimeIndex([])

        times_ns = np.concatenate(parts)
        times_ns = np.sort(times_ns, kind="stable")
        keep = np.ones(len(times_ns), dtype=bool)
        keep[1:] = times_ns[1:] != times_ns[:-1]
        return pd.DatetimeIndex(times_ns[keep].view("datetime64[ns]"))

    def render_grid(
        self,
        time_bins: Optional[int] = 12000,
        t0=None,
        t1=None,
    ) -> RenderGrid:
        """
        Build the global downsampling grid (§8) from the manifest.

        Reproduces the legacy global step exactly:
        ``step = max(1.0, total_duration_s / time_bins)`` seconds,
        applied only when it exceeds the native frame spacing (the
        median positive gap). The surviving global bin edges are taken
        from a default-origin ``resample`` of the global frame index —
        identical to the bins the legacy ``resample(step).mean()`` then
        ``dropna(how="all")`` would keep — and ``origin`` is set to the
        first such edge so per-window resamples align to this same grid.
        """
        frames = self._global_frame_index(t0, t1)
        native = RenderGrid(do_downsample=False, labels=frames)

        if (
            time_bins is None
            or time_bins <= 0
            or len(frames) <= 1
        ):
            return native

        duration = frames[-1] - frames[0]
        if not (
            isinstance(duration, pd.Timedelta) and duration > pd.Timedelta(0)
        ):
            return native

        step = pd.Timedelta(seconds=max(1.0, duration.total_seconds() / time_bins))

        diffs = frames.to_series().diff().dropna()
        positive = diffs[diffs > pd.Timedelta(0)]
        native_step = positive.median() if len(positive) else pd.Timedelta(0)
        if not isinstance(native_step, pd.Timedelta):
            native_step = pd.Timedelta(0)

        if step <= native_step:
            return native

        # Default-origin resample == the legacy global bins; keep only
        # occupied bins (every native frame is a full row, so 'occupied'
        # == survives dropna(how="all")). origin = first global bin edge.
        counts = pd.Series(1, index=frames).resample(step).count()
        labels = pd.DatetimeIndex(counts.index[counts.to_numpy() > 0])
        if len(labels) == 0:
            return native
        return RenderGrid(
            do_downsample=True,
            labels=labels,
            origin=pd.Timestamp(labels[0]),
            step=step,
        )

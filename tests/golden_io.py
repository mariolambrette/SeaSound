"""
tests/golden_io.py

Serialisation for the committed golden snapshots that replace the live
legacy oracle once Stage 6 deletes the legacy path (see the golden.py
docstring and refactor plan §9).

These helpers deliberately use plain parquet / npz / JSON rather than the
production cache I/O (``save_base_matrix`` and the npz STFT helpers), which
Stage 6 removes — the snapshots must outlive the code that produced them, so
they must not depend on it. The loaders return the *same dict shape* that
``golden.legacy_*`` / ``golden.streamed_*`` return, so the existing
``assert_artifacts_identical`` / ``assert_stft_entries_identical`` helpers
compare snapshot-vs-streamed unchanged.

Layout under ``tests/golden_fixtures/``::

    <id>.meta.json        # per-channel metadata, in channel order; "kind" tag
    <id>_ch<n>.parquet    # base-matrix snapshot (one per output channel)
    <id>_ch<n>.npz        # STFT snapshot: freqs_hz, times_s, power

Regenerate (deliberately, when the baseline legitimately changes) with::

    SEASOUND_UPDATE_GOLDEN=1 python -m pytest tests/test_golden_baseline.py

then commit the resulting ``tests/golden_fixtures/`` files.
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pandas as pd

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden_fixtures")


# ---------------------------------------------------------------------------
# Paths + small (de)serialisers
# ---------------------------------------------------------------------------

def _meta_path(snapshot_id: str) -> str:
    return os.path.join(GOLDEN_DIR, f"{snapshot_id}.meta.json")


def _bm_path(snapshot_id: str, channel: int) -> str:
    return os.path.join(GOLDEN_DIR, f"{snapshot_id}_ch{channel}.parquet")


def _stft_path(snapshot_id: str, channel: int) -> str:
    return os.path.join(GOLDEN_DIR, f"{snapshot_id}_ch{channel}.npz")


def _dt_to_json(ts: Any) -> str | None:
    return None if ts is None else pd.Timestamp(ts).isoformat()


def _dt_from_json(s: str | None):
    return None if s is None else pd.Timestamp(s)


def _load_meta(snapshot_id: str, expected_kind: str) -> dict:
    path = _meta_path(snapshot_id)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Golden snapshot '{snapshot_id}' is missing ({path}). Generate it "
            f"with: SEASOUND_UPDATE_GOLDEN=1 python -m pytest "
            f"tests/test_golden_baseline.py — then commit tests/golden_fixtures/."
        )
    with open(path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("kind") != expected_kind:
        raise ValueError(
            f"Snapshot '{snapshot_id}' is kind '{meta.get('kind')}', "
            f"expected '{expected_kind}'."
        )
    return meta


# ---------------------------------------------------------------------------
# Base-matrix snapshots
# ---------------------------------------------------------------------------

def save_base_matrix_snapshot(
    snapshot_id: str, artifacts: list[dict[str, Any]]
) -> None:
    """Write a per-channel base-matrix artifact list to committed fixtures.

    ``artifacts`` is the shape returned by ``legacy_base_matrix_artifacts`` /
    ``streamed_base_matrix_artifacts``: dicts with channel, serial,
    datetime_start, calibrated, base_matrix (DataFrame).
    """
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    channels = []
    for a in artifacts:
        channel = int(a["channel"])
        a["base_matrix"].to_parquet(_bm_path(snapshot_id, channel))
        channels.append({
            "channel": channel,
            "serial": a["serial"],
            "datetime_start": _dt_to_json(a["datetime_start"]),
            "calibrated": bool(a["calibrated"]),
        })
    with open(_meta_path(snapshot_id), "w", encoding="utf-8") as fh:
        json.dump({"kind": "base_matrix", "channels": channels}, fh, indent=2)


def load_base_matrix_snapshot(snapshot_id: str) -> list[dict[str, Any]]:
    """Load a base-matrix snapshot back into artifact shape."""
    meta = _load_meta(snapshot_id, "base_matrix")
    out: list[dict[str, Any]] = []
    for m in meta["channels"]:
        channel = m["channel"]
        out.append({
            "channel": channel,
            "serial": m["serial"],
            "datetime_start": _dt_from_json(m["datetime_start"]),
            "calibrated": m["calibrated"],
            "base_matrix": pd.read_parquet(_bm_path(snapshot_id, channel)),
        })
    return out


# ---------------------------------------------------------------------------
# STFT snapshots
# ---------------------------------------------------------------------------

def save_stft_snapshot(
    snapshot_id: str, entries: list[dict[str, Any]]
) -> None:
    """Write a per-channel STFT entry list to committed fixtures.

    ``entries`` is the shape returned by ``legacy_stft_entries`` /
    ``streamed_stft_entries``: dicts with channel, serial, datetime_start,
    freqs_hz, times_s, power. ``power`` dtype (float32/float16) is preserved.
    """
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    channels = []
    for e in entries:
        channel = int(e["channel"])
        np.savez(
            _stft_path(snapshot_id, channel),
            freqs_hz=np.asarray(e["freqs_hz"]),
            times_s=np.asarray(e["times_s"]),
            power=np.asarray(e["power"]),
        )
        channels.append({
            "channel": channel,
            "serial": e["serial"],
            "datetime_start": _dt_to_json(e["datetime_start"]),
        })
    with open(_meta_path(snapshot_id), "w", encoding="utf-8") as fh:
        json.dump({"kind": "stft", "channels": channels}, fh, indent=2)


def load_stft_snapshot(snapshot_id: str) -> list[dict[str, Any]]:
    """Load an STFT snapshot back into entry shape."""
    meta = _load_meta(snapshot_id, "stft")
    out: list[dict[str, Any]] = []
    for m in meta["channels"]:
        channel = m["channel"]
        with np.load(_stft_path(snapshot_id, channel)) as z:
            out.append({
                "channel": channel,
                "serial": m["serial"],
                "datetime_start": _dt_from_json(m["datetime_start"]),
                "freqs_hz": z["freqs_hz"],
                "times_s": z["times_s"],
                "power": z["power"],
            })
    return out

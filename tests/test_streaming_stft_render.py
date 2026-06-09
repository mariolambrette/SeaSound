"""
§9 test 7 — pixel-level rendered-array equivalence (refactor plan §8).

The spectrogram and annotated-spectrogram consumers render
``window.T.values`` (the array handed to imshow) for each window yielded
by ``iter_stft_windows``. This gate freezes the *pre-refactor* legacy
algorithm — concat every file's dB frames, one global
``resample(step).mean()`` then ``dropna(how="all")``, then slice by
``time_chunk`` — as the oracle, and asserts that the store-backed
windowed path produces bit-identical rendered arrays (and bin labels)
over the same data.

Coverage: a single file, two contiguous files, two gapped files, and a
full re-recording overlap (keep-first dedup), each with downsampling;
plus the whole-extent (time_chunk=None) and the native (time_bins=None,
no downsample) branches.
"""

import os

import numpy as np
import pandas as pd
import pytest

from seasound.core.stft import iter_stft_windows
from seasound.loader.stft_store import (
    StftShardWriter, stft_dir_for, shard_name, frame_datetimes,
)

SR, WIN, HOP, NFREQ = 48000, 1024, 1024, 32
FREQS = np.linspace(20.0, 22000.0, NFREQ)
F32 = np.float32
TINY = np.finfo(np.float32).tiny


def _frame_seconds(n_frames):
    """Seconds spanned on the frame grid by n_frames (grid step = HOP/SR)."""
    return n_frames * HOP / SR


def _make_spec(basename, dt_start, n_frames, seed):
    rng = np.random.default_rng(seed)
    power = rng.uniform(1e-10, 8.0, (NFREQ, n_frames)).astype(F32)
    return (basename, pd.Timestamp(dt_start), power)


def _seed_store(cache_dir, specs):
    """Write each spec as one complete shard (uneven appends to seam-test)."""
    for basename, dt_start, power in specs:
        path = os.path.join(stft_dir_for(cache_dir), shard_name(basename, 0))
        w = StftShardWriter(
            path, FREQS, SR, HOP, WIN, dt_start.to_pydatetime(),
            channel=0, serial="9999", time_chunk_frames=53,
        )
        i = 0
        for step in (37, 5, 211, 999999):
            if i >= power.shape[1]:
                break
            k = min(step, power.shape[1] - i)
            w.append(power[:, i:i + k])
            i += k
        w.finalise()


def _legacy_render_windows(specs, ref, time_bins, time_chunk):
    """Frozen pre-refactor algorithm = the oracle. specs MUST already be in
    (t_start, shard_path) order so keep-first dedup matches the store."""
    cols = [f"{float(f):.2f}Hz" for f in FREQS]
    frames = []
    for _, dt_start, power in specs:
        safe = np.maximum(power.astype(F32), TINY)
        pdb = (10.0 * np.log10(safe / F32(ref ** 2))).astype(F32)
        times = frame_datetimes(dt_start, power.shape[1], WIN, HOP, SR)
        frames.append(pd.DataFrame(pdb.T, index=times, columns=cols))

    matrix = pd.concat(frames).sort_index(kind="stable")
    matrix = matrix[~matrix.index.duplicated(keep="first")]

    if time_bins and len(matrix) > 1:
        duration = matrix.index.max() - matrix.index.min()
        step = pd.Timedelta(seconds=max(1.0, duration.total_seconds() / time_bins))
        diffs = matrix.index.to_series().diff().dropna()
        positive = diffs[diffs > pd.Timedelta(0)]
        native = positive.median() if len(positive) else pd.Timedelta(0)
        if step > native:
            matrix = matrix.resample(step).mean().dropna(how="all")

    if time_chunk is None:
        return [(matrix.index.min(), matrix.index.max(), matrix)]
    out = []
    for _, group in matrix.resample(time_chunk):
        if group.empty:
            continue
        out.append((group.index.min(), group.index.max(), group))
    return out


# (label, specs_builder, time_bins, time_chunk)
def _scenarios(dt0):
    n1 = 600
    contiguous_dt = dt0 + pd.Timedelta(seconds=_frame_seconds(n1))
    gapped_dt = dt0 + pd.Timedelta(seconds=_frame_seconds(n1) + 5.0)
    return [
        (
            "single_downsample_10s",
            [_make_spec("9999.a.wav", dt0, n1, 1)],
            60, "10s",
        ),
        (
            "contiguous_pair_5s",
            [_make_spec("9999.a.wav", dt0, n1, 1),
             _make_spec("9999.b.wav", contiguous_dt, 400, 2)],
            80, "5s",
        ),
        (
            "gapped_pair_5s",
            [_make_spec("9999.a.wav", dt0, n1, 1),
             _make_spec("9999.b.wav", gapped_dt, 400, 2)],
            80, "5s",
        ),
        (
            "overlap_dedup_keep_first",  # full re-recording at same start
            [_make_spec("9999.a.wav", dt0, n1, 1),
             _make_spec("9999.b.wav", dt0, n1, 99)],
            60, "10s",
        ),
        (
            "whole_extent_none",
            [_make_spec("9999.a.wav", dt0, n1, 1)],
            60, None,
        ),
        (
            "native_no_downsample",
            [_make_spec("9999.a.wav", dt0, 200, 1)],
            None, "1s",
        ),
    ]


@pytest.mark.parametrize(
    "label,specs,time_bins,time_chunk",
    _scenarios(pd.Timestamp("2026-05-21 08:04:17")),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_streaming_render_matches_legacy(
    label, specs, time_bins, time_chunk, test_config, tmp_path,
):
    ref = float(test_config.pipeline.reference_pressure_pa)
    # Order specs by (t_start, shard_path) to mirror the store's shard order.
    specs = sorted(specs, key=lambda s: (s[1], shard_name(s[0], 0)))

    _seed_store(tmp_path, specs)
    ctx = {"pipeline_config": test_config, "cache_dir": str(tmp_path)}

    oracle = _legacy_render_windows(specs, ref, time_bins, time_chunk)
    candidate = list(iter_stft_windows(
        ctx, time_chunk=time_chunk, time_bins=time_bins,
    ))

    assert len(candidate) == len(oracle), (
        f"{label}: window count {len(candidate)} != {len(oracle)}"
    )
    for (ows, owe, odf), (cws, cwe, cdf) in zip(oracle, candidate):
        assert cws == ows and cwe == owe, f"{label}: window bounds differ"
        assert list(cdf.columns) == list(odf.columns), f"{label}: columns differ"
        assert cdf.index.equals(odf.index), f"{label}: bin labels differ"
        # The rendered array imshow receives is the transpose.
        np.testing.assert_array_equal(
            cdf.T.values, odf.T.values,
            err_msg=f"{label}: rendered array not bit-identical",
        )


def test_overlap_dedup_actually_keeps_first(test_config, tmp_path):
    """Guard: the dedup scenario must really differ between the two files,
    so keep-first is exercised rather than trivially passing."""
    dt0 = pd.Timestamp("2026-05-21 08:04:17")
    a = _make_spec("9999.a.wav", dt0, 600, 1)
    b = _make_spec("9999.b.wav", dt0, 600, 99)
    assert not np.array_equal(a[2], b[2])
    specs = sorted([a, b], key=lambda s: (s[1], shard_name(s[0], 0)))
    assert specs[0][0] == "9999.a.wav"  # 'a' shard sorts first → kept

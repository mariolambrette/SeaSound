"""
tests/test_stft_store.py

Stage 3a gates for the STFT shard store (refactor plan §9 test 5, plus
the D8 timestamp convention and crash-safety behaviours the store
guarantees on its own, independent of the pipeline wiring that arrives
in Stage 3b).

The store is array-level — no audio is involved. Shards are written
from deterministic synthetic power arrays with known frame datetimes.
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from seasound.core.exceptions import StftStoreError
from seasound.loader.stft_store import (
    MANIFEST_COLUMNS,
    StftShardWriter,
    StftStore,
    frame_datetimes,
    load_manifest,
    rebuild_manifest_rows,
    shard_name,
    stft_dir_for,
    write_manifest,
)

# "Nice" STFT geometry: frame times = (1000 + 1000k)/8000 = 0.125(1+k) s,
# so a file starting 0.5 s after another has exactly frame-aligned
# timestamps — the dedup test needs truly coincident frames.
SR, WIN, HOP = 8000, 2000, 1000
N_FREQ = 16
FREQS = np.linspace(10.0, 3500.0, N_FREQ)


def _power(n_frames, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.1, 5.0, (N_FREQ, n_frames)).astype(np.float32)


def _write_shard(cache_dir, basename, dt_start, power, *, channel=0,
                 freqs=FREQS, block_sizes=(7, 5, 11), serial="9999"):
    """Write one shard in uneven blocks; return its manifest row."""
    path = os.path.join(stft_dir_for(cache_dir), shard_name(basename, channel))
    writer = StftShardWriter(
        path, freqs, SR, HOP, WIN, dt_start,
        channel=channel, serial=serial, time_chunk_frames=7,
    )
    i = 0
    sizes = list(block_sizes)
    while i < power.shape[1]:
        k = min(sizes[0] if sizes else 9, power.shape[1] - i)
        writer.append(power[:, i:i + k])
        i += k
        if sizes:
            sizes.pop(0)
    return writer.finalise()


@pytest.fixture
def two_file_store(tmp_path):
    """Two contiguous-ish shards: 30 frames from 12:00:00 and 20 frames
    from 12:05:00."""
    cache_dir = str(tmp_path)
    p1 = _power(30, seed=1)
    p2 = _power(20, seed=2)
    d1 = datetime(2026, 1, 1, 12, 0, 0)
    d2 = datetime(2026, 1, 1, 12, 5, 0)
    rows = [
        _write_shard(cache_dir, "9999.260101120000.wav", d1, p1),
        _write_shard(cache_dir, "9999.260101120500.wav", d2, p2),
    ]
    return cache_dir, (d1, p1), (d2, p2), rows


class TestRoundTrip:
    """..."""

    def test_full_extent_roundtrip_bit_exact(self, two_file_store): #pylint: disable=redefined-outer-name
        """..."""
        cache_dir, (d1, p1), (d2, p2), _ = two_file_store
        store = StftStore(cache_dir, channel=0)

        freqs, times, power = store.read()
        np.testing.assert_array_equal(freqs, FREQS)
        assert power.dtype == np.float32
        np.testing.assert_array_equal(
            power, np.concatenate([p1, p2], axis=1),
        )
        expected_times = frame_datetimes(d1, 30, WIN, HOP, SR).append(
            frame_datetimes(d2, 20, WIN, HOP, SR)
        )
        assert times.equals(expected_times)

    def test_timestamps_match_legacy_convention_to_the_ns(self, two_file_store): #pylint: disable=redefined-outer-name
        """D8: store datetimes == datetime_start + times_s under the
        exact pandas conversion build_stft_matrix uses (== not allclose)."""
        cache_dir, (d1, _), _, _ = two_file_store
        store = StftStore(cache_dir, channel=0)
        _, times, _ = store.read(t1=pd.Timestamp(d1) + pd.Timedelta(seconds=4))

        legacy_times_s = (WIN / 2 + HOP * np.arange(len(times))) / SR
        legacy_dts = pd.Timestamp(d1) + pd.to_timedelta(legacy_times_s, unit="s")
        assert times.equals(pd.DatetimeIndex(legacy_dts))

    def test_extent_from_manifest_only(self, two_file_store): #pylint: disable=redefined-outer-name
        """..."""
        cache_dir, (d1, _), (d2, p2), _ = two_file_store
        store = StftStore(cache_dir, channel=0)
        t0, t1 = store.extent()
        assert t0 == frame_datetimes(d1, 1, WIN, HOP, SR)[0]
        assert t1 == frame_datetimes(d2, p2.shape[1], WIN, HOP, SR)[-1]


class TestWindowedReads:
    """..."""

    def test_window_spanning_two_files(self, two_file_store): #pylint: disable=redefined-outer-name
        """..."""
        cache_dir, (d1, p1), (d2, p2), _ = two_file_store
        store = StftStore(cache_dir, channel=0)

        # last 5 frames of file 1 through first 3 frames of file 2
        dts1 = frame_datetimes(d1, 30, WIN, HOP, SR)
        dts2 = frame_datetimes(d2, 20, WIN, HOP, SR)
        t0, t1 = dts1[25], dts2[2]

        _, times, power = store.read(t0, t1)
        assert times.equals(dts1[25:].append(dts2[:3]))
        np.testing.assert_array_equal(
            power, np.concatenate([p1[:, 25:], p2[:, :3]], axis=1),
        )

    def test_window_inside_one_shard_across_chunks(self, two_file_store): #pylint: disable=redefined-outer-name
        """time_chunk_frames=7, so frames 5..16 span three zarr chunks."""
        cache_dir, (d1, p1), _, _ = two_file_store
        store = StftStore(cache_dir, channel=0)
        dts1 = frame_datetimes(d1, 30, WIN, HOP, SR)

        _, times, power = store.read(dts1[5], dts1[16])
        assert times.equals(dts1[5:17])
        np.testing.assert_array_equal(power, p1[:, 5:17])

    def test_arbitrary_subrange_between_frames(self, two_file_store): #pylint: disable=redefined-outer-name
        """t0/t1 that fall between frame datetimes: inclusive [t0, t1]
        semantics, matching DataFrame.loc slicing."""
        cache_dir, (d1, p1), _, _ = two_file_store
        store = StftStore(cache_dir, channel=0)
        dts1 = frame_datetimes(d1, 30, WIN, HOP, SR)

        t0 = dts1[3] + pd.Timedelta(milliseconds=1)   # excludes frame 3
        t1 = dts1[7] - pd.Timedelta(milliseconds=1)   # excludes frame 7
        _, times, power = store.read(t0, t1)
        assert times.equals(dts1[4:7])
        np.testing.assert_array_equal(power, p1[:, 4:7])

    def test_empty_window_returns_empty(self, two_file_store): #pylint: disable=redefined-outer-name
        """..."""
        cache_dir, (d1, _), (d2, _), _ = two_file_store
        store = StftStore(cache_dir, channel=0)
        # the silent gap between the two files
        t0 = pd.Timestamp(d1) + pd.Timedelta(seconds=10)
        t1 = pd.Timestamp(d2) - pd.Timedelta(seconds=10)
        freqs, times, power = store.read(t0, t1)
        assert len(times) == 0
        assert power.shape == (N_FREQ, 0)
        np.testing.assert_array_equal(freqs, FREQS)


class TestOverlapDedup:
    """..."""

    def test_keep_first_earlier_recording_wins(self, tmp_path):
        """Two files overlapping with exactly frame-aligned timestamps:
        the overlap region must carry the EARLIER shard's values."""
        cache_dir = str(tmp_path)
        d1 = datetime(2026, 1, 1, 12, 0, 0)
        # 0.5 s later = 4 frame periods (0.125 s each): frames coincide
        d2_offset_frames = 4
        d2 = datetime(2026, 1, 1, 12, 0, 0, 500000)

        p1 = _power(12, seed=11)
        p2 = _power(12, seed=22)
        _write_shard(cache_dir, "9999.a.wav", d1, p1)
        _write_shard(cache_dir, "9999.b.wav", d2, p2)

        store = StftStore(cache_dir, channel=0)
        _, times, power = store.read()

        dts1 = frame_datetimes(d1, 12, WIN, HOP, SR)
        dts2 = frame_datetimes(d2, 12, WIN, HOP, SR)
        # sanity: the overlap really is timestamp-coincident
        assert dts1[d2_offset_frames] == dts2[0]

        # union of frames: 12 from file 1, plus file 2's trailing 4
        assert len(times) == 16
        # overlap region [4..11] == file 1's values, not file 2's
        np.testing.assert_array_equal(power[:, :12], p1)
        np.testing.assert_array_equal(power[:, 12:], p2[:, 8:])


class TestCrashSafetyAndManifest:
    """..."""

    def test_incomplete_shard_is_invisible(self, two_file_store): #pylint: disable=redefined-outer-name
        """..."""
        cache_dir, _, _, _ = two_file_store
        # a writer that crashes before finalise()
        path = os.path.join(
            stft_dir_for(cache_dir), shard_name("9999.crashed.wav", 0)
        )
        w = StftShardWriter(
            path, FREQS, SR, HOP, WIN, datetime(2026, 1, 2), channel=0,
            serial="9999",
        )
        w.append(_power(5, seed=99))
        # no finalise()

        rows = rebuild_manifest_rows(stft_dir_for(cache_dir))
        assert all(r["shard_path"] != os.path.basename(path) for r in rows)

        store = StftStore(cache_dir, channel=0)
        assert len(store.shards) == 2
        _, times, _ = store.read()
        assert len(times) == 50  # 30 + 20, crashed shard absent

    def test_manifest_written_and_loaded(self, two_file_store): #pylint: disable=redefined-outer-name
        """..."""
        cache_dir, _, _, rows = two_file_store
        write_manifest(rows, stft_dir_for(cache_dir))
        manifest = load_manifest(stft_dir_for(cache_dir))
        assert manifest is not None
        assert list(manifest.columns) == MANIFEST_COLUMNS
        assert len(manifest) == 2

    def test_rebuilt_rows_equal_writer_rows(self, two_file_store): #pylint: disable=redefined-outer-name
        """§9 test 8d precursor: scanning shard attributes reproduces
        the rows the writers returned."""
        cache_dir, _, _, rows = two_file_store
        rebuilt = rebuild_manifest_rows(stft_dir_for(cache_dir))
        a = pd.DataFrame(rows, columns=MANIFEST_COLUMNS).sort_values("shard_path")
        b = pd.DataFrame(rebuilt, columns=MANIFEST_COLUMNS).sort_values("shard_path")
        pd.testing.assert_frame_equal(
            a.reset_index(drop=True), b.reset_index(drop=True)
        )

    def test_inconsistent_manifest_triggers_rebuild(self, two_file_store): #pylint: disable=redefined-outer-name
        """..."""
        cache_dir, _, _, rows = two_file_store
        bogus = dict(rows[0])
        bogus["shard_path"] = "9999.deleted.wav_ch0.zarr"
        write_manifest(rows + [bogus], stft_dir_for(cache_dir))

        store = StftStore(cache_dir, channel=0)
        assert len(store.shards) == 2  # rebuilt from attrs, bogus row gone
        _, times, _ = store.read()
        assert len(times) == 50

    def test_missing_manifest_rebuilds(self, two_file_store): #pylint: disable=redefined-outer-name
        """..."""
        cache_dir, _, _, _ = two_file_store
        # never written at all
        store = StftStore(cache_dir, channel=0)
        assert len(store.shards) == 2


class TestWriterContract:
    """..."""

    def test_requires_datetime_start(self, tmp_path):
        """..."""
        with pytest.raises(StftStoreError):
            StftShardWriter(
                str(tmp_path / "x.zarr"), FREQS, SR, HOP, WIN, None, #type: ignore
                channel=0, serial="9999",
            )

    def test_block_shape_validated(self, tmp_path):
        """..."""
        w = StftShardWriter(
            str(tmp_path / "x.zarr"), FREQS, SR, HOP, WIN,
            datetime(2026, 1, 1), channel=0, serial="9999",
        )
        with pytest.raises(StftStoreError):
            w.append(np.zeros((N_FREQ + 1, 4), dtype=np.float32))

    def test_append_after_finalise_raises(self, tmp_path):
        """..."""
        w = StftShardWriter(
            str(tmp_path / "x.zarr"), FREQS, SR, HOP, WIN,
            datetime(2026, 1, 1), channel=0, serial="9999",
        )
        w.append(_power(3, seed=1))
        w.finalise()
        with pytest.raises(StftStoreError):
            w.append(_power(1, seed=2))

    def test_zero_frame_shard_is_complete_but_empty(self, tmp_path):
        """..."""
        cache_dir = str(tmp_path)
        path = os.path.join(stft_dir_for(cache_dir), shard_name("9999.e.wav", 0))
        w = StftShardWriter(
            path, FREQS, SR, HOP, WIN, datetime(2026, 1, 1),
            channel=0, serial="9999",
        )
        row = w.finalise()
        assert row["complete"] is True and row["n_frames"] == 0
        store = StftStore(cache_dir, channel=0)
        assert store.extent() == (None, None)
        _, times, _ = store.read()
        assert len(times) == 0

    def test_rewrite_replaces_crash_artifact(self, tmp_path):
        """Opening a writer at an existing (incomplete) shard path
        recreates it — the resume rule's rebuild behaviour."""
        cache_dir = str(tmp_path)
        path = os.path.join(stft_dir_for(cache_dir), shard_name("9999.r.wav", 0))
        w1 = StftShardWriter(
            path, FREQS, SR, HOP, WIN, datetime(2026, 1, 1),
            channel=0, serial="9999",
        )
        w1.append(_power(9, seed=5))  # crash: no finalise

        p = _power(6, seed=6)
        w2 = StftShardWriter(
            path, FREQS, SR, HOP, WIN, datetime(2026, 1, 1),
            channel=0, serial="9999",
        )
        w2.append(p)
        w2.finalise()

        store = StftStore(cache_dir, channel=0)
        _, times, power = store.read()
        assert len(times) == 6
        np.testing.assert_array_equal(power, p)


class TestTransientLockRetries:
    """The Windows mitigation: zarr metadata replaces racing with
    antivirus/indexer scans surface as transient PermissionError.
    Verified platform-independently against the helper itself."""

    def test_retries_through_transient_permission_errors(self, monkeypatch):
        """..."""
        from seasound.loader import stft_store as mod
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise PermissionError("[WinError 5] Access is denied")
            return "ok"

        assert mod._retry_transient_locks(flaky, what="testing", path="x") == "ok" #pylint: disable=protected-access
        assert calls["n"] == 3

    def test_persistent_lock_raises_store_error(self, monkeypatch):
        """..."""
        from seasound.loader import stft_store as mod
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

        def always_locked():
            raise PermissionError("[WinError 5] Access is denied")

        with pytest.raises(StftStoreError):
            mod._retry_transient_locks(always_locked, what="testing", path="x") #pylint: disable=protected-access

    def test_non_permission_errors_propagate_immediately(self, monkeypatch):
        """..."""
        from seasound.loader import stft_store as mod
        monkeypatch.setattr(mod.time, "sleep", lambda _s: None)

        calls = {"n": 0}

        def broken():
            calls["n"] += 1
            raise ValueError("a real bug, not a lock")

        with pytest.raises(ValueError):
            mod._retry_transient_locks(broken, what="testing", path="x") #pylint: disable=protected-access
        assert calls["n"] == 1  # no retries for non-lock errors


class TestChannelsAndFreqAxis:
    """..."""
    def test_channel_filtering(self, tmp_path):
        """..."""
        cache_dir = str(tmp_path)
        d = datetime(2026, 1, 1, 12, 0, 0)
        p0, p1 = _power(8, seed=31), _power(8, seed=32)
        _write_shard(cache_dir, "9999.m.wav", d, p0, channel=0)
        _write_shard(cache_dir, "9999.m.wav", d, p1, channel=1)

        _, _, got0 = StftStore(cache_dir, channel=0).read()
        _, _, got1 = StftStore(cache_dir, channel=1).read()
        np.testing.assert_array_equal(got0, p0)
        np.testing.assert_array_equal(got1, p1)

    def test_mismatched_frequency_axes_raise(self, tmp_path):
        """..."""
        cache_dir = str(tmp_path)
        d1 = datetime(2026, 1, 1, 12, 0, 0)
        d2 = datetime(2026, 1, 1, 12, 5, 0)
        _write_shard(cache_dir, "9999.f1.wav", d1, _power(5, seed=41))
        _write_shard(
            cache_dir, "9999.f2.wav", d2, _power(5, seed=42),
            freqs=FREQS + 1.0,
        )
        store = StftStore(cache_dir, channel=0)
        with pytest.raises(StftStoreError):
            store.read()

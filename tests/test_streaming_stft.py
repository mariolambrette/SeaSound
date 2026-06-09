"""
tests/test_streaming_stft.py

Stage 3b gates (refactor plan §9 tests 3 and 4) plus the pipeline
wiring behaviours: the streaming path must produce STFT shards that
are frame-for-frame identical to the legacy per-file
``compute_stft_power`` output, with the carry buffer reset between
files and window-centre timestamps preserved to the nanosecond; npz
production moves to the legacy path only; the coordinator writes the
shard manifest after the run.
"""

import copy
import glob
import os
from functools import partial

import numpy as np
import pandas as pd
import pytest

from seasound.core.pipeline import _process_one_file, run_loading
from seasound.loader.calibration import load_calibration
from seasound.loader.filename_parsers import get_parser
from seasound.loader.stft import StftAccumulator, compute_stft_power
from seasound.loader.stft_store import StftStore, load_manifest, stft_dir_for
from tests.golden import (
    assert_candidate_matches_legacy_stft,
    assert_stft_entries_identical,
    legacy_stft_entries,
    streamed_stft_entries,
)


def _streaming_cfg(golden_config, **pipeline_overrides):
    cfg = copy.deepcopy(golden_config)
    cfg.pipeline.streaming_enabled = True
    # Force STFT production via the resolver override (refactor §7). The
    # streaming path is resolver-driven; the legacy stft_cache_enabled
    # flag no longer triggers STFT there.
    cfg.pipeline.stft_enabled = True
    for key, value in pipeline_overrides.items():
        setattr(cfg.pipeline, key, value)
    return cfg


class TestSeamCorrectness:
    """§9 test 3: streamed STFT == legacy per-file compute_stft_power,
    frame-for-frame, at several block sizes."""

    @pytest.mark.parametrize("block_seconds", [7, 60])
    def test_streamed_equals_legacy_awkward_file(
        self, awkward_wav, golden_config, block_seconds
    ):
        # 13 s file: block=7 puts seams at non-frame-aligned sample
        # positions (carry exercised); block=60 covers the whole file
        # in one push (single-block degenerate case).
        assert_candidate_matches_legacy_stft(
            partial(streamed_stft_entries, block_seconds=block_seconds),
            awkward_wav,
            golden_config,
            context=f"awkward, block={block_seconds}s",
        )

    def test_streamed_equals_legacy_fractional_file(
        self, fractional_wav, golden_config
    ):
        # 13.5 s file: the trailing half second exercises both the
        # reader's partial-bin drop and the accumulator's trailing
        # carry discard in the same run.
        assert_candidate_matches_legacy_stft(
            partial(streamed_stft_entries, block_seconds=7),
            fractional_wav,
            golden_config,
            context="fractional, block=7s",
        )

    def test_carry_resets_between_files(
        self, gapped_wav_pair, golden_config, tmp_path
    ):
        """Two files into one store must yield exactly the union of the
        two per-file frame sets — no fabricated cross-file frame, no
        leaked carry."""
        _, (wav1, wav2) = gapped_wav_pair
        cfg = _streaming_cfg(golden_config, cache_base_matrix=False)
        cal_df = load_calibration(cfg.calibration)
        parser = get_parser(cfg.input)
        cache_dir = str(tmp_path / "cache")

        for wav in (wav1, wav2):
            _process_one_file(wav, cfg, cal_df, cache_dir, parser)

        legacy1 = legacy_stft_entries(wav1, golden_config)[0]
        legacy2 = legacy_stft_entries(wav2, golden_config)[0]

        freqs, times, power = StftStore(cache_dir, channel=0).read()

        n1 = legacy1["power"].shape[1]
        n2 = legacy2["power"].shape[1]
        assert power.shape[1] == n1 + n2, (
            "frame count is not the union of the per-file frame sets"
        )
        np.testing.assert_array_equal(power[:, :n1], legacy1["power"])
        np.testing.assert_array_equal(power[:, n1:], legacy2["power"])
        np.testing.assert_array_equal(freqs, legacy1["freqs_hz"])

        expected_times = pd.DatetimeIndex(
            pd.Timestamp(legacy1["datetime_start"])
            + pd.to_timedelta(legacy1["times_s"], unit="s")
        ).append(pd.DatetimeIndex(
            pd.Timestamp(legacy2["datetime_start"])
            + pd.to_timedelta(legacy2["times_s"], unit="s")
        ))
        assert times.equals(expected_times)


class TestTimestampConvention:
    """§9 test 4, end-to-end: shard frame datetimes equal
    datetime_start + the legacy scipy times_s, to the nanosecond,
    through the full streaming chain."""

    def test_store_datetimes_equal_legacy_scipy_times(
        self, awkward_wav, golden_config, tmp_path
    ):
        legacy = legacy_stft_entries(awkward_wav, golden_config)[0]

        cfg = _streaming_cfg(golden_config, cache_base_matrix=False)
        cal_df = load_calibration(cfg.calibration)
        cache_dir = str(tmp_path / "cache")
        _process_one_file(
            awkward_wav, cfg, cal_df, cache_dir, get_parser(cfg.input)
        )

        _, times, _ = StftStore(cache_dir, channel=0).read()
        expected = pd.DatetimeIndex(
            pd.Timestamp(legacy["datetime_start"])
            + pd.to_timedelta(legacy["times_s"], unit="s")
        )
        assert times.equals(expected)  # nanosecond equality, not allclose


class TestPipelineWiring:
    """Decision 1: shards replace npz on the streaming path; the legacy
    path is untouched; the coordinator writes the manifest."""

    def test_streaming_writes_shards_not_npz(
        self, awkward_wav, golden_config, tmp_path
    ):
        cfg = _streaming_cfg(golden_config)
        cal_df = load_calibration(cfg.calibration)
        cache_dir = str(tmp_path / "cache")
        _process_one_file(
            awkward_wav, cfg, cal_df, cache_dir, get_parser(cfg.input)
        )

        assert not glob.glob(os.path.join(cache_dir, "*_stft.npz")), (
            "streaming run must not produce legacy npz STFT caches"
        )
        shards = glob.glob(os.path.join(stft_dir_for(cache_dir), "*.zarr"))
        assert len(shards) == 1

        # worker never writes the manifest (coordinator's job, D8)
        assert load_manifest(stft_dir_for(cache_dir)) is None

    def test_legacy_path_writes_npz_not_shards(
        self, awkward_wav, golden_config, tmp_path
    ):
        cfg = copy.deepcopy(golden_config)
        cfg.pipeline.streaming_enabled = False
        cfg.pipeline.stft_cache_enabled = True
        cal_df = load_calibration(cfg.calibration)
        cache_dir = str(tmp_path / "cache")
        os.makedirs(cache_dir, exist_ok=True)
        _process_one_file(
            awkward_wav, cfg, cal_df, cache_dir, get_parser(cfg.input)
        )

        assert glob.glob(os.path.join(cache_dir, "*_stft.npz")), (
            "legacy run must still produce npz STFT caches"
        )
        assert not os.path.isdir(stft_dir_for(cache_dir))

    def test_stft_dtype_flows_to_shard(
        self, awkward_wav, golden_config
    ):
        """float16 config: streamed shard values equal the legacy
        float16-cast entries (identical float32 → identical float16)."""
        cfg16 = copy.deepcopy(golden_config)
        cfg16.pipeline.stft_dtype = "float16"

        expected = legacy_stft_entries(awkward_wav, cfg16)
        actual = streamed_stft_entries(awkward_wav, cfg16, block_seconds=7)

        assert actual[0]["power"].dtype == np.float16
        assert_stft_entries_identical(expected, actual, context="float16")

    def test_run_loading_writes_manifest_and_resume_preserves_it(
        self, gapped_wav_pair, golden_config
    ):
        """The coordinator writes manifest.parquet after the pool; on a
        resumed run that skips every file, the manifest still lists all
        shards on disk (the attribute-scan property)."""
        directory, _ = gapped_wav_pair

        cfg = _streaming_cfg(golden_config)
        cfg.input.path = directory
        cfg.pipeline.resume = False
        cfg.pipeline.workers = 1
        cfg.output.directory = os.path.join(directory, "out")
        run_loading(cfg)

        cache_dir = os.path.join(cfg.output.directory, "cache")
        manifest = load_manifest(stft_dir_for(cache_dir))
        assert manifest is not None and len(manifest) == 2
        assert bool(manifest["complete"].all())

        # Resumed run: base-matrix parquets exist, so both files are
        # skipped — yet the manifest must still cover both shards.
        cfg2 = copy.deepcopy(cfg)
        cfg2.pipeline.resume = True
        run_loading(cfg2)
        manifest2 = load_manifest(stft_dir_for(cache_dir))
        assert manifest2 is not None and len(manifest2) == 2


class TestReaderTail:
    """read_tail(): the fractional remainder past the last whole bin,
    added so the STFT producer can match the legacy full-file compute
    (the base-matrix path never consumes it)."""

    def test_tail_is_exact_trailing_samples(self, fractional_wav, golden_config):
        sf = pytest.importorskip("soundfile")
        from seasound.loader.reader import AudioBlockReader

        whole, sr = sf.read(fractional_wav, dtype="float32")
        with AudioBlockReader(
            fractional_wav, golden_config.input, bin_seconds=1
        ) as reader:
            for _ in reader.blocks(7):
                pass
            tail = reader.read_tail()
        assert tail is not None
        np.testing.assert_array_equal(tail, whole[reader.n_bins * sr:])

    def test_tail_none_when_file_divides_evenly(self, awkward_wav, golden_config):
        from seasound.loader.reader import AudioBlockReader

        with AudioBlockReader(
            awkward_wav, golden_config.input, bin_seconds=1
        ) as reader:
            for _ in reader.blocks(7):
                pass
            assert reader.read_tail() is None

    def test_tail_respects_start_trim(self, fractional_wav, golden_config):
        sf = pytest.importorskip("soundfile")
        from seasound.loader.reader import AudioBlockReader

        cfg = copy.deepcopy(golden_config)
        cfg.input.per_file_trim_start_s = 3.0

        whole, sr = sf.read(fractional_wav, dtype="float32")
        with AudioBlockReader(
            fractional_wav, cfg.input, bin_seconds=1
        ) as reader:
            for _ in reader.blocks(7):
                pass
            tail = reader.read_tail()
        # 13.5 s - 3 s trim = 10.5 s usable → 10 bins + 0.5 s tail
        assert reader.n_bins == 10
        assert tail is not None
        np.testing.assert_array_equal(
            tail, whole[3 * sr + reader.n_bins * sr:]
        )


class TestAccumulatorContract:
    """Unit-level pins for the overlap-save carry (small, fast
    geometry: SR 8000, win 2000, hop 1000, nfft 2048)."""

    SR, NFFT, WIN, HOP = 8000, 2048, 2000, 1000

    def _full(self, audio):
        return compute_stft_power(
            audio, self.SR, self.NFFT, self.WIN, self.HOP,
            "hann", 10.0, 3500.0,
        )

    def _acc(self):
        return StftAccumulator(
            self.SR, self.NFFT, self.WIN, self.HOP, "hann", 10.0, 3500.0,
        )

    @pytest.mark.parametrize("push_sizes", [
        [100000],                       # single push
        [8000] * 13,                    # whole-second blocks
        [500, 1500, 8000, 3333, 7, 2000],  # awkward, incl. sub-win pushes
    ])
    def test_bit_identical_to_full_compute(self, push_sizes):
        rng = np.random.default_rng(13)
        audio = rng.standard_normal(self.SR * 11 + 357).astype(np.float32)
        freqs_full, _, power_full = self._full(audio)

        acc = self._acc()
        parts, i, sizes = [], 0, list(push_sizes)
        while i < len(audio):
            k = sizes.pop(0) if sizes else self.SR
            out = acc.push(audio[i:i + k])
            i += k
            if out is not None:
                parts.append(out)
        n_total = acc.finalise()

        streamed = np.concatenate(parts, axis=1)
        assert n_total == power_full.shape[1] == streamed.shape[1]
        np.testing.assert_array_equal(streamed, power_full)
        np.testing.assert_array_equal(acc.freqs_hz, freqs_full)

    def test_caller_block_reuse_is_safe(self):
        """The carry must be a copy: mutating the pushed block's
        storage afterwards (as the in-place calibration loop does with
        its discarded blocks) must not corrupt subsequent frames."""
        rng = np.random.default_rng(29)
        audio = rng.standard_normal(self.SR * 3).astype(np.float32)
        _, _, power_full = self._full(audio)

        acc = self._acc()
        parts = []
        scratch = np.empty(self.SR, dtype=np.float32)
        for i in range(3):
            scratch[:] = audio[i * self.SR:(i + 1) * self.SR]
            out = acc.push(scratch)
            if out is not None:
                parts.append(out)
            scratch[:] = np.nan  # simulate the block being recycled
        acc.finalise()
        np.testing.assert_array_equal(
            np.concatenate(parts, axis=1), power_full
        )

    def test_file_shorter_than_window_yields_no_frames(self):
        acc = self._acc()
        assert acc.push(np.zeros(self.WIN - 1, dtype=np.float32)) is None
        assert acc.finalise() == 0
        assert acc.freqs_hz is None

    def test_push_after_finalise_raises(self):
        acc = self._acc()
        acc.push(np.zeros(self.WIN, dtype=np.float32))
        acc.finalise()
        with pytest.raises(RuntimeError):
            acc.push(np.zeros(10, dtype=np.float32))
        with pytest.raises(RuntimeError):
            acc.finalise()

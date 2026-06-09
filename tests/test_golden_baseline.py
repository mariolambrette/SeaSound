"""
tests/test_golden_baseline.py

Frozen-snapshot regression gate for the streaming path (refactor plan §9,
option A — a small representative set rather than the full migration matrix;
the structural properties are covered legacy-free by test_streaming_*.py and
test_streaming_stft_render.py).

Each case asserts the streaming output equals a committed golden snapshot.
While the legacy path still exists it also asserts the snapshot is faithful
to live legacy. The snapshots are seeded once, deliberately:

    SEASOUND_UPDATE_GOLDEN=1 python -m pytest tests/test_golden_baseline.py

In that mode each case computes both paths, asserts legacy == streamed (the
capture-time cross-check, so the committed snapshot is provably the legacy
baseline), and writes the legacy output to tests/golden_fixtures/. Commit
those files. Normal runs then compare streamed (and, for now, legacy) against
the committed snapshot.

Stage 6 removes the legacy path: delete the two lines marked
"# Stage 6: delete" in each test and the regen branch's legacy cross-check;
the streamed-vs-snapshot assertion is the surviving gate.
"""

import copy
import os

import pytest

from tests.golden import (
    assert_artifacts_identical,
    assert_stft_entries_identical,
    legacy_base_matrix_artifacts,
    legacy_stft_entries,
    streamed_base_matrix_artifacts,
    streamed_stft_entries,
)
from tests.golden_io import (
    load_base_matrix_snapshot,
    load_stft_snapshot,
    save_base_matrix_snapshot,
    save_stft_snapshot,
)

_UPDATE = os.environ.get("SEASOUND_UPDATE_GOLDEN") == "1"


# --- config recipes (applied to a deepcopy of the named base config) -------

def _identity(cfg):
    return cfg


def _auto(cfg):
    cfg.input.channel_strategy = "auto"
    return cfg


def _float16(cfg):
    cfg.pipeline.stft_dtype = "float16"
    return cfg


# (snapshot_id, wav_fixture, base_config_fixture, mutate)
BASE_CASES = [
    # mono base matrix with a fractional tail (read_tail / end-of-file drop)
    ("base_fractional", "fractional_wav", "golden_config", _identity),
    # multi-channel base matrix (auto strategy → one snapshot per channel)
    ("base_stereo_auto", "synthetic_stereo_wav", "golden_config", _auto),
    # seek-based start trim + datetime shift (test_config defaults to 3 s)
    ("base_trim3", "awkward_wav", "test_config", _identity),
]

# (snapshot_id, wav_fixture, base_config_fixture, mutate, block_seconds)
STFT_CASES = [
    # block=7 puts seams at non-frame-aligned positions → carry exercised
    ("stft_awkward_b7", "awkward_wav", "golden_config", _identity, 7),
    # float16 storage dtype flows through to the shard values
    ("stft_float16_b7", "awkward_wav", "golden_config", _float16, 7),
]


@pytest.mark.parametrize(
    "snapshot_id,wav_fixture,cfg_fixture,mutate",
    BASE_CASES,
    ids=[c[0] for c in BASE_CASES],
)
def test_base_matrix_golden(snapshot_id, wav_fixture, cfg_fixture, mutate, request):
    wav = request.getfixturevalue(wav_fixture)
    cfg = mutate(copy.deepcopy(request.getfixturevalue(cfg_fixture)))

    streamed = streamed_base_matrix_artifacts(wav, cfg)

    if _UPDATE:
        legacy = legacy_base_matrix_artifacts(wav, cfg)
        assert_artifacts_identical(
            legacy, streamed, context=f"{snapshot_id} capture cross-check"
        )
        save_base_matrix_snapshot(snapshot_id, legacy)
        return

    golden = load_base_matrix_snapshot(snapshot_id)
    assert_artifacts_identical(golden, streamed, context=f"{snapshot_id} streamed")
    # Stage 6: delete the next two lines (legacy oracle removed).
    legacy = legacy_base_matrix_artifacts(wav, cfg)
    assert_artifacts_identical(golden, legacy, context=f"{snapshot_id} faithfulness")


@pytest.mark.parametrize(
    "snapshot_id,wav_fixture,cfg_fixture,mutate,block_seconds",
    STFT_CASES,
    ids=[c[0] for c in STFT_CASES],
)
def test_stft_golden(
    snapshot_id, wav_fixture, cfg_fixture, mutate, block_seconds, request
):
    wav = request.getfixturevalue(wav_fixture)
    cfg = mutate(copy.deepcopy(request.getfixturevalue(cfg_fixture)))

    streamed = streamed_stft_entries(wav, cfg, block_seconds=block_seconds)

    if _UPDATE:
        legacy = legacy_stft_entries(wav, cfg)
        assert_stft_entries_identical(
            legacy, streamed, context=f"{snapshot_id} capture cross-check"
        )
        save_stft_snapshot(snapshot_id, legacy)
        return

    golden = load_stft_snapshot(snapshot_id)
    assert_stft_entries_identical(golden, streamed, context=f"{snapshot_id} streamed")
    # Stage 6: delete the next two lines (legacy oracle removed).
    legacy = legacy_stft_entries(wav, cfg)
    assert_stft_entries_identical(golden, legacy, context=f"{snapshot_id} faithfulness")

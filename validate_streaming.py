#!/usr/bin/env python
"""
Real-data validation harness for the streaming refactor (runbook close-out).

Runs three checks against a real deployment, while the legacy non-streaming
path still exists as the comparison baseline (this must be done before the
Stage-6 cleanup deletes it):

  1. Bit-identity (the hard correctness gate). For every file, the streaming
     path is compared against the legacy path frame-for-frame, reusing the
     same gate machinery the test suite trusts (tests/golden.py):
       - base matrix : streamed vs legacy, np.array_equal on values + index;
       - STFT        : shard-derived vs a fresh legacy compute_stft_power,
                       identical frequency axis, frame times, and power.
     Each file also reports its fractional tail (samples past the last whole
     base bin) so you can confirm the read_tail path actually fired on real
     data, not just on the synthetic fixture.

  2. Resume / manifest on real data. After a streaming run, every file is
     reported fully-cached (a resumed run would skip all of them), and the
     manifest rebuilds from shard attributes to the same row count.

  3. Memory. The streaming and legacy paths each run through run_loading under
     a process-tree RSS sampler; the headline number is peak *per-worker* RSS
     (memory is meant to be bounded per worker, independent of file length),
     alongside peak whole-tree RSS and wall-clock for each path.

Nothing here writes to your real cache or output directories — every run uses
a throwaway temp cache.

Usage (from the repo root):

    python validate_streaming.py --config config/your_config.yaml

    # quick smoke on the first few files:
    python validate_streaming.py --config config/your_config.yaml --limit 3

    # skip sections:
    python validate_streaming.py --config ... --skip-memory --skip-resume

The full report is printed and also written to <output.directory>/
streaming_validation_report.txt — attach that.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import threading
import time
import traceback

# Repo root + tests/ on the path so seasound.* and golden import cleanly.
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


class PeakRSS:
    """Sample the process tree's RSS on a background thread, recording the
    peak total (this process + all children) and the peak single child."""

    def __init__(self, interval_s: float = 0.05):
        import psutil  # local import: only needed for the memory section

        self._psutil = psutil
        self.interval_s = interval_s
        self.peak_total = 0
        self.peak_worker = 0
        self.peak_main = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        proc = self._psutil.Process()
        while not self._stop.is_set():
            try:
                main = proc.memory_info().rss
                self.peak_main = max(self.peak_main, main)
                total = main
                worker_peak = 0
                for child in proc.children(recursive=True):
                    try:
                        rss = child.memory_info().rss
                    except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                        continue
                    total += rss
                    worker_peak = max(worker_peak, rss)
                self.peak_total = max(self.peak_total, total)
                self.peak_worker = max(self.peak_worker, worker_peak)
            except (self._psutil.NoSuchProcess, self._psutil.AccessDenied):
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "PeakRSS":
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)


def _tail_samples(wav_path: str, config) -> tuple[int, int, float]:
    """(frames_after_trim, tail_samples, tail_seconds) for one file, matching
    the reader's trim + whole-bin convention. Best-effort; returns (-1,-1,-1)
    if soundfile can't read the header."""
    try:
        import soundfile as sf

        info = sf.info(wav_path)
        sr = info.samplerate
        trim = int(round(config.input.per_file_trim_start_s * sr))
        effective = max(0, info.frames - trim)
        bin_samples = int(round(config.pipeline.base_resolution_s * sr))
        if bin_samples <= 0:
            return effective, -1, -1.0
        tail = effective % bin_samples
        return effective, tail, tail / sr
    except Exception:
        return -1, -1, -1.0


# ---------------------------------------------------------------------------
# Section 1: per-file bit-identity
# ---------------------------------------------------------------------------

def run_identity(config, files, log) -> bool:
    from golden import (
        assert_candidate_matches_legacy_base_matrix,
        assert_candidate_matches_legacy_stft,
        streamed_base_matrix_artifacts,
        streamed_stft_entries,
    )

    log("")
    log("=" * 78)
    log("1. PER-FILE BIT-IDENTITY  (streaming vs legacy, frame-for-frame)")
    log("=" * 78)
    log(f"{'file':<34} {'tail(smp)':>9} {'tail(s)':>8} {'base':>6} {'stft':>6}")
    log("-" * 78)

    base_pass = stft_pass = 0
    tail_files = 0
    failures: list[str] = []

    for wav in files:
        name = os.path.basename(wav)
        _eff, tail, tail_s = _tail_samples(wav, config)
        if tail > 0:
            tail_files += 1

        try:
            assert_candidate_matches_legacy_base_matrix(
                streamed_base_matrix_artifacts, wav, config, context=name
            )
            base_ok = True
            base_pass += 1
        except Exception as exc:  # noqa: BLE001
            base_ok = False
            failures.append(f"[base] {name}: {exc}")

        try:
            assert_candidate_matches_legacy_stft(
                streamed_stft_entries, wav, config, context=name
            )
            stft_ok = True
            stft_pass += 1
        except Exception as exc:  # noqa: BLE001
            stft_ok = False
            failures.append(f"[stft] {name}: {exc}")

        log(
            f"{name:<34} {tail:>9} {tail_s:>8.3f} "
            f"{'PASS' if base_ok else 'FAIL':>6} "
            f"{'PASS' if stft_ok else 'FAIL':>6}"
        )

    n = len(files)
    log("-" * 78)
    log(f"base matrix : {base_pass}/{n} identical")
    log(f"STFT        : {stft_pass}/{n} identical")
    log(f"files exercising the fractional-tail (read_tail) path: {tail_files}/{n}")
    if tail_files == 0:
        log("  NOTE: no file had a fractional tail — read_tail was not exercised "
            "on this dataset.")
    if failures:
        log("")
        log("FAILURES:")
        for f in failures:
            log(f"  {f}")
    return base_pass == n and stft_pass == n


# ---------------------------------------------------------------------------
# Section 2: resume / manifest on real data
# ---------------------------------------------------------------------------

def run_resume(config, files, log) -> bool:
    from seasound.core.pipeline import _is_fully_cached, run_loading
    from seasound.core.substrates import resolve_producers
    from seasound.loader import stft_store

    log("")
    log("=" * 78)
    log("2. RESUME / MANIFEST  (real data)")
    log("=" * 78)

    resolved = resolve_producers(config)
    log(f"resolved producers: {', '.join(sorted(resolved))}")

    with tempfile.TemporaryDirectory(prefix="seasound_resume_") as cache_dir:
        cfg = _clone_with(config, {
            "pipeline.streaming_enabled": True,
            "pipeline.resume": False,
            "pipeline.cache_directory": cache_dir,
        })
        run_loading(cfg)

        fully = sum(
            1 for f in files
            if _is_fully_cached(f, cfg, cache_dir, resolved)
        )
        all_skipped = fully == len(files)
        log(f"fully-cached after run (a resume would skip): {fully}/{len(files)}")

        stft_dir = stft_store.stft_dir_for(cache_dir)
        rows = stft_store.rebuild_manifest_rows(stft_dir)
        manifest_path = stft_store.write_manifest(rows, stft_dir)
        os.remove(manifest_path)
        rebuilt = stft_store.rebuild_manifest_rows(stft_dir)
        manifest_ok = bool(rebuilt) and rebuilt == rows
        log(f"manifest rebuild from shard attributes: "
            f"{'OK' if manifest_ok else 'MISMATCH'} ({len(rebuilt)} rows)")

    ok = all_skipped and manifest_ok
    log(f"resume/manifest: {'PASS' if ok else 'FAIL'}")
    return ok


# ---------------------------------------------------------------------------
# Section 3: memory
# ---------------------------------------------------------------------------

def _profile_path(config, streaming: bool, files_limited, log) -> tuple[int, int, float]:
    from seasound.core.pipeline import run_loading

    with tempfile.TemporaryDirectory(prefix="seasound_mem_") as cache_dir:
        overrides = {
            "pipeline.streaming_enabled": streaming,
            "pipeline.resume": False,
            "pipeline.cache_directory": cache_dir,
        }
        # Legacy path produces STFT only when stft_cache_enabled (npz);
        # leave the user's flag as-is so the baseline is the real legacy run.
        cfg = _clone_with(config, overrides)
        t0 = time.time()
        with PeakRSS() as p:
            run_loading(cfg)
        wall = time.time() - t0
    label = "streaming" if streaming else "legacy"
    log(f"  {label:<10} peak/worker={_human_bytes(p.peak_worker):>10}  "
        f"peak/main={_human_bytes(p.peak_main):>10}  "
        f"peak/tree={_human_bytes(p.peak_total):>10}  wall={wall:6.1f}s")
    return p.peak_worker, p.peak_main, p.peak_total, wall


def run_memory(config, files, log) -> bool:
    log("")
    log("=" * 78)
    log("3. MEMORY  (run_loading under a process-tree RSS sampler)")
    log("=" * 78)
    log(f"  workers (config): {config.pipeline.workers} "
        f"(0 = auto, CPU-2); block_seconds: "
        f"{config.pipeline.streaming_block_seconds}")

    s_worker, s_main, s_tree, s_wall = _profile_path(config, True, files, log)
    l_worker, l_main, l_tree, l_wall = _profile_path(config, False, files, log)

    if s_worker and l_worker:
        log(f"  per-worker reduction : {l_worker / max(s_worker, 1):.1f}x "
            f"({_human_bytes(l_worker)} -> {_human_bytes(s_worker)})  <-- headline")
    else:
        log("  per-worker: 0 B on at least one path -> that run went serial "
            "(single process, no Pool children). On a multi-core box with "
            "many files it runs parallel and this populates.")
        log(f"  peak-process reduction : {l_main / max(s_main, 1):.1f}x "
            f"({_human_bytes(l_main)} -> {_human_bytes(s_main)})  "
            f"(main-process peak; the serial-mode stand-in for per-worker)")
    if s_tree and l_tree:
        log(f"  whole-tree reduction : {l_tree / max(s_tree, 1):.1f}x "
            f"({_human_bytes(l_tree)} -> {_human_bytes(s_tree)})")
    log("  (per-worker is the headline for the refactor's claim; whole-tree "
        "scales with worker count. Cross-check with profile_run.py for the "
        "authoritative process-tree number.)")
    return True


# ---------------------------------------------------------------------------
# Helpers + main
# ---------------------------------------------------------------------------

def _clone_with(config, overrides: dict):
    """A fresh PipelineConfig from the same YAML with dot-notation overrides,
    so each run is independent and the original config is never mutated."""
    from seasound.core.config import load_config

    return load_config(config._source_path, overrides)  # type: ignore[attr-defined]


def _bootstrap_analyses(log) -> None:
    """Populate the analysis registry by importing the concrete analysis
    modules, which self-register at import. A real run relies on this
    happening before resolve_producers; missing optional modules are skipped.
    """
    import importlib

    for name in ("spectrogram", "event_detection", "ltsa", "tob_levels",
                 "spectral_percentiles"):
        try:
            importlib.import_module(f"seasound.analysis.{name}")
        except ImportError:
            pass
    from seasound.analysis.registry import list_registered

    log(f"analysis registry after bootstrap: "
        f"{sorted(list_registered().keys()) or '(EMPTY)'}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Streaming refactor real-data validation")
    ap.add_argument("--config", required=True, help="path to the deployment YAML")
    ap.add_argument("--limit", type=int, default=0, help="validate only the first N files")
    ap.add_argument("--skip-identity", action="store_true")
    ap.add_argument("--skip-resume", action="store_true")
    ap.add_argument("--skip-memory", action="store_true")
    args = ap.parse_args()

    from seasound.core.config import load_config
    from seasound.core.pipeline import find_audio_files

    config = load_config(args.config)
    # Stash the source path so _clone_with can reload independently.
    config._source_path = args.config  # type: ignore[attr-defined]

    files = find_audio_files(config)
    if args.limit:
        files = files[: args.limit]
    if not files:
        print("No audio files found for the given config.", file=sys.stderr)
        return 2

    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    log(f"SeaSound streaming validation — {len(files)} file(s)")
    log(f"config: {args.config}")

    # --- Registry diagnostic (the integration check) ---------------------
    # resolve_producers (called inside run_loading at Stage 1) maps each
    # enabled analysis to its required substrates via the analysis registry.
    # The registry is populated only when the concrete analysis modules are
    # imported. If nothing in the normal import chain has done that by now,
    # a clean `seasound --config` run would resolve base_matrix only and
    # silently produce no STFT. Report the state *before* we bootstrap, so
    # this line tells you whether your real CLI runs are at risk.
    from seasound.analysis.registry import list_registered

    before = sorted(list_registered().keys())
    log(f"analysis registry BEFORE bootstrap (i.e. what a clean run would see "
        f"at Stage 1): {before or '(EMPTY)'}")
    if not before:
        log("  *** WARNING: empty. Nothing in the import chain registered an "
            "analysis. In a clean CLI run, resolve_producers would see this "
            "and resolve base_matrix only — no STFT. See the report notes. ***")
    _bootstrap_analyses(log)

    results: dict[str, bool] = {}
    if not args.skip_identity:
        results["identity"] = run_identity(config, files, log)
    if not args.skip_resume:
        results["resume"] = run_resume(config, files, log)
    if not args.skip_memory:
        results["memory"] = run_memory(config, files, log)

    log("")
    log("=" * 78)
    log("SUMMARY")
    log("=" * 78)
    for name, ok in results.items():
        log(f"  {name:<10} {'PASS' if ok else 'FAIL'}")

    report_dir = config.output.directory
    try:
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, "streaming_validation_report.txt")
        with open(report_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
        log(f"\nreport written to: {report_path}")
    except Exception as exc:  # noqa: BLE001
        log(f"\n(could not write report file: {exc})")

    # Identity is the hard gate; memory is informational.
    gate = all(
        results.get(k, True) for k in ("identity", "resume")
    )
    return 0 if gate else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(3)

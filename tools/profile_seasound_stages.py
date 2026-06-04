#!/usr/bin/env python3
"""
profile_seasound_stages.py — attribute per-file memory to SeaSound pipeline stages.

Runs ONE worker's worth of work (a single WAV file) through the exact stages
that _process_one_file() uses, measuring each stage's memory two ways:

  * peak process RSS during the stage (via memory_profiler) — this CAPTURES
    numpy/native allocations, so it is the number that matters for RAM sizing.
  * top Python allocation lines (via tracemalloc) — Python-side only, so it
    tells you WHERE in the code, but will under-report large numpy buffers.

Usage (run from anywhere, with the seasound package importable):

    python profile_seasound_stages.py --config path\\to\\config.yaml
    python profile_seasound_stages.py --config cfg.yaml --wav path\\to\\one_file.wav

If --wav is omitted, the first file matched by the config's input settings is used.

Requires: psutil, memory_profiler  (pip install psutil memory_profiler)
plus seasound and its own dependencies installed in the same environment.
"""
import argparse
import csv
import os
import sys
import tracemalloc

import psutil

try:
    from memory_profiler import memory_usage
except ImportError:
    sys.exit("memory_profiler not installed. Run: pip install memory_profiler")

# --- SeaSound internals (mirrors seasound/core/pipeline.py:_process_one_file) ---
try:
    from seasound.core.config import load_config
    from seasound.loader.reader import read_audio
    from seasound.loader.calibration import load_calibration, apply_calibration
    from seasound.loader.base_matrix import compute_base_matrix
    from seasound.loader.filename_parsers import get_parser
    from seasound.analysis.calculate_stft import get_stft_for_file
    from seasound.core.pipeline import find_audio_files
except ImportError as e:
    sys.exit(f"Could not import seasound ({e}). Install it in this environment first.")


def run_stage(name, func, *args, **kwargs):
    """Run one stage, returning (retval, record-dict of memory metrics)."""
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / 1e6
    snap0 = tracemalloc.take_snapshot()
    tracemalloc.reset_peak()
    peak_rss, ret = memory_usage((func, args, kwargs), interval=0.01, #type: ignore
                                 max_usage=True, retval=True)
    _, py_peak = tracemalloc.get_traced_memory()
    snap1 = tracemalloc.take_snapshot()
    top = snap1.compare_to(snap0, "lineno")[:5]
    top_lines = [f"{s.traceback[0].filename.split(os.sep)[-1]}:"
                 f"{s.traceback[0].lineno} (+{s.size_diff/1e6:.1f}MB)" for s in top]
    rec = {
        "stage": name,
        "rss_before_mb": round(rss_before, 1),
        "peak_rss_mb": round(peak_rss, 1),
        "transient_delta_mb": round(peak_rss - rss_before, 1),
        "py_traced_peak_mb": round(py_peak / 1e6, 1),
        "top_python_alloc_lines": " | ".join(top_lines),
    }
    return ret, rec


def main():
    """Main entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to SeaSound YAML config")
    ap.add_argument("--wav", default=None, help="single WAV to profile (default: first matched)")
    ap.add_argument("--outdir", default="profile_results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    config = load_config(args.config)

    # Resolve the single file to profile
    wav = args.wav
    if wav is None:
        files = find_audio_files(config)
        if not files:
            sys.exit("No audio files matched the config; pass one explicitly with --wav.")
        wav = files[0]
    print(f"Profiling stages for: {wav}\n")

    # Shared setup (mirrors _process_one_file / run_loading)
    cache_dir = config.pipeline.cache_directory or os.path.join(
        config.output.directory, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    parser = get_parser(config.input)
    cal_df = load_calibration(config.calibration)

    tracemalloc.start()
    records = []

    # Stage 0: optional STFT cache step (only if enabled in config)
    if config.pipeline.stft_cache_enabled:
        _, rec = run_stage("get_stft_for_file", get_stft_for_file, wav, config, cache_dir)
        records.append(rec)

    # Stage 1: read audio -> segments
    segments, rec = run_stage("read_audio", read_audio, wav, config.input, parser=parser)
    records.append(rec)
    if not segments:
        sys.exit("read_audio returned no segments.")
    segment = segments[0]   # homogeneous-worker assumption: profile one segment

    # Stage 2: calibrate
    (cal_out), rec = run_stage("apply_calibration", apply_calibration,
                               segment, cal_df, config.calibration)
    audio_pa, _calibrated = cal_out
    records.append(rec)

    # Stage 3: compute base matrix (TOB SPL)
    _matrix, rec = run_stage("compute_base_matrix", compute_base_matrix,
                             audio_pa, segment.sample_rate, config.pipeline)
    records.append(rec)

    tracemalloc.stop()

    # ---- report ----
    csv_path = os.path.join(args.outdir, "seasound_stage_memory.csv")
    with open(csv_path, "w", newline="") as f: #pylint: disable=unspecified-encoding
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    heaviest = max(records, key=lambda r: r["transient_delta_mb"])
    print(f"{'stage':<22}{'peak RSS':>12}{'+delta':>12}{'py-traced':>12}")
    print("-" * 58)
    for r in records:
        mark = "  <-- heaviest" if r is heaviest else ""
        print(f"{r['stage']:<22}{r['peak_rss_mb']:>10.1f}MB"
              f"{r['transient_delta_mb']:>10.1f}MB{r['py_traced_peak_mb']:>10.1f}MB{mark}")
    print("-" * 58)
    print(f"Heaviest stage by transient memory: {heaviest['stage']} "
          f"(+{heaviest['transient_delta_mb']:.1f} MB)\n")
    print("Top Python allocation lines per stage:")
    for r in records:
        print(f"  [{r['stage']}] {r['top_python_alloc_lines']}")
    print(f"\nCSV: {csv_path}")
    print("\nNote: 'peak RSS'/'+delta' include native numpy buffers (the real RAM cost).")
    print("'py-traced' is Python-side only and under-reports numpy; use it to locate code,")
    print("not to size memory. For a per-line Python-vs-native split, re-run under Scalene (WSL2).")


if __name__ == "__main__":
    main()

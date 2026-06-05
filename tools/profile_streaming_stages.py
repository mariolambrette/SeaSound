#!/usr/bin/env python3
"""
profile_streaming_stages.py — attribute per-file memory to the STREAMING
pipeline stages (refactor plan Stage 2+).

Companion to profile_seasound_stages.py, which profiles the legacy
decomposition (read_audio -> apply_calibration -> compute_base_matrix)
and is kept unchanged as the baseline reference. This script mirrors
_process_one_file_streaming() instead:

    [stft_npz_legacy]   only if stft_cache_enabled (full-file legacy STFT;
                        replaced by the streaming StftAccumulator at Stage 3 —
                        if this stage dominates, the whole-tree peak will be
                        insensitive to streaming_block_seconds)
    reader_open         AudioBlockReader header probe + seek (no samples)
    stream_loop         read -> extract channel -> calibrate in place ->
                        accumulate, for every block (peak across the loop is
                        THE per-worker number for the streamed base matrix)
    finalise            assemble the per-channel DataFrame(s)

Cache writes are deliberately skipped: this measures compute, not I/O side
effects.

Usage:
    python tools/profile_streaming_stages.py --config cfg.yaml
    python tools/profile_streaming_stages.py --config cfg.yaml \
        --wav path/to/file.wav --block-seconds 30

Requires: psutil, memory_profiler  (pip install -e .[profile])
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
    sys.exit("memory_profiler not installed. Run: pip install -e .[profile]")

try:
    from seasound.core.config import load_config
    from seasound.core.pipeline import find_audio_files
    from seasound.loader.base_matrix import BaseMatrixAccumulator
    from seasound.loader.calibration import load_calibration, resolve_calibration
    from seasound.loader.filename_parsers import get_parser
    from seasound.loader.reader import AudioBlockReader, extract_channel_block
    from seasound.analysis.calculate_stft import get_stft_for_file
except ImportError as e:
    sys.exit(f"Could not import seasound ({e}). Install it in this environment first.")


def run_stage(name, func, *args, **kwargs):
    """Run one stage, returning (retval, record-dict of memory metrics)."""
    proc = psutil.Process()
    rss_before = proc.memory_info().rss / 1e6
    snap0 = tracemalloc.take_snapshot()
    tracemalloc.reset_peak()
    peak_rss, ret = memory_usage((func, args, kwargs), interval=0.01,  # type: ignore
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
    ap.add_argument("--wav", default=None,
                    help="single WAV to profile (default: first matched)")
    ap.add_argument("--block-seconds", type=int, default=None,
                    help="override pipeline.streaming_block_seconds for this run")
    ap.add_argument("--outdir", default="profile_results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    config = load_config(args.config)
    block_seconds = args.block_seconds or config.pipeline.streaming_block_seconds

    wav = args.wav
    if wav is None:
        files = find_audio_files(config)
        if not files:
            sys.exit("No audio files matched the config; pass one explicitly with --wav.")
        wav = files[0]

    cache_dir = config.pipeline.cache_directory or os.path.join(
        config.output.directory, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    parser = get_parser(config.input)
    cal_df = load_calibration(config.calibration)

    tracemalloc.start()
    records = []

    # Stage 0: legacy full-file STFT npz step, exactly as _process_one_file
    # still runs it before the streaming dispatch (until Stage 3).
    if config.pipeline.stft_cache_enabled:
        _, rec = run_stage("stft_npz_legacy", get_stft_for_file, wav, config, cache_dir)
        records.append(rec)

    # Stage 1: open the reader (header probe + trim seek; no sample data)
    reader_cm = AudioBlockReader(
        wav, config.input, parser=parser,
        bin_seconds=config.pipeline.base_resolution_s,
    )
    reader, rec = run_stage("reader_open", reader_cm.__enter__)
    records.append(rec)

    try:
        duration_s = reader.n_bins * config.pipeline.base_resolution_s
        print(f"Profiling streaming stages for: {wav}")
        print(f"  {reader.sample_rate} Hz, {reader.n_channels} channel(s), "
              f"{reader.n_bins} bins (~{duration_s}s usable), "
              f"block_seconds={block_seconds}, "
              f"strategy={config.input.channel_strategy}\n")

        resolved = resolve_calibration(reader, cal_df, config.calibration)
        accumulators = {}
        for ch in reader.channels:
            acc = BaseMatrixAccumulator(
                reader.sample_rate, reader.n_bins, config.pipeline)
            acc.set_anchor(reader.datetime_start)
            accumulators[ch] = acc

        # Stage 2: the block loop — read, extract, calibrate in place, push.
        def stream_loop():
            n_blocks = 0
            for raw_block, t0 in reader.blocks(block_seconds):
                for ch in reader.channels:
                    channel_block = extract_channel_block(
                        raw_block, config.input.channel_strategy, ch)
                    accumulators[ch].push(
                        resolved.apply_inplace(channel_block), t0)
                n_blocks += 1
            return n_blocks

        n_blocks, rec = run_stage("stream_loop", stream_loop)
        records.append(rec)

        # Stage 3: assemble the output frame(s).
        def finalise():
            return {
                ch: acc.finalise(reader.datetime_start)
                for ch, acc in accumulators.items()
            }

        matrices, rec = run_stage("finalise", finalise)
        records.append(rec)
    finally:
        reader_cm.__exit__(None, None, None)

    tracemalloc.stop()

    # ---- report ----
    csv_path = os.path.join(args.outdir, "seasound_streaming_stage_memory.csv")
    with open(csv_path, "w", newline="") as f:  # pylint: disable=unspecified-encoding
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)

    heaviest = max(records, key=lambda r: r["transient_delta_mb"])
    print(f"{'stage':<22}{'peak RSS':>12}{'+delta':>12}{'py-traced':>12}")
    print("-" * 58)
    for r in records:
        mark = "  <-- heaviest" if r is heaviest else ""
        print(f"{r['stage']:<22}{r['peak_rss_mb']:>10.1f}MB"
              f"{r['transient_delta_mb']:>10.1f}MB"
              f"{r['py_traced_peak_mb']:>10.1f}MB{mark}")
    print("-" * 58)
    n_rows = sum(len(m) for m in matrices.values())
    print(f"{n_blocks} block(s) streamed; {n_rows} output row(s) across "
          f"{len(matrices)} channel(s)")
    print(f"Heaviest stage by transient memory: {heaviest['stage']} "
          f"(+{heaviest['transient_delta_mb']:.1f} MB)\n")
    print("Top Python allocation lines per stage:")
    for r in records:
        print(f"  [{r['stage']}] {r['top_python_alloc_lines']}")
    print(f"\nCSV: {csv_path}")
    print("\nNotes: 'peak RSS'/'+delta' include native numpy buffers (the real RAM")
    print("cost). stream_loop's peak is the per-worker streaming number; it should")
    print("scale with --block-seconds. If stft_npz_legacy is present and dominates,")
    print("the whole-tree peak will not respond to block size until Stage 3 lands.")


if __name__ == "__main__":
    main()

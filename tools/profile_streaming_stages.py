#!/usr/bin/env python3
"""
profile_streaming_stages.py — attribute per-file memory to the STREAMING
pipeline stages (refactor plan Stages 2–3).

Companion to profile_seasound_stages.py, which profiles the legacy
decomposition (read_audio -> apply_calibration -> compute_base_matrix)
and is kept unchanged as the baseline reference. This script mirrors
_process_one_file() / _process_one_file_streaming() as of Stage 3:

    [stft_npz_legacy]   only if stft_cache_enabled AND NOT
                        streaming_enabled — exactly the condition under
                        which the pipeline still runs the legacy
                        full-file npz step. With streaming on, this
                        stage no longer exists in production and is not
                        profiled.
    reader_open         AudioBlockReader header probe + seek (no samples)
    stream_loop         read -> extract channel -> calibrate in place ->
                        base-matrix push, plus (if stft_cache_enabled)
                        StftAccumulator push and incremental shard
                        append, plus the fractional-tail push. Peak
                        across the loop is THE per-worker streaming
                        number for both producers.
    finalise            assemble the per-channel DataFrame(s); finalise
                        STFT accumulators and shard writers.

Base-matrix parquet writes are skipped (compute, not I/O); STFT shard
writes are INCLUDED because the zarr append buffers are part of the
loop's real memory story — they go to a throwaway directory under
--outdir, never the production cache.

Usage:
    python tools/profile_streaming_stages.py --config cfg.yaml
    python tools/profile_streaming_stages.py --config cfg.yaml \
        --wav path/to/file.wav --block-seconds 30

Requires: psutil, memory_profiler  (pip install -e .[profile])
"""
import argparse
import csv
import os
import shutil
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
    from seasound.loader.stft import StftAccumulator
    from seasound.loader import stft_store
    from seasound.loader.stft_store import StftShardWriter
    from seasound.analysis.calculate_stft import get_stft_for_file
except ImportError as e:
    sys.exit(f"Could not import seasound ({e}). Install it in this environment first.")


def run_stage(name, func, *args, **kwargs):
    """Run one stage, returning (retval, record-dict of memory metrics).

    The call is memoized: memory_profiler re-invokes functions that
    finish faster than its sampling interval, which would double-run
    stateful stages (accumulator/writer finalisation guards would
    raise). The second invocation returns the cached result instantly.
    """
    box = {}

    def once():
        if "v" not in box:
            box["v"] = func(*args, **kwargs)
        return box["v"]

    proc = psutil.Process()
    rss_before = proc.memory_info().rss / 1e6
    snap0 = tracemalloc.take_snapshot()
    tracemalloc.reset_peak()
    peak_rss, ret = memory_usage((once, (), {}), interval=0.01,  # type: ignore
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


def profile_file(config, wav, block_seconds, outdir):
    """Profile one file through the streaming stages; return records."""
    cache_dir = config.pipeline.cache_directory or os.path.join(
        config.output.directory, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    parser = get_parser(config.input)
    cal_df = load_calibration(config.calibration)

    streaming = bool(config.pipeline.streaming_enabled)
    stft_enabled = bool(config.pipeline.stft_cache_enabled) and streaming

    # Shards from a profiling run never touch the production cache.
    shard_root = os.path.join(outdir, "profile_stft_shards")
    if os.path.isdir(shard_root):
        shutil.rmtree(shard_root)

    tracemalloc.start()
    records = []

    # Legacy full-file STFT npz step — profiled only under the exact
    # condition the pipeline still runs it (Stage 3 removed it from the
    # streaming path).
    if config.pipeline.stft_cache_enabled and not streaming:
        _, rec = run_stage("stft_npz_legacy", get_stft_for_file, wav, config, cache_dir)
        records.append(rec)

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
              f"strategy={config.input.channel_strategy}, "
              f"stft_in_loop={stft_enabled}\n")

        resolved = resolve_calibration(reader, cal_df, config.calibration)
        accumulators = {}
        for ch in reader.channels:
            acc = BaseMatrixAccumulator(
                reader.sample_rate, reader.n_bins, config.pipeline)
            acc.set_anchor(reader.datetime_start)
            accumulators[ch] = acc

        stft_accs, stft_writers = {}, {}
        if stft_enabled and reader.datetime_start is None:
            print("  (no datetime for file: STFT shard production skipped, "
                  "as in the pipeline)\n")
            stft_enabled = False
        if stft_enabled:
            for ch in reader.channels:
                stft_accs[ch] = StftAccumulator(
                    sample_rate=reader.sample_rate,
                    nfft=config.pipeline.stft_nfft,
                    win_length=config.pipeline.stft_win_length,
                    hop_length=config.pipeline.stft_hop_length,
                    window=config.pipeline.stft_window,
                    fmin_hz=config.pipeline.stft_fmin_hz,
                    fmax_hz=config.pipeline.stft_fmax_hz,
                )

        def _stft_push(ch, samples_pa):
            # Mirrors the pipeline's lazy-writer push exactly.
            frames = stft_accs[ch].push(samples_pa)
            if frames is None:
                return
            writer = stft_writers.get(ch)
            if writer is None:
                writer = StftShardWriter(
                    os.path.join(stft_store.stft_dir_for(shard_root),
                                 stft_store.shard_name(wav, ch)),
                    stft_accs[ch].freqs_hz,
                    reader.sample_rate,
                    config.pipeline.stft_hop_length,
                    config.pipeline.stft_win_length,
                    reader.datetime_start,
                    channel=ch,
                    serial=reader.serial,
                    window=config.pipeline.stft_window,
                    dtype=config.pipeline.stft_dtype,
                    time_chunk_frames=config.pipeline.stft_time_chunk_frames,
                )
                stft_writers[ch] = writer
            writer.append(frames)

        # The block loop — read, extract, calibrate in place, push to
        # both producers; then the fractional tail (STFT only).
        def stream_loop():
            n_blocks = 0
            for raw_block, t0 in reader.blocks(block_seconds):
                for ch in reader.channels:
                    channel_block = extract_channel_block(
                        raw_block, config.input.channel_strategy, ch)
                    block_pa = resolved.apply_inplace(channel_block)
                    accumulators[ch].push(block_pa, t0)
                    if stft_enabled:
                        _stft_push(ch, block_pa)
                n_blocks += 1
            if stft_enabled:
                tail = reader.read_tail()
                if tail is not None:
                    for ch in reader.channels:
                        channel_tail = extract_channel_block(
                            tail, config.input.channel_strategy, ch)
                        _stft_push(ch, resolved.apply_inplace(channel_tail))
            return n_blocks

        n_blocks, rec = run_stage("stream_loop", stream_loop)
        records.append(rec)

        # Assemble outputs; seal accumulators and shard writers.
        def finalise():
            out = {
                ch: acc.finalise(reader.datetime_start)
                for ch, acc in accumulators.items()
            }
            if stft_enabled:
                for ch in reader.channels:
                    stft_accs[ch].finalise()
                    if ch in stft_writers:
                        stft_writers[ch].finalise()
            return out

        matrices, rec = run_stage("finalise", finalise)
        records.append(rec)
    finally:
        reader_cm.__exit__(None, None, None)

    tracemalloc.stop()
    return records, n_blocks, matrices, stft_writers


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

    records, n_blocks, matrices, stft_writers = profile_file(
        config, wav, block_seconds, args.outdir)

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
    n_frames = sum(w._n_frames for w in stft_writers.values()) if stft_writers else 0  # pylint: disable=protected-access
    print(f"{n_blocks} block(s) streamed; {n_rows} output row(s) across "
          f"{len(matrices)} channel(s); {n_frames} STFT frame(s) sharded")
    print(f"Heaviest stage by transient memory: {heaviest['stage']} "
          f"(+{heaviest['transient_delta_mb']:.1f} MB)\n")
    print("Top Python allocation lines per stage:")
    for r in records:
        print(f"  [{r['stage']}] {r['top_python_alloc_lines']}")
    print(f"\nCSV: {csv_path}")
    print("\nNotes: 'peak RSS'/'+delta' include native numpy buffers (the real RAM")
    print("cost). With streaming+stft enabled, stream_loop is the true Stage 3")
    print("per-worker number (both producers, shard appends, fractional tail)")
    print("and should scale with --block-seconds. stft_npz_legacy appears only")
    print("when the pipeline itself would run it (stft on, streaming off).")


if __name__ == "__main__":
    main()

#pylint: disable=too-many-lines
"""
seasound/core/pipeline.py

Pipeline orchestrator: coordinates Stage 1 (loading) and Stage 2 (analysis).

Phase 1 implements Stage 1 only. Stage 2 is a placeholder that will be
filled in Phase 2.
"""

import os
import glob
import json
import logging
import time as time_module
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool, cpu_count
from functools import partial #pylint: disable=unused-import
from typing import Optional

import numpy as np #pylint: disable=unused-import
import pandas as pd

import seasound
import seasound.analysis # noqa: F401  (registers built-in analyses)
from seasound.core.config import PipelineConfig, load_config
from seasound.core.logging import setup_logging
from seasound.core.exceptions import SeaSoundError, ConfigError #pylint: disable=unused-import
from seasound.loader.reader import (
    AudioBlockReader,
    AudioSegment,
    extract_channel_block,
    probe_output_channels,
)
from seasound.loader.calibration import load_calibration, resolve_calibration
from seasound.loader.base_matrix import BaseMatrixAccumulator
from seasound.loader.cache import ( #pylint: disable=unused-import
    is_cached,
    base_matrix_cache_path,
    save_base_matrix,
    load_base_matrix,
    load_all_cached,
    load_cached_for_sources,
)
from seasound.loader.filename_parsers import FilenameParser, get_parser
from seasound.loader.metadata_readers import get_metadata_reader
from seasound.loader.loaded_artifacts import SegmentArtifact, LoadingOutput #pylint: disable=unused-import
from seasound.loader.stft import StftAccumulator
# Module-qualified: stft_store.write_manifest is the SHARD manifest;
# this module's own write_manifest is the run_manifest.json writer.
from seasound.loader import stft_store
from seasound.loader.stft_store import StftShardWriter

from seasound.core.substrates import (
    BASE_MATRIX,
    STFT,
    resolve_producers,
    subtract_cached,
    validate_substrates,
    validate_analyses_registered,
    validate_resolved_coverage,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_audio_files(config: PipelineConfig) -> list[str]:
    """Find all matching audio files in the input directory."""
    pattern = os.path.join(
        config.input.path,
        "**" if config.input.recursive else "",
        config.input.pattern,
    )
    files = sorted(glob.glob(pattern, recursive=config.input.recursive))

    if not files:
        logger.warning(
            "No files matching '%s' found in %s",
            config.input.pattern,
            config.input.path
        )

    return files


# ---------------------------------------------------------------------------
# Deployment clipping
# ---------------------------------------------------------------------------

def _resolve_raw_bounds(
    config: PipelineConfig,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Resolve unbuffered clipping bounds from the configured source."""
    method = config.deployment.clip_method

    if method == "none":
        return None, None

    if method == "manual":
        start_raw = config.deployment.start_utc
        end_raw = config.deployment.end_utc

        if start_raw is None or end_raw is None:
            logger.warning(
                "Manual clipping requested but start_utc/end_utc are missing; "
                "proceeding without clipping"
            )
            return None, None

        try:
            return pd.Timestamp(start_raw), pd.Timestamp(end_raw)
        except Exception as exc: #pylint: disable=broad-except
            logger.warning(
                "Could not parse manual clip datetimes (%s); "
                "proceeding without clipping",
                exc
            )
            return None, None

    if method == "metadata":
        try:
            return _load_clip_from_metadata(config)
        except Exception as exc: #pylint: disable=broad-except
            logger.warning(
                "Could not resolve metadata clip bounds (%s); "
                "proceeding without clipping",
                exc
            )
            return None, None

    raise SeaSoundError(f"Unknown clip_method: {method}")


def _apply_shared_buffer(
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    config: PipelineConfig,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    """Apply shared start/end clipping buffer in hours."""
    if start is None or end is None:
        return None, None

    buf = config.deployment.buffer_hours
    clip_start = start + timedelta(hours=buf.start)
    clip_end = end - timedelta(hours=buf.end)

    if clip_start >= clip_end:
        raise SeaSoundError(
            f"Clip window is empty after buffers: {clip_start} → {clip_end}"
        )

    return clip_start, clip_end


def get_clip_bounds(
    config: PipelineConfig,
    matrix: pd.DataFrame | None = None,
) -> tuple:
    """Determine temporal clip bounds and apply shared buffering."""
    if not config.deployment.enabled:
        return None, None

    method = config.deployment.clip_method
    raw_start, raw_end = _resolve_raw_bounds(config)

    # Only "none" mode is allowed to derive bounds from matrix extent.
    if method == "none" and matrix is not None and not matrix.empty:
        raw_start = matrix.index.min()
        raw_end = matrix.index.max()

    clip_start, clip_end = _apply_shared_buffer(raw_start, raw_end, config)

    if clip_start is None:
        logger.info("No clipping applied; using full dataset")
    else:
        logger.info("Clip window: %s → %s", clip_start, clip_end)

    return clip_start, clip_end


def _load_clip_from_metadata(config: PipelineConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    reader = get_metadata_reader(config.deployment)
    window = reader.read(
        config.deployment.metadata_file,
        config.deployment.location_id,
        config.deployment.hydrophone,
    )
    return pd.Timestamp(window.deploy_utc), pd.Timestamp(window.retrieve_utc)


def _merge_base_matrices(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge base matrices deterministically."""
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames).sort_index()
    full = full[~full.index.duplicated(keep="first")]
    return full


# ---------------------------------------------------------------------------
# Single-file processing (used by both serial and parallel modes)
# ---------------------------------------------------------------------------

def _process_one_file(
    wav_path: str,
    config: PipelineConfig,
    cal_df: Optional[pd.DataFrame],
    cache_dir: str,
    parser: Optional[FilenameParser] = None,
    resolved: Optional[set[str]] = None,
) -> list[SegmentArtifact]:
    """
    Process a single WAV file: stream → calibrate → produce.

    The file is streamed in whole-bin blocks (bit-identical output,
    per-worker memory bounded by one block, independent of file
    duration). Cache writes are side effects of producing; in-memory
    artifacts are returned regardless.

    ``resolved`` is this run's resolved producer set (§7); when omitted
    it is computed from ``config`` so direct callers need not supply it.
    Only ``resolved`` minus this file's already-cached products is
    produced (per-file subtraction, resume-aware).
    """
    if resolved is None:
        resolved = resolve_producers(config)

    # Produce only what's missing for this file. With resume off,
    # reproduce the whole resolved set (overwriting any cache).
    if config.pipeline.resume:
        to_produce = subtract_cached(
            resolved, _cached_products(wav_path, config, cache_dir)
        )
    else:
        to_produce = set(resolved)
    return _process_one_file_streaming(
        wav_path, config, cal_df, cache_dir, to_produce, parser
    )


def _process_one_file_streaming(
    wav_path: str,
    config: PipelineConfig,
    cal_df: Optional[pd.DataFrame],
    cache_dir: str,
    to_produce: set[str],
    parser: Optional[FilenameParser] = None,
) -> list[SegmentArtifact]:
    """
    Streaming per-file processing (refactor plan Stages 2–3): one pass
    over the file in whole-bin blocks; per output channel, in-place
    block calibration feeds the producers named in ``to_produce`` — a
    BaseMatrixAccumulator and/or an StftAccumulator writing an STFT
    shard incrementally. Per-worker memory holds one block plus the
    per-channel output arrays — independent of file duration.

    ``to_produce`` is the resolved producer set for this file (§7): the
    run's resolved producers minus what's already cached. A file
    processed for STFT only (base already cached) loads its base matrix
    from cache so the Stage-1 merge still sees its rows, without
    recomputing it.
    """
    produce_base = BASE_MATRIX in to_produce
    produce_stft = STFT in to_produce

    with AudioBlockReader(
        wav_path,
        config.input,
        parser=parser,
        bin_seconds=config.pipeline.base_resolution_s,
    ) as reader:
        # Per-file calibration state, resolved before any samples are
        # read (the reader duck-types via .serial / .source_file).
        resolved = resolve_calibration(reader, cal_df, config.calibration) #type: ignore

        # One base-matrix accumulator per output channel, single pass —
        # only when producing the base matrix (plan Stage 2; under
        # 'auto' the per-worker peak scales with channel count, never
        # with file length).
        accumulators: dict[int, BaseMatrixAccumulator] = {}
        if produce_base:
            for ch in reader.channels:
                acc = BaseMatrixAccumulator(
                    reader.sample_rate, reader.n_bins, config.pipeline
                )
                acc.set_anchor(reader.datetime_start)
                accumulators[ch] = acc

        # STFT shards (plan Stage 3): same blocks, second producer.
        # The store is datetime-addressed, so a file without a parsed
        # start time cannot have a shard (build_stft_matrix likewise
        # skips such files); warn and continue with the base matrix.
        stft_enabled = produce_stft
        if stft_enabled and reader.datetime_start is None:
            logger.warning(
                "No datetime for %s; STFT shard skipped (store is "
                "datetime-addressed)", wav_path,
            )
            stft_enabled = False

        stft_accs: dict[int, StftAccumulator] = {}
        stft_writers: dict[int, StftShardWriter] = {}
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

        def _stft_push(ch: int, samples_pa) -> None:
            """Feed calibrated samples to channel ch's accumulator;
            append any completed frames to its shard. The writer is
            created lazily on the first completed frames, when the
            masked frequency axis becomes known from the shared
            compute."""
            frames = stft_accs[ch].push(samples_pa)
            if frames is None:
                return
            writer = stft_writers.get(ch)
            if writer is None:
                writer = StftShardWriter(
                    os.path.join(
                        stft_store.stft_dir_for(cache_dir),
                        stft_store.shard_name(wav_path, ch),
                    ),
                    stft_accs[ch].freqs_hz, #type: ignore
                    reader.sample_rate,
                    config.pipeline.stft_hop_length,
                    config.pipeline.stft_win_length,
                    reader.datetime_start, #type: ignore
                    channel=ch,
                    serial=reader.serial,
                    window=config.pipeline.stft_window,
                    dtype=config.pipeline.stft_dtype,
                    time_chunk_frames=(
                        config.pipeline.stft_time_chunk_frames
                    ),
                )
                stft_writers[ch] = writer
            writer.append(frames)

        for raw_block, t0 in reader.blocks(
            config.pipeline.streaming_block_seconds
        ):
            for ch in reader.channels:
                # View or per-sample mean of THIS block only; in-place
                # calibration mutates the raw block's storage, which is
                # discarded at the end of the iteration. Under 'auto'
                # the channels are disjoint columns, so per-channel
                # in-place calibration cannot interact.
                channel_block = extract_channel_block(
                    raw_block, config.input.channel_strategy, ch
                )
                block_pa = resolved.apply_inplace(channel_block)
                if produce_base:
                    accumulators[ch].push(block_pa, t0)

                if stft_enabled:
                    # Same calibrated samples, second producer; the
                    # base accumulator does not mutate the block.
                    _stft_push(ch, block_pa)

        if stft_enabled:
            # The STFT must see the file's fractional tail — the samples
            # past the last whole bin, which the whole-bin blocks
            # exclude (§9 test 3). Feed it to the STFT producers only;
            # the base matrix drops it, by whole-bin construction.
            tail = reader.read_tail()
            if tail is not None:
                for ch in reader.channels:
                    channel_tail = extract_channel_block(
                        tail, config.input.channel_strategy, ch
                    )
                    _stft_push(ch, resolved.apply_inplace(channel_tail))

        artifacts: list[SegmentArtifact] = []
        for ch in reader.channels:
            if produce_base:
                matrix = accumulators[ch].finalise(reader.datetime_start)
            else:
                # STFT-only run for a file whose base matrix is already
                # cached: load it so the Stage-1 merge still sees its
                # rows — no recompute, parquet untouched.
                matrix = load_base_matrix(
                    base_matrix_cache_path(wav_path, ch, cache_dir)
                )

            if stft_enabled:
                # Discards the trailing carry (legacy parity) and seals
                # the shard: the complete flag lands in one atomic
                # metadata write. The manifest row is not carried back —
                # the coordinator regenerates the manifest from shard
                # attributes, which also covers shards from prior
                # (resumed) runs.
                stft_accs[ch].finalise()
                writer = stft_writers.get(ch)
                if writer is not None:
                    writer.finalise()
                else:
                    logger.warning(
                        "No STFT frames produced for %s ch%d (shorter "
                        "than stft_win_length?); no shard written",
                        wav_path, ch,
                    )

            cache_paths: list[str] = []
            if produce_base:
                # Producing the base matrix means caching it — the
                # resolver and resume rule treat the parquet as the
                # base-matrix product. save_base_matrix consumes segment
                # *metadata* only; the streaming path has no sample
                # array, so it passes an empty placeholder.
                meta_segment = AudioSegment(
                    data=np.empty(0, dtype=np.float32),
                    sample_rate=reader.sample_rate,
                    serial=reader.serial,
                    datetime_start=reader.datetime_start,
                    channel=ch,
                    source_file=wav_path,
                )
                path = save_base_matrix(
                    matrix, meta_segment, resolved.calibrated, cache_dir
                )
                cache_paths.append(path)

            artifacts.append(
                SegmentArtifact(
                    source_file=wav_path,
                    channel=ch,
                    serial=reader.serial,
                    datetime_start=reader.datetime_start,
                    calibrated=resolved.calibrated,
                    base_matrix=matrix,
                    cache_paths=cache_paths,
                )
            )

    return artifacts


def _worker_fn(args):
    """Wrapper for multiprocessing (can't pickle lambdas)."""
    wav_path, config, cal_df, cache_dir, resolved = args
    # Create parser inside each worker (parser objects may not be picklable)
    parser = get_parser(config.input)
    t0 = time_module.time()
    try:
        artifacts = _process_one_file(
            wav_path, config, cal_df, cache_dir, parser, resolved=resolved
        )
        return True, time_module.time() - t0, artifacts
    except Exception as exc: #pylint: disable=broad-exception-caught
        logger.error("Error processing %s: %s", wav_path, exc)
        return False, time_module.time() - t0, []


# ---------------------------------------------------------------------------
# Stage 1: Loading
# ---------------------------------------------------------------------------

def run_loading(config: PipelineConfig) -> pd.DataFrame:
    """
    Stage 1: Read WAV files, calibrate, compute base matrices, cache.

    Returns the full merged base matrix (clipped to deployment window).
    """
    # Resolve cache directory
    cache_dir = config.pipeline.cache_directory or os.path.join(
        config.output.directory, "cache"
    )
    os.makedirs(cache_dir, exist_ok=True)
    os.makedirs(config.output.directory, exist_ok=True)

    # Find files
    wav_files = find_audio_files(config)
    if not wav_files:
        raise SeaSoundError(
            f"No audio files found in {config.input.path} "
            f"matching '{config.input.pattern}'"
        )
    logger.info("Found %d audio file(s)", len(wav_files))

    # Preflight (hard error): every enabled analysis must resolve to a
    # registered, instantiable module, else it would contribute no substrate
    # needs silently — the registry import-order failure mode.
    reg_errors = validate_analyses_registered(config)
    if reg_errors:
        raise ConfigError(
            "Analysis configuration errors:\n"
            + "\n".join(f"  • {e}" for e in reg_errors)
        )

    resolved = resolve_producers(config)

    # Hard error: a needed substrate missing from the resolved set for any
    # reason other than being force-disabled is a resolver/registration bug.
    cov_errors = validate_resolved_coverage(config, resolved)
    if cov_errors:
        raise ConfigError(
            "Substrate resolution errors:\n"
            + "\n".join(f"  • {e}" for e in cov_errors)
        )

    # Warn-and-skip (user choice): an enabled analysis whose required
    # substrate was force-disabled degrades gracefully.
    for warning in validate_substrates(config):
        logger.warning(warning)
    logger.info("Resolved producers: %s", ", ".join(sorted(resolved)) or "(none)")

    # Filter already-cached files if resume=True
    if config.pipeline.resume:
        files_to_process = [
            f for f in wav_files
            if not _is_fully_cached(f, config, cache_dir, resolved)
        ]
        files_to_process_set = set(files_to_process)
        skipped_files = [f for f in wav_files if f not in files_to_process_set]
        n_skipped = len(skipped_files)
        if n_skipped > 0:
            logger.info("Resuming: skipping %d fully-cached file(s)", n_skipped)
    else:
        files_to_process = wav_files
        skipped_files = []

    # Load calibration
    cal_df = load_calibration(config.calibration)

    load_outputs: list[SegmentArtifact] = []

    # Process files
    total = len(files_to_process)
    if total > 0:
        n_workers = config.pipeline.workers
        if n_workers == 0:
            n_workers = max(1, cpu_count() - 2)
        n_workers = min(n_workers, total)

        logger.info(
            "Processing %d file(s) with %d worker%s",
            total,
            n_workers,
            "s" if n_workers > 1 else "",
        )

        durations = []
        completed = 0
        successes = 0

        if n_workers == 1 or total == 1:
            # Serial processing (easier to debug)
            parser = get_parser(config.input)
            for i, wav_file in enumerate(files_to_process, 1): #pylint: disable=unused-variable
                t0 = time_module.time()
                try:
                    artifacts = _process_one_file(
                        wav_file, config, cal_df, cache_dir, parser,
                        resolved=resolved,
                    )
                    load_outputs.extend(artifacts)
                    success = True
                except Exception as exc: #pylint: disable=broad-exception-caught
                    logger.error("Error processing %s: %s", wav_file, exc)
                    success = False
                dur = time_module.time() - t0
                completed += 1
                successes += int(success)
                durations.append(dur)

                # ETA
                avg = sum(durations) / len(durations)
                remaining = (total - completed) * avg
                eta = timedelta(seconds=int(remaining))
                logger.info(
                    "[%d/%d] %s (%.1fs) ETA: %s",
                    completed, total,
                    'OK' if success else 'FAIL',
                    dur, eta
                )
        else:
            # Parallel processing
            args = [
                (f, config, cal_df, cache_dir, resolved) for f in files_to_process
            ]
            with Pool(n_workers) as pool:
                for success, dur, artifacts in pool.imap_unordered(_worker_fn, args):

                    load_outputs.extend(artifacts)

                    completed += 1
                    successes += int(success)
                    durations.append(dur)

                    if completed % 10 == 0 or completed == total:
                        avg = sum(durations) / len(durations) / n_workers
                        remaining = (total - completed) * avg
                        eta = timedelta(seconds=int(remaining))
                        logger.info(
                            "Progress: %d/%d (%d OK) ETA: %s",
                            completed,
                            total,
                            successes,
                            eta,
                        )

        logger.info(
            "Loading complete: %d/%d files processed", successes, total
        )

    # STFT shard manifest (refactor Stage 3, D8): written serially by
    # the coordinator only, after all workers have finished — and on
    # every run, including a resumed run that skipped every file. The
    # rows come from a scan of shard attributes — the declared source
    # of truth — rather than worker-returned rows, so the manifest
    # always covers every complete shard on disk, including those
    # produced by earlier (resumed) runs.
    if STFT in resolved:
        stft_dir = stft_store.stft_dir_for(cache_dir)
        shard_rows = stft_store.rebuild_manifest_rows(stft_dir)
        if shard_rows:
            stft_store.write_manifest(shard_rows, stft_dir)
            logger.info(
                "STFT manifest written: %d shard(s)", len(shard_rows)
            )

    # Load all cached matrices (including previously cached ones)
    in_memory_frames = [a.base_matrix for a in load_outputs]
    full_matrix = _merge_base_matrices(in_memory_frames)

    if skipped_files:
        skipped_basenames = {os.path.basename(p) for p in skipped_files}
        skipped_cached = load_cached_for_sources(cache_dir, skipped_basenames)
        if not skipped_cached.empty:
            full_matrix = _merge_base_matrices([full_matrix, skipped_cached])

    if full_matrix.empty:
        raise SeaSoundError("No base matrix rows available after Stage 1 processing")

    # Clip to deployment window
    clip_start, clip_end = get_clip_bounds(config, full_matrix)
    if clip_start is not None:
        before = len(full_matrix)
        full_matrix = full_matrix.loc[
            (full_matrix.index >= clip_start) & (full_matrix.index <= clip_end)
        ]
        logger.info(
            "Deployment clipping: %s → %s rows",
            f"{before:,}",
            f"{len(full_matrix):,}",
        )

    return full_matrix


def _expected_channels_for_file(
    wav_path: str,
    config: PipelineConfig,
) -> list[int]:
    """
    Return output channel IDs expected for this file under current strategy.

    Header-only probe: previously this called read_audio, which decoded
    the entire file in the coordinator just to count channels — once
    per file, per resumed run. probe_output_channels resolves the same
    channel set from the file header (equivalence is gated by a test).
    """
    return probe_output_channels(wav_path, config.input)


def _cached_products(
    wav_path: str,
    config: PipelineConfig,
    cache_dir: str,
) -> set[str]:
    """The products already complete on disk for this file (§12).

    A product counts as cached only when present for **every** expected
    output channel: base matrix → per-channel parquet; STFT → per-channel
    complete shard. Used by both the resume rule and the per-file
    producer subtraction.
    """
    channels = _expected_channels_for_file(wav_path, config)
    cached: set[str] = set()
    if not channels:
        return cached

    if all(is_cached(wav_path, ch, cache_dir) for ch in channels):
        cached.add(BASE_MATRIX)

    stft_dir = stft_store.stft_dir_for(cache_dir)
    if all(
        stft_store.shard_complete(stft_dir, wav_path, ch) for ch in channels
    ):
        cached.add(STFT)

    return cached


def _is_fully_cached(
    wav_path: str,
    config: PipelineConfig,
    cache_dir: str,
    resolved: Optional[set[str]] = None,
) -> bool:
    """True iff every product this run resolved to produce is already
    complete for the file (conjunctive, product-aware; §12). ``resolved``
    is computed from ``config`` when not supplied."""
    if resolved is None:
        resolved = resolve_producers(config)
    return not subtract_cached(resolved, _cached_products(wav_path, config, cache_dir))

# ---------------------------------------------------------------------------
# Stage 2: Analysis (placeholder for Phase 2)
# ---------------------------------------------------------------------------

def run_analyses(base_matrix: pd.DataFrame, config: PipelineConfig) -> dict:
    """
    Run enabled analysis modules and return results.
    
    Parameters
    ----------
    base_matrix : pd.DataFrame
        Computed TOB base matrix from Stage 1.
    config : PipelineConfig
        Full pipeline config with analyses entries.
    
    Returns
    -------
    dict
        Per-module results: {name → {"outputs", "summary", "warnings"}}.  
    """
    from seasound.analysis.registry import get_analysis

    results = {}
    analyses = config.analyses or {}

    cache_dir = config.pipeline.cache_directory or os.path.join(
        config.output.directory, "cache"
    )
    runtime_context = {
        "pipeline_config": config,
        "cache_dir": cache_dir,
        "input_files": find_audio_files(config),
    }

    for name, entry in analyses.items():
        # Skip disabled analyses
        if not isinstance(entry, dict) or not entry.get("enabled", False):
            logger.debug("Skipping disabled analysis: %s", name)
            continue

        required = entry.get("required", True)

        try:
            # Instantiate and validate module
            module = get_analysis(name)
            module.set_runtime_context(runtime_context)
            module_cfg = entry.get("config", {})
            module.validate_config(module_cfg)

            # Run analysis
            logger.info("Running analysis: %s (required=%s)", name, required)
            res = module.run(base_matrix, module_cfg, config.output.directory)

            results[name] = {
                "outputs": res.outputs,
                "summary": res.summary,
                "warnings": res.warnings,
            }
            # Log warnings if any
            if res.warnings:
                for warning in res.warnings:
                    logger.warning("  [%s] %s", name, warning)

        except Exception as e: #pylint: disable=broad-exception-caught
            if required:
                logger.error("Required analysis '%s' failed: %s", name, e)
                raise
            else:
                logger.warning("Optional analysis '%s' failed (continuing): %s", name, e)
                continue

    if not results:
        logger.info("No enabled analyses found in config")

    return results


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def write_manifest(
    config: PipelineConfig,
    input_files: list[str],
    output_dir: str,
    elapsed_s: float,
    analysis_results: dict | None = None,
):
    """
    Write run_manifest.json with full provenance and analysis results.
    
    Parameters
    ----------
    config : PipelineConfig
        Full pipeline config.
    input_files : list
        List of input audio file paths.
    output_dir : str
        Output directory path.
    elapsed_s : float
        Pipeline elapsed time in seconds.
    analysis_results : dict or None
        Results from run_analyses() or None if analyses were skipped.
    """
    manifest = {
        "seasound_version": seasound.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": round(elapsed_s, 1),
        "n_input_files": len(input_files),
        "config_summary": {
            "input_path": config.input.path,
            "filename_format": config.input.filename_format,
            "calibration_enabled": config.calibration.enabled,
            "deployment_clipping": config.deployment.enabled,
            "max_freq_hz": config.pipeline.max_freq_hz,
            "base_resolution_s": config.pipeline.base_resolution_s,
        },
    }

    # Add analysis results if present
    if analysis_results:
        manifest["analyses"] = analysis_results

    path = os.path.join(output_dir, "run_manifest.json")
    with open(path, "w") as f: #pylint: disable=unspecified-encoding
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written: %s", path)


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def run_pipeline(config: PipelineConfig) -> None:
    """Run the full pipeline (Stage 1 + Stage 2)."""
    t0 = time_module.time()

    if not config.analyse_only:
        logger.info("=" * 60)
        logger.info("STAGE 1: DATA LOADING")
        logger.info("=" * 60)
        base_matrix = run_loading(config)
    else:
        logger.info("--analyse-only: loading from cache")
        cache_dir = config.pipeline.cache_directory or os.path.join(
            config.output.directory, "cache"
        )
        base_matrix = load_all_cached(cache_dir)

        # Clip
        clip_start, clip_end = get_clip_bounds(config, base_matrix)
        if clip_start is not None:
            base_matrix = base_matrix.loc[
                (base_matrix.index >= clip_start)
                & (base_matrix.index <= clip_end)
            ]

    analysis_results = None
    if not config.load_only:
        logger.info("=" * 60)
        logger.info("STAGE 2: ANALYSIS")
        logger.info("=" * 60)
        analysis_results = run_analyses(base_matrix, config)

    elapsed = time_module.time() - t0
    input_files = find_audio_files(config)
    write_manifest(config, input_files, config.output.directory, elapsed, analysis_results)

    logger.info("=" * 60)
    logger.info("Pipeline complete (%s)", timedelta(seconds=int(elapsed)))
    logger.info("Output: %s", config.output.directory)
    logger.info("=" * 60)

def run_plot_mode(
    kind: str,
    csv_path: str,
    output_path: str | None,
    config_path: str | None,
) -> None:
    """
    Standalone plot mode: read a CSV produced by an analysis module and
    write a plot.

    Parameters
    ----------
    kind : str
        Analysis name; currently 'ltsa' or 'spectral_percentiles'.
    csv_path : str
        Path to the analysis output CSV.
    output_path : str, optional
        Path for the output image. If None, defaults to a sibling of the
        input CSV named '<csv_stem>_<plot_kind>.png'.
    config_path : str, optional
        If given, read the matching `analyses.<kind>.config.plot` block to
        configure the plot. Otherwise built-in defaults are used.
    """
    import os #pylint: disable=redefined-outer-name
    import matplotlib.pyplot as plt

    if not os.path.isfile(csv_path):
        raise SeaSoundError(f"Input CSV not found: {csv_path}")

    # --- Load optional plot config from YAML ---
    plot_cfg: dict = {}
    if config_path:
        try:
            from seasound.core.config import load_yaml
            raw = load_yaml(config_path)
            an = raw.get("analyses", {}).get(kind, {}) or {}
            plot_cfg = (an.get("config", {}) or {}).get("plot", {}) or {}
        except Exception as exc: #pylint: disable=broad-exception-caught
            logger.warning(
                "Could not load plot config from %s: %s; "
                "using defaults.",
                config_path,
                exc
            )

    # --- Build plotter and figure ---
    if kind == "ltsa":
        from seasound.plotting.ltsa import LTSAPlotter
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        plotter = LTSAPlotter(df)

        types = plot_cfg.get("types", ["heatmap"])
        plot_kind = types[0]
        kind_cfg = plot_cfg.get(plot_kind, {}) or {}
        if plot_kind == "heatmap":
            fig = plotter.heatmap(**kind_cfg)
        elif plot_kind == "band_timeseries":
            fig = plotter.band_timeseries(**kind_cfg)
        else:
            raise SeaSoundError(f"Unknown LTSA plot kind: '{plot_kind}'")

    elif kind == "spectral_percentiles":
        from seasound.plotting.spectral_percentiles import (
            SpectralPercentilesPlotter,
        )
        df = pd.read_csv(csv_path)
        plotter = SpectralPercentilesPlotter(df)

        types = plot_cfg.get("types", ["curves"])
        plot_kind = types[0]
        kind_cfg = plot_cfg.get(plot_kind, {}) or {}
        if plot_kind == "curves":
            fig = plotter.curves(**kind_cfg)
        else:
            raise SeaSoundError(
                f"Unknown spectral_percentiles plot kind: '{plot_kind}'"
            )

    else:
        raise SeaSoundError(f"Unsupported plot kind: '{kind}'")

    # --- Resolve output path and save ---
    if output_path is None:
        base = os.path.splitext(os.path.basename(csv_path))[0]
        directory = os.path.dirname(csv_path) or "."
        output_path = os.path.join(directory, f"{base}_{plot_kind}.png")

    dpi = plot_cfg.get("dpi", 300)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot written: %s", output_path)

def cli_main():
    """CLI entry point (called by the `seasound` command)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SeaSound Acoustic Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", 
        help="Path to YAML configuration file (required for pipeline modes)"
    )
    parser.add_argument(
        "--plot",
        choices=["ltsa", "spectral_percentiles"],
        help=(
            "Standalone plot mode: generate a plot from an existing CSV and "
            "exit. Use with --input <csv_path>. Optional: --output <png_path> "
            "and --config <yaml> to read the matching plot block."
        ),
    )
    parser.add_argument("--input", help="Override input.path")
    parser.add_argument("--output", help="Override output.directory")
    parser.add_argument("--workers", type=int, help="Override pipeline.workers")
    parser.add_argument("--max-freq", type=float, help="Override pipeline.max_freq_hz")
    parser.add_argument(
        "--load-only", action="store_true",
        help="Only produce base matrices without any analysis."
    )
    parser.add_argument(
        "--analyse-only", action="store_true",
        help="Run analysis only (requires existing cache). Useful if running new analyses on previously loaded data." #pylint: disable=line-too-long
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate config and count files; no processing"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)"
    )
    parser.add_argument(
        "--list-analyses", action="store_true",
        help="List all registered analysis modules and exit"
    )

    args = parser.parse_args()

    # Build CLI overrides dict
    overrides = {}
    if args.input:
        overrides["input.path"] = args.input
    if args.output:
        overrides["output.directory"] = args.output
    if args.workers is not None:
        overrides["pipeline.workers"] = args.workers
    if args.max_freq is not None:
        overrides["pipeline.max_freq_hz"] = args.max_freq

    # Setup logging
    setup_logging(args.log_level)

    # --- Standalone plot mode ---
    if args.plot:
        if args.load_only or args.analyse_only or args.dry_run:
            print("Error: --plot is mutually exclusive with --load-only, "
                "--analyse-only, and --dry-run")
            raise SystemExit(1)
        if not args.input:
            print("Error: --plot requires --input <csv_path>")
            raise SystemExit(1)
        try:
            return run_plot_mode(
                kind=args.plot,
                csv_path=args.input,
                output_path=args.output,
                config_path=args.config,
            )
        except SeaSoundError as exc:
            logger.error("\nPlot error: %s", exc)
            raise SystemExit(1) from exc

    # List analyses if requested
    if args.list_analyses:
        from seasound.analysis.registry import list_registered
        modules = list_registered()
        if modules:
            print("\nRegistered analysis modules:")
            for name, cls in sorted(modules.items()):
                print(f"  {name:25} ({cls})")
        else:
            print("No analysis modules registered")
        return

    if not args.config:
        print("Error: --config is required (or use --plot / --list-analyses)")
        raise SystemExit(1)

    # Load config
    try:
        config = load_config(args.config, overrides)
    except SeaSoundError as exc:
        print(f"\nConfiguration error:\n{exc}")
        raise SystemExit(1) from exc

    # Apply runtime flags
    config.load_only = args.load_only
    config.analyse_only = args.analyse_only
    config.dry_run = args.dry_run

    if args.load_only and args.analyse_only:
        print("Error: --load-only and --analyse-only are mutually exclusive")
        raise SystemExit(1)

    # Dry run
    if config.dry_run:
        files = find_audio_files(config)
        print(f"\nDry run:") #pylint: disable=f-string-without-interpolation
        print(f"  Config: {args.config}")
        print(f"  Input:  {config.input.path}")
        print(f"  Files:  {len(files)}")
        print(f"  Output: {config.output.directory}")
        print(f"\nConfig validated successfully.") #pylint: disable=f-string-without-interpolation
        return

    # Run
    try:
        run_pipeline(config)
    except SeaSoundError as exc:
        logger.error("\nPipeline error: %s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt as exc:
        logger.warning("\nInterrupted by user")
        raise SystemExit(130) from exc

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
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd

import seasound
from seasound.core.config import PipelineConfig, load_config
from seasound.core.logging import setup_logging
from seasound.core.exceptions import SeaSoundError
from seasound.loader.reader import read_audio
from seasound.loader.calibration import load_calibration, apply_calibration
from seasound.loader.base_matrix import compute_base_matrix
from seasound.loader.cache import (
    is_cached,
    save_base_matrix,
    load_base_matrix,
    load_all_cached,
)
from seasound.loader.filename_parsers import FilenameParser, get_parser
from seasound.loader.metadata_readers import get_metadata_reader

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
            f"No files matching '{config.input.pattern}' "
            f"found in {config.input.path}"
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
        except Exception as exc:
            logger.warning(
                f"Could not parse manual clip datetimes ({exc}); "
                "proceeding without clipping"
            )
            return None, None

    if method == "metadata":
        try:
            return _load_clip_from_metadata(config)
        except Exception as exc:
            logger.warning(
                f"Could not resolve metadata clip bounds ({exc}); "
                "proceeding without clipping"
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
        logger.info(f"Clip window: {clip_start} → {clip_end}")

    return clip_start, clip_end


def _load_clip_from_metadata(config: PipelineConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    reader = get_metadata_reader(config.deployment)
    window = reader.read(
        config.deployment.metadata_file,
        config.deployment.location_id,
        config.deployment.hydrophone,
    )
    return pd.Timestamp(window.deploy_utc), pd.Timestamp(window.retrieve_utc)

# ---------------------------------------------------------------------------
# Single-file processing (used by both serial and parallel modes)
# ---------------------------------------------------------------------------

def _process_one_file(
    wav_path: str,
    config: PipelineConfig,
    cal_df: Optional[pd.DataFrame],
    cache_dir: str,
    parser: Optional[FilenameParser] = None,
) -> list[str]:
    """
    Process a single WAV file: read → calibrate → compute matrix → cache.

    Returns list of Parquet paths created.
    """
    segments = read_audio(wav_path, config.input, parser=parser)
    parquet_paths = []

    for segment in segments:
        audio_pa, calibrated = apply_calibration(
            segment, cal_df, config.calibration
        )

        matrix = compute_base_matrix(audio_pa, segment.sample_rate, config.pipeline)

        if config.pipeline.cache_base_matrix:
            path = save_base_matrix(matrix, segment, calibrated, cache_dir)
            parquet_paths.append(path)

    return parquet_paths


def _worker_fn(args):
    """Wrapper for multiprocessing (can't pickle lambdas)."""
    wav_path, config, cal_df, cache_dir = args
    # Create parser inside each worker (parser objects may not be picklable)
    parser = get_parser(config.input)
    t0 = time_module.time()
    try:
        paths = _process_one_file(wav_path, config, cal_df, cache_dir, parser)
        return True, time_module.time() - t0, paths
    except Exception as exc:
        logger.error(f"Error processing {wav_path}: {exc}")
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
    logger.info(f"Found {len(wav_files)} audio file(s)")

    # Filter already-cached files if resume=True
    if config.pipeline.resume:
        uncached = [
            f for f in wav_files
            if not _is_fully_cached(f, config, cache_dir)
        ]
        n_skipped = len(wav_files) - len(uncached)
        if n_skipped > 0:
            logger.info(f"Resuming: skipping {n_skipped} fully-cached file(s)")
        files_to_process = uncached
    else:
        files_to_process = wav_files

    # Load calibration
    cal_df = load_calibration(config.calibration)

    # Process files
    total = len(files_to_process)
    if total > 0:
        n_workers = config.pipeline.workers
        if n_workers == 0:
            n_workers = max(1, cpu_count() - 2)
        n_workers = min(n_workers, total)

        logger.info(
            f"Processing {total} file(s) with "
            f"{n_workers} worker{'s' if n_workers > 1 else ''}"
        )

        durations = []
        completed = 0
        successes = 0

        if n_workers == 1 or total == 1:
            # Serial processing (easier to debug)
            parser = get_parser(config.input)
            for i, wav_file in enumerate(files_to_process, 1):
                t0 = time_module.time()
                try:
                    _process_one_file(wav_file, config, cal_df, cache_dir, parser)
                    success = True
                except Exception as exc:
                    logger.error(f"Error processing {wav_file}: {exc}")
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
                    f"[{completed}/{total}] "
                    f"{'OK' if success else 'FAIL'} "
                    f"({dur:.1f}s) ETA: {eta}"
                )
        else:
            # Parallel processing
            args = [
                (f, config, cal_df, cache_dir) for f in files_to_process
            ]
            with Pool(n_workers) as pool:
                for success, dur, _ in pool.imap_unordered(_worker_fn, args):
                    completed += 1
                    successes += int(success)
                    durations.append(dur)

                    if completed % 10 == 0 or completed == total:
                        avg = sum(durations) / len(durations) / n_workers
                        remaining = (total - completed) * avg
                        eta = timedelta(seconds=int(remaining))
                        logger.info(
                            f"Progress: {completed}/{total} "
                            f"({successes} OK) ETA: {eta}"
                        )

        logger.info(
            f"Loading complete: {successes}/{total} files processed"
        )

    # Load all cached matrices (including previously cached ones)
    logger.info("Loading cached base matrices…")
    full_matrix = load_all_cached(cache_dir)

    # Clip to deployment window
    clip_start, clip_end = get_clip_bounds(config, full_matrix)
    if clip_start is not None:
        before = len(full_matrix)
        full_matrix = full_matrix.loc[
            (full_matrix.index >= clip_start) & (full_matrix.index <= clip_end)
        ]
        logger.info(
            f"Deployment clipping: {before:,} → {len(full_matrix):,} rows"
        )

    return full_matrix


def _expected_channels_for_file(wav_path: str, config: PipelineConfig) -> list[int]:
    """Return output channel IDs expected for this file under current strategy."""
    parser = get_parser(config.input)
    segments = read_audio(wav_path, config.input, parser=parser)
    return [seg.channel for seg in segments]


def _is_fully_cached(wav_path: str, config: PipelineConfig, cache_dir: str) -> bool:
    """True only if all expected output channels are present in cache."""
    channels = _expected_channels_for_file(wav_path, config)
    return all(is_cached(wav_path, ch, cache_dir) for ch in channels)

# ---------------------------------------------------------------------------
# Stage 2: Analysis (placeholder for Phase 2)
# ---------------------------------------------------------------------------

def run_analyses(base_matrix: pd.DataFrame, config: PipelineConfig) -> dict:
    from seasound.analysis.registry import get_analysis

    results = {}
    analyses = config.analyses or {}

    for name, entry in analyses.items():
        if not isinstance(entry, dict) or not entry.get("enabled", False):
            continue

        module = get_analysis(name)
        module_cfg = entry.get("config", {})
        module.validate_config(module_cfg)

        res = module.run(base_matrix, module_cfg, config.output.directory)
        results[name] = {
            "outputs": res.outputs,
            "summary": res.summary,
            "warnings": res.warnings,
        }

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
):
    """Write run_manifest.json alongside outputs."""
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
    path = os.path.join(output_dir, "run_manifest.json")
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"Manifest written: {path}")


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

    if not config.load_only:
        logger.info("=" * 60)
        logger.info("STAGE 2: ANALYSIS")
        logger.info("=" * 60)
        run_analyses(base_matrix, config)

    elapsed = time_module.time() - t0
    input_files = find_audio_files(config)
    write_manifest(config, input_files, config.output.directory, elapsed)

    logger.info("=" * 60)
    logger.info(f"Pipeline complete ({timedelta(seconds=int(elapsed))})")
    logger.info(f"Output: {config.output.directory}")
    logger.info("=" * 60)


def cli_main():
    """CLI entry point (called by the `seasound` command)."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SeaSound Acoustic Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", required=True, help="Path to YAML configuration file"
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
        help="Run analysis only (requires existing cache). Useful if running new analyses on previously loaded data."
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

    # Load config
    try:
        config = load_config(args.config, overrides)
    except SeaSoundError as exc:
        print(f"\nConfiguration error:\n{exc}")
        raise SystemExit(1)

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
        print(f"\nDry run:")
        print(f"  Config: {args.config}")
        print(f"  Input:  {config.input.path}")
        print(f"  Files:  {len(files)}")
        print(f"  Output: {config.output.directory}")
        print(f"\nConfig validated successfully.")
        return

    # Run
    try:
        run_pipeline(config)
    except SeaSoundError as exc:
        logger.error(f"\nPipeline error: {exc}")
        raise SystemExit(1)
    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        raise SystemExit(130)
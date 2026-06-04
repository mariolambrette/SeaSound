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
from seasound.core.config import PipelineConfig, load_config
from seasound.core.logging import setup_logging
from seasound.core.exceptions import SeaSoundError
from seasound.loader.reader import read_audio
from seasound.loader.calibration import load_calibration, apply_calibration
from seasound.loader.base_matrix import compute_base_matrix
from seasound.loader.cache import ( #pylint: disable=unused-import
    is_cached,
    save_base_matrix,
    load_base_matrix, #pylint: disable=unused-import
    load_all_cached,
    load_cached_for_sources,
) #pylint: disable=unused-import
from seasound.loader.filename_parsers import FilenameParser, get_parser
from seasound.loader.metadata_readers import get_metadata_reader
from seasound.loader.loaded_artifacts import SegmentArtifact, LoadingOutput #pylint: disable=unused-import

from seasound.analysis.calculate_stft import get_stft_for_file


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
) -> list[SegmentArtifact]:
    """
    Process a single WAV file: read → calibrate → compute matrix.

    Cache writes are optional side effects.
    Returns in-memory artifacts regardless of cache mode.
    """
    # Optional STFT cache generation during Stage 1.
    # This ensures load-only runs can precompute STFT products.
    if config.pipeline.stft_cache_enabled:
        try:
            get_stft_for_file(wav_path, config, cache_dir)
        except Exception as exc:
            logger.error("STFT caching failed for %s: %s", wav_path, exc)
            raise
    
    segments = read_audio(wav_path, config.input, parser=parser)
    artifacts: list[SegmentArtifact] = []

    for segment in segments:
        audio_pa, calibrated = apply_calibration(
            segment, cal_df, config.calibration
        )
        matrix = compute_base_matrix(audio_pa, segment.sample_rate, config.pipeline)

        # Keep Stage 1 in-memory artifacts aligned with cached representation:
        # analysis modules require DatetimeIndex for temporal resampling.
        if (
            segment.datetime_start is not None
            and not isinstance(matrix.index, pd.DatetimeIndex)
        ):
            dt_index = pd.date_range(
                start=segment.datetime_start,
                periods=len(matrix),
                freq="1s",
            )
            matrix = matrix.copy()
            matrix.index = dt_index
            matrix.index.name = "datetime"

        cache_paths: list[str] = []
        if config.pipeline.cache_base_matrix:
            path = save_base_matrix(matrix, segment, calibrated, cache_dir)
            cache_paths.append(path)

        artifacts.append(
            SegmentArtifact(
                source_file=segment.source_file,
                channel=segment.channel,
                serial=segment.serial,
                datetime_start=segment.datetime_start,
                calibrated=calibrated,
                base_matrix=matrix,
                cache_paths=cache_paths,
            )
        )

    return artifacts


def _worker_fn(args):
    """Wrapper for multiprocessing (can't pickle lambdas)."""
    wav_path, config, cal_df, cache_dir = args
    # Create parser inside each worker (parser objects may not be picklable)
    parser = get_parser(config.input)
    t0 = time_module.time()
    try:
        artifacts = _process_one_file(wav_path, config, cal_df, cache_dir, parser)
        return True, time_module.time() - t0, artifacts
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
        files_to_process = [
            f for f in wav_files
            if not _is_fully_cached(f, config, cache_dir)
        ]
        files_to_process_set = set(files_to_process)
        skipped_files = [f for f in wav_files if f not in files_to_process_set]
        n_skipped = len(skipped_files)
        if n_skipped > 0:
            logger.info(f"Resuming: skipping {n_skipped} fully-cached file(s)")
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
                    artifacts = _process_one_file(wav_file, config, cal_df, cache_dir, parser)
                    load_outputs.extend(artifacts)
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
                            f"Progress: {completed}/{total} "
                            f"({successes} OK) ETA: {eta}"
                        )

        logger.info(
            f"Loading complete: {successes}/{total} files processed"
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
            f"Deployment clipping: {before:,} → {len(full_matrix):,} rows"
        )

    return full_matrix


def _expected_channels_for_file(
    wav_path: str, 
    config: PipelineConfig,
) -> list[int]:
    """
    Return output channel IDs expected for this file under current strategy.
    """
    parser = get_parser(config.input)
    segments = read_audio(wav_path, config.input, parser=parser)
    return [seg.channel for seg in segments]


def _is_fully_cached(
    wav_path: str, 
    config: PipelineConfig, 
    cache_dir: str,
) -> bool:
    """True only if all expected output channels are present in cache."""
    channels = _expected_channels_for_file(wav_path, config)
    
    base_ok = all(is_cached(wav_path, ch, cache_dir) for ch in channels)
    if not base_ok:
        return False
    
    if not config.pipeline.stft_cache_enabled:
        return True
    
    base = os.path.splitext(os.path.basename(wav_path))[0]
    return all(
        os.path.isfile(os.path.join(cache_dir, f"{base}_ch{ch}_stft.npz"))
        for ch in channels
    )

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
            logger.debug(f"Skipping disabled analysis: {name}")
            continue
        
        required = entry.get("required", True)
        
        try:
            # Instantiate and validate module
            module = get_analysis(name)
            module.set_runtime_context(runtime_context)
            module_cfg = entry.get("config", {})
            module.validate_config(module_cfg)
            
            # Run analysis
            logger.info(f"Running analysis: {name} (required={required})")
            res = module.run(base_matrix, module_cfg, config.output.directory)
            
            results[name] = {
                "outputs": res.outputs,
                "summary": res.summary,
                "warnings": res.warnings,
            }
            
            # Log warnings if any
            if res.warnings:
                for warning in res.warnings:
                    logger.warning(f"  [{name}] {warning}")
        
        except Exception as e:
            if required:
                logger.error(f"Required analysis '{name}' failed: {e}")
                raise
            else:
                logger.warning(f"Optional analysis '{name}' failed (continuing): {e}")
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
    import os
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
        except Exception as exc:
            logger.warning(
                f"Could not load plot config from {config_path}: {exc}; "
                f"using defaults."
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
    logger.info(f"Plot written: {output_path}")

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
            logger.error(f"\nPlot error: {exc}")
            raise SystemExit(1)   

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
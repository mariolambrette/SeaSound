"""
Spectrogram analysis module.

Creates time-frequency heatmap visualizations using matplotlib.
"""

import os
import logging
from collections.abc import Iterator
from copy import copy

import pandas as pd
import numpy as np

from seasound.analysis.base import AnalysisModule, AnalysisResult, AnalysisModuleError
from seasound.analysis.registry import register_analysis

logger = logging.getLogger(__name__)


class SpectrogramAnalysis(AnalysisModule):
    """
    Spectrogram analysis.
    
    Creates a time-frequency heatmap (PNG) visualization of the base matrix.
    X-axis: time, Y-axis: frequency (Hz), color: SPL (dB).
    """
    
    name = "spectrogram"

    def validate_config(self, cfg: dict) -> None:
        """
        Validate Spectrogram configuration.
        
        Optional keys:
        - freq_range: [freq_min_hz, freq_max_hz] or null for all
        - db_range: [db_min, db_max] or null for automatic scaling
        - colormap: matplotlib colormap name (default: "viridis")
        - output_format: "png" (default), "pdf", or "csv"
        - dpi: output DPI (default: 300)
        - time_chunk: pandas offset alias (e.g. "1h", "1d") or null
        - preserve_time_gaps: bool, render missing seconds as blank columns
        """
        errors = []
        
        freq_range = cfg.get("freq_range")
        if freq_range is not None:
            if not isinstance(freq_range, (list, tuple)) or len(freq_range) != 2:
                errors.append(
                    f"spectrogram.config.freq_range must be [freq_min, freq_max] or null; "
                    f"got {freq_range}"
                )
            elif freq_range[0] >= freq_range[1]:
                errors.append(
                    f"spectrogram.config.freq_range must have min < max"
                )
        
        db_range = cfg.get("db_range")
        if db_range is not None:
            if not isinstance(db_range, (list, tuple)) or len(db_range) != 2:
                errors.append(
                    f"spectrogram.config.db_range must be [db_min, db_max] or null; "
                    f"got {db_range}"
                )
            elif db_range[0] >= db_range[1]:
                errors.append(
                    f"spectrogram.config.db_range must have min < max"
                )
        
        colormap = cfg.get("colormap", "viridis")
        if not isinstance(colormap, str):
            errors.append(
                f"spectrogram.config.colormap must be a string; got {type(colormap).__name__}"
            )
        
        output_format = cfg.get("output_format", "png")
        if output_format not in {"png", "pdf", "csv"}:
            errors.append(
                f"spectrogram.config.output_format must be one of "
                f"{{'png', 'pdf', 'csv'}}; got '{output_format}'"
            )
        
        dpi = cfg.get("dpi", 300)
        if not isinstance(dpi, int) or dpi <= 0:
            errors.append(
                f"spectrogram.config.dpi must be a positive integer; got {dpi}"
            )

        time_chunk = cfg.get("time_chunk")
        if time_chunk is not None:
            if not isinstance(time_chunk, str):
                errors.append(
                    f"spectrogram.config.time_chunk must be a pandas offset alias "
                    f"(e.g. '1h', '1d') or null; got {type(time_chunk).__name__}"
                )
            else:
                try:
                    pd.tseries.frequencies.to_offset(time_chunk)
                except (ValueError, TypeError):
                    errors.append(
                        f"spectrogram.config.time_chunk '{time_chunk}' is not a valid "
                        f"pandas offset alias"
                    )

        preserve_time_gaps = cfg.get("preserve_time_gaps", False)
        if not isinstance(preserve_time_gaps, bool):
            errors.append(
                f"spectrogram.config.preserve_time_gaps must be a boolean; "
                f"got {type(preserve_time_gaps).__name__}"
            )
        
        if errors:
            raise ValueError("\n".join(errors))


    def run(
        self,
        base_matrix: pd.DataFrame,
        cfg: dict,
        output_dir: str,
    ) -> AnalysisResult:
        """Execute Spectrogram analysis."""
        self.validate_config(cfg)
        self._validate_base_matrix(base_matrix)
        
        try:
            # Filter frequencies
            freq_range = cfg.get("freq_range")
            work_matrix = self._filter_frequencies(base_matrix, freq_range)
            
            # Extract metadata
            db_range = cfg.get("db_range")
            colormap = cfg.get("colormap", "viridis")
            output_format = cfg.get("output_format", "png")
            dpi = cfg.get("dpi", 300)
            time_chunk = cfg.get("time_chunk")
            preserve_time_gaps = cfg.get("preserve_time_gaps", False)
            warnings: list[str] = []

            if work_matrix.empty:
                raise AnalysisModuleError("Spectrogram input is empty after filtering")
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            def _format_ts_for_filename(ts: pd.Timestamp) -> str:
                return ts.strftime("%Y%m%dT%H%M%S")

            def _iter_time_chunks() -> Iterator[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
                if not isinstance(work_matrix.index, pd.DatetimeIndex):
                    raise AnalysisModuleError("Spectrogram requires a DatetimeIndex")

                if time_chunk is None:
                    start = work_matrix.index.min()
                    end = work_matrix.index.max()
                    if not isinstance(start, pd.Timestamp) or not isinstance(end, pd.Timestamp):
                        raise AnalysisModuleError(
                            "Could not determine spectrogram time bounds"
                        )
                    yield start, end, work_matrix
                    return

                for _, group in work_matrix.resample(time_chunk):
                    if group.empty:
                        continue

                    if not isinstance(group.index, pd.DatetimeIndex):
                        raise AnalysisModuleError("Chunk index must be DatetimeIndex")

                    window_start = group.index.min()
                    window_end = group.index.max()
                    if (
                        not isinstance(window_start, pd.Timestamp)
                        or not isinstance(window_end, pd.Timestamp)
                    ):
                        raise AnalysisModuleError("Could not determine chunk bounds")

                    yield window_start, window_end, group

            chunks = list(_iter_time_chunks())
            if not chunks:
                raise AnalysisModuleError("No spectrogram chunks produced")

            outputs: list[str] = []
            
            # Handle different output formats
            if output_format == "csv":
                if preserve_time_gaps:
                    warnings.append(
                        "preserve_time_gaps has no effect for CSV output"
                    )
                # Save raw matrix as CSV (single file or chunked files)
                for i, (window_start, window_end, chunk_matrix) in enumerate(chunks):
                    if len(chunks) == 1 and time_chunk is None:
                        filename = "spectrogram.csv"
                    else:
                        s = _format_ts_for_filename(window_start)
                        e = _format_ts_for_filename(window_end)
                        filename = f"spectrogram_{i:04d}_{s}_{e}.csv"
                    output_file = os.path.join(output_dir, filename)
                    chunk_matrix.to_csv(output_file)
                    outputs.append(output_file)
                    logger.info(f"Spectrogram output (CSV): {output_file}")

                return AnalysisResult(
                    name=self.name,
                    outputs=outputs,
                    summary={
                        "output_format": "csv",
                        "n_outputs": len(outputs),
                        "time_chunk": time_chunk if time_chunk is not None else "full",
                        "preserve_time_gaps": preserve_time_gaps,
                        "n_time_samples": len(work_matrix),
                        "n_frequencies": len(work_matrix.columns),
                        "time_range": f"{work_matrix.index.min()} to {work_matrix.index.max()}",
                    },
                    warnings=warnings,
                )
            
            # Create PNG/PDF visualization using matplotlib
            try:
                import matplotlib
                matplotlib.use("Agg")  # Non-interactive backend
                import matplotlib.pyplot as plt
            except ImportError:
                raise AnalysisModuleError(
                    "Spectrogram visualization requires matplotlib; "
                    "install with: pip install matplotlib"
                )

            missing_seconds_visualized = 0
            
            for i, (window_start, window_end, chunk_matrix) in enumerate(chunks):
                # Prepare data for visualization
                # Transpose so time is X-axis, frequency is Y-axis
                plot_matrix = chunk_matrix
                if preserve_time_gaps:
                    full_index = pd.date_range(
                        start=chunk_matrix.index.min(),
                        end=chunk_matrix.index.max(),
                        freq="1s",
                    )
                    plot_matrix = chunk_matrix.reindex(full_index)
                    missing_seconds_visualized += len(full_index) - len(chunk_matrix)

                spec_data = plot_matrix.T.values
                if preserve_time_gaps:
                    spec_data = np.ma.masked_invalid(spec_data)

                times = plot_matrix.index
                freq_labels = [
                    float(col[:-2]) for col in chunk_matrix.columns
                ]

                # Create figure
                fig, ax = plt.subplots(figsize=(14, 8), dpi=100)

                cmap_obj = copy(plt.get_cmap(colormap))
                if preserve_time_gaps:
                    cmap_obj.set_bad(color="white", alpha=1.0)

                # Plot heatmap
                im = ax.imshow(
                    spec_data,
                    aspect="auto",
                    cmap=cmap_obj,
                    vmin=db_range[0] if db_range else None,
                    vmax=db_range[1] if db_range else None,
                    origin="lower",
                )

                # Set axis labels and title
                ax.set_xlabel("Time (UTC)")
                ax.set_ylabel("Frequency (Hz)")
                if len(chunks) == 1 and time_chunk is None:
                    if preserve_time_gaps:
                        ax.set_title("Spectrogram (gaps preserved)")
                    else:
                        ax.set_title("Spectrogram")
                else:
                    ax.set_title(f"Spectrogram: {window_start} to {window_end}")

                # Format time axis (show ~10 ticks)
                n_time_ticks = min(10, len(times))
                time_idx = np.linspace(0, len(times) - 1, n_time_ticks, dtype=int)
                ax.set_xticks(time_idx)
                ax.set_xticklabels([str(times[j])[:16] for j in time_idx], rotation=45)

                # Format frequency axis
                n_freq_ticks = min(10, len(freq_labels))
                freq_idx = np.linspace(0, len(freq_labels) - 1, n_freq_ticks, dtype=int)
                ax.set_yticks(freq_idx)
                ax.set_yticklabels([f"{freq_labels[j]:.1f}" for j in freq_idx])

                # Add colorbar
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("SPL (dB re 1 µPa)")

                # Tight layout
                plt.tight_layout()

                # Save figure
                if len(chunks) == 1 and time_chunk is None:
                    filename = f"spectrogram.{output_format}"
                else:
                    s = _format_ts_for_filename(window_start)
                    e = _format_ts_for_filename(window_end)
                    filename = f"spectrogram_{i:04d}_{s}_{e}.{output_format}"
                output_file = os.path.join(output_dir, filename)
                fig.savefig(output_file, dpi=dpi, format=output_format)
                plt.close(fig)
                outputs.append(output_file)
                logger.info(
                    f"Spectrogram output ({output_format.upper()}): {output_file}"
                )
            
            return AnalysisResult(
                name=self.name,
                outputs=outputs,
                summary={
                    "output_format": output_format,
                    "dpi": dpi,
                    "colormap": colormap,
                    "n_outputs": len(outputs),
                    "time_chunk": time_chunk if time_chunk is not None else "full",
                    "preserve_time_gaps": preserve_time_gaps,
                    "missing_seconds_visualized": missing_seconds_visualized,
                    "n_time_samples": len(work_matrix),
                    "n_frequencies": len(work_matrix.columns),
                    "time_range": f"{work_matrix.index.min()} to {work_matrix.index.max()}",
                    "db_range": db_range if db_range else "auto",
                },
                warnings=warnings,
            )
        
        except AnalysisModuleError:
            raise
        except Exception as exc:
            raise AnalysisModuleError(f"Spectrogram analysis failed: {exc}")


register_analysis("spectrogram", SpectrogramAnalysis)
"""
LTSA (Long-Term Spectral Average) analysis module for SeaSound.

Resamples a 1-second base matrix to a coarser time resolution and computes
summary statistics per frequency band.
"""

import os
import logging
from typing import Literal #pylint: disable=unused-import

import pandas as pd
import numpy as np

from seasound.analysis.base import (
    AnalysisModule,
    AnalysisResult,
    AnalysisModuleError,
)
from seasound.analysis.registry import register_analysis
from seasound.plotting._common import _validate_plot_block

logger = logging.getLogger(__name__)


class LTSAAnalysis(AnalysisModule):
    """
    Long-Term Spectral Average (LTSA) analysis.

    Aggregates 1-second resolution base_matrix to a coarser time resolution and
    computes per-frequency summary statistics.
    """

    name = "ltsa"


    def validate_config(self, cfg: dict) -> None:
        """
        Validate LTSA configuration.

        Required keys:
        - time_resolution: pandas offset alias (e.g. "1h", "6h", "1d")
        - statistic: one of {"median", "mean", "max", "min", "std"}

        Optional keys:
        - freq_range: [freq_min_hz, freq_max_hz] or null for all
        - output_format: "csv" (default)
        """
        errors = []

        if "time_resolution" not in cfg:
            errors.append("ltsa.config.time_resolution is required")
        else:
            time_res = cfg["time_resolution"]
            if not isinstance(time_res, str):
                errors.append(
                    f"ltsa.config.time_resolution must be a string "
                    f"(pandas offset alias like '1h', '6h', '1d'); "
                    f"got {type(time_res).__name__}"
                )
            else:
                try:
                    pd.tseries.frequencies.to_offset(time_res)
                except (ValueError, TypeError):
                    errors.append(
                        f"ltsa.config.time_resolution must be a valid pandas "
                        f"offset alias (examples: '1h', '6h', '1d'); got '{time_res}'"
                    )

        statistic = cfg.get("statistic", "median")
        valid_stats = {"median", "mean", "max", "min", "std"}
        if statistic not in valid_stats:
            errors.append(
                f"ltsa.config.statistic must be one of {valid_stats}; "
                f"got '{statistic}'"
            )

        freq_range = cfg.get("freq_range")
        if freq_range is not None:
            if not isinstance(freq_range, (list, tuple)) or len(freq_range) != 2:
                errors.append(
                    f"ltsa.config.freq_range must be [freq_min, freq_max] or null; "
                    f"got {freq_range}"
                )
            elif not all(isinstance(f, (int, float)) for f in freq_range):
                errors.append(
                    "ltsa.config.freq_range values must be numeric"
                )
            elif freq_range[0] >= freq_range[1]:
                errors.append(
                    f"ltsa.config.freq_range must be [min, max] with min < max; "
                    f"got [{freq_range[0]}, {freq_range[1]}]"
                )

        output_format = cfg.get("output_format", "csv")
        if output_format not in {"csv"}:
            errors.append(
                f"ltsa.config.output_format must be 'csv'; got '{output_format}'"
            )

        # --- Plot block (optional) ---
        _validate_plot_block(
            module_name="ltsa",
            cfg=cfg,
            valid_types={"heatmap", "band_timeseries"},
            errors=errors,
        )
        plot_cfg = cfg.get("plot") or {}
        if "band_timeseries" in (plot_cfg.get("types") or []):
            bt = plot_cfg.get("band_timeseries", {}) or {}
            bands_hz = bt.get("bands_hz")
            if not isinstance(bands_hz, list) or not bands_hz:
                errors.append(
                    "ltsa.config.plot.band_timeseries.bands_hz must be a non-empty "
                    "list of numeric frequencies when 'band_timeseries' is in "
                    "plot.types"
                )
            elif not all(isinstance(b, (int, float)) for b in bands_hz):
                errors.append(
                    "ltsa.config.plot.band_timeseries.bands_hz entries must be numeric"
                )

        if errors:
            raise ValueError("\n".join(errors))

    def run(
        self,
        base_matrix: pd.DataFrame,
        cfg: dict,
        output_dir: str,
    ) -> AnalysisResult:
        """
        Execute LTSA analysis.

        resamplesbase_matrix to time_resolution, applies statistic, and writes
        result CSV.
        """
        self.validate_config(cfg)
        self._validate_base_matrix(base_matrix)

        try:
            # Filter frequencies if requested
            freq_range = cfg.get("freq_range")
            work_matrix = self._filter_frequencies(base_matrix, freq_range)

            time_resolution = cfg.get("time_resolution", "1h")
            statistic = cfg.get("statistic", "median")

            if statistic == "median":
                aggregated = work_matrix.resample(time_resolution).median()
            elif statistic == "mean":
                # Mean must be computed in linear power domain, not dB
                aggregated = self._compute_linear_mean(
                    work_matrix, time_resolution
                )
            elif statistic == "max":
                aggregated = work_matrix.resample(time_resolution).max()
            elif statistic == "min":
                aggregated = work_matrix.resample(time_resolution).min()
            elif statistic == "std":
                aggregated = work_matrix.resample(time_resolution).std()
            else:
                raise AnalysisModuleError(f"Unknown statistic: {statistic}")

            # Write CSV
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "ltsa.csv")
            aggregated.to_csv(output_file)
            logger.info("LTSA output: %s (%d rows)", output_file, len(aggregated))

            # Generate plots if configured
            plot_outputs, plot_warnings = self._generate_plots(
                aggregated, cfg.get("plot") or {}, output_dir,
            )

            return AnalysisResult(
                name=self.name,
                outputs=[output_file] + plot_outputs,
                summary={
                    "n_time_bins": len(aggregated),
                    "n_frequencies": len(aggregated.columns),
                    "time_resolution": time_resolution,
                    "statistic": statistic,
                    "time_range": (
                        f"{aggregated.index.min()} to {aggregated.index.max()}"
                    ),
                    "plots_generated": len(plot_outputs),
                },
                warnings=plot_warnings,
            )

        except AnalysisModuleError:
            raise
        except Exception as exc:
            raise AnalysisModuleError(f"LTSA analysis failed: {exc}") from exc


    # --- Helper methods ---
    def _compute_linear_mean(
        self, work_matrix: pd.DataFrame,
        time_resolution: str,
    ) -> pd.DataFrame:
        """Compute mean in linear power domain, then convert back to dB."""
        linear_power = 10 ** (work_matrix / 10)
        aggregated_linear = linear_power.resample(time_resolution).mean()
        return aggregated_linear.apply(lambda col: 10 * np.log10(col))

    def _generate_plots(
        self,
        aggregated: pd.DataFrame,
        plot_cfg: dict,
        output_dir: str,
    ) -> tuple[list[str], list[str]]:
        """
        Produce LTSA plots according to plot_cfg. Returns (output_paths, warnings).

        Failures are logged and recorded as warnings but never abort the run.
        """
        outputs: list[str] = []
        warnings: list[str] = []

        if not plot_cfg.get("enabled", False):
            return outputs, warnings

        try:
            from seasound.plotting.ltsa import LTSAPlotter
            import matplotlib.pyplot as plt
        except ImportError as exc:
            warnings.append(f"LTSA plotting requires matplotlib: {exc}")
            logger.warning(warnings[-1])
            return outputs, warnings

        plotter = LTSAPlotter(aggregated)
        types = plot_cfg.get("types", ["heatmap"])
        output_format = plot_cfg.get("output_format", "png")
        dpi = plot_cfg.get("dpi", 300)

        for kind in types:
            kind_cfg = plot_cfg.get(kind, {}) or {}
            try:
                if kind == "heatmap":
                    fig = plotter.heatmap(**kind_cfg)
                elif kind == "band_timeseries":
                    fig = plotter.band_timeseries(**kind_cfg)
                else:
                    warnings.append(f"Unknown LTSA plot type '{kind}'; skipped.")
                    logger.warning(warnings[-1])
                    continue

                plot_path = os.path.join(
                    output_dir, f"ltsa_{kind}.{output_format}"
                )
                fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                outputs.append(plot_path)
                logger.info("LTSA plot: %s", plot_path)
            except Exception as exc: #pylint: disable=broad-except
                warnings.append(f"LTSA plot '{kind}' failed: {exc}")
                logger.warning(warnings[-1])

        return outputs, warnings


register_analysis("ltsa", LTSAAnalysis)

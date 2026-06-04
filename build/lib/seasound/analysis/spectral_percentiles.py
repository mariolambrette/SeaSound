"""
Spectral Percentiles analysis module.

Computes percentile distributions per frequency band over time windows.
"""

from abc import abstractmethod #pylint: disable=unused-import
import os
import logging

import pandas as pd

from seasound.analysis.base import AnalysisModule, AnalysisResult, AnalysisModuleError
from seasound.analysis.registry import register_analysis
from seasound.plotting._common import _validate_plot_block


logger = logging.getLogger(__name__)


class SpectralPercentilesAnalysis(AnalysisModule):
    """
    Spectral Percentiles analysis.
    
    Computes percentile distributions (e.g., p5, p50, p95) per frequency band
    over the entire deployment or windowed time periods.
    """

    name = "spectral_percentiles"


    def validate_config(self, cfg: dict) -> None:
        """
        Validate Spectral Percentiles configuration.
        
        Required keys:
        - percentiles: list of percentile values (0-100), e.g., [5, 25, 50, 75, 95]
        
        Optional keys:
        - window: "full" (entire deployment) or pandas offset alias (e.g., "1h")
        - freq_range: [freq_min_hz, freq_max_hz] or null for all
        - output_format: "csv" (default)
        """
        errors = []

        if "percentiles" not in cfg:
            errors.append("spectral_percentiles.config.percentiles is required")
        else:
            percentiles = cfg["percentiles"]
            if not isinstance(percentiles, list):
                errors.append(
                    f"spectral_percentiles.config.percentiles must be a list; "
                    f"got {type(percentiles).__name__}"
                )
            elif not percentiles:
                errors.append("spectral_percentiles.config.percentiles cannot be empty")
            else:
                for p in percentiles:
                    if not isinstance(p, (int, float)) or p < 0 or p > 100:
                        errors.append(
                            f"spectral_percentiles.config.percentiles must be in [0, 100]; "
                            f"got {p}"
                        )

        window = cfg.get("window", "full")
        if not isinstance(window, str):
            errors.append(
                f"spectral_percentiles.config.window must be 'full' or a "
                f"pandas offset alias (e.g., '1h'); got {window}"
            )
        elif window != "full":
            try:
                pd.tseries.frequencies.to_offset(window)
            except (ValueError, TypeError):
                errors.append(
                    f"spectral_percentiles.config.window must be 'full' or a valid "
                    f"pandas offset alias (e.g., '1h', '6h', '1d'); got '{window}'"
                )

        freq_range = cfg.get("freq_range")
        if freq_range is not None:
            if not isinstance(freq_range, (list, tuple)) or len(freq_range) != 2:
                errors.append(
                    f"spectral_percentiles.config.freq_range must be [freq_min, freq_max] or null; "
                    f"got {freq_range}"
                )
            elif not all(isinstance(f, (int, float)) and f >= 0 for f in freq_range):
                errors.append(
                    f"spectral_percentiles.config.freq_range values must be non-negative numbers; "
                    f"got {freq_range}"
                )
            elif freq_range[0] >= freq_range[1]:
                errors.append(
                    "spectral_percentiles.config.freq_range must have min < max"
                )

        output_format = cfg.get("output_format", "csv")
        if output_format not in {"csv"}:
            errors.append(
                f"spectral_percentiles.config.output_format must be 'csv'; "
                f"got '{output_format}'"
            )

        # --- Plot block (optional) ---
        _validate_plot_block(
            module_name="spectral_percentiles",
            cfg=cfg,
            valid_types={"curves"},
            errors=errors,
        )
        plot_cfg = cfg.get("plot") or {}
        curves_cfg = plot_cfg.get("curves", {}) or {}
        if "shaded_percentiles" in curves_cfg:
            sp = curves_cfg["shaded_percentiles"]
            if (
                not isinstance(sp, (list, tuple))
                or len(sp) != 2
                or not all(isinstance(x, int) for x in sp)
                or not all(0 <= x <= 100 for x in sp)
            ):
                errors.append(
                    "spectral_percentiles.config.plot.curves.shaded_percentiles "
                    "must be [lo, hi] with 0 <= lo < hi <= 100"
                )
        if "max_panels" in curves_cfg:
            mp = curves_cfg["max_panels"]
            if not isinstance(mp, int) or mp <= 0:
                errors.append(
                    "spectral_percentiles.config.plot.curves.max_panels must be a "
                    "positive integer"
                )

        if errors:
            raise ValueError("\n".join(errors))

    def run(
        self,
        base_matrix: pd.DataFrame,
        cfg: dict,
        output_dir: str,
    ) -> AnalysisResult:
        """Execute Spectral Percentiles analysis."""
        self.validate_config(cfg)
        self._validate_base_matrix(base_matrix)

        try:
            # Filter frequencies
            freq_range = cfg.get("freq_range")
            work_matrix = self._filter_frequencies(base_matrix, freq_range)

            # Compute percentiles
            percentiles = cfg.get("percentiles", [5, 25, 50, 75, 95])
            window = cfg.get("window", "full")

            # Force percentiles to integers
            percentile_labels = [
                str(int(p)) 
                if float(p).is_integer() 
                else str(p) for p in percentiles
            ]

            if window == "full":
                # Compute percentiles across entire deployment
                output_data = {}
                for freq_col in work_matrix.columns:
                    for p, p_label in zip(percentiles, percentile_labels):
                        col_name = f"{freq_col}_p{p_label}"
                        output_data[col_name] = float(
                            work_matrix[freq_col].quantile(float(p) / 100.0)
                        )

                output_df = pd.DataFrame([output_data])
            else:
                # Compute percentiles per window
                rows: list[dict[str, object]] = []
                offset = pd.tseries.frequencies.to_offset(window)

                for _, window_group in work_matrix.resample(window):
                    if window_group.empty:
                        continue

                    if not isinstance(window_group.index, pd.DatetimeIndex):
                        raise AnalysisModuleError(
                            "Windowed spectral percentiles require a "
                            "DatetimeIndex"
                        )

                    ws = window_group.index.min()
                    if not isinstance(ws, pd.Timestamp):
                        raise AnalysisModuleError("Could not determine window start timestamp")

                    row: dict[str, object] = {
                        "window_start": ws,
                        "window_end": ws + offset,
                    }
                    for freq_col in window_group.columns:
                        for p, p_label in zip(percentiles, percentile_labels):
                            col_name = f"{freq_col}_p{p_label}"
                            row[col_name] = float(
                                window_group[freq_col].quantile(float(p) / 100.0)
                            )
                    rows.append(row)

                output_df = pd.DataFrame(rows)

            if output_df.empty:
                raise AnalysisModuleError(
                    "No non-empty windows produced for spectral percentiles"
                )

            # Write CSV
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "spectral_percentiles.csv")
            output_df.to_csv(output_file, index=False)
            logger.info("Spectral Percentiles output: %s", output_file)

            # Generate plots if configured
            plot_outputs, plot_warnings = self._generate_plots(
                output_df, cfg.get("plot") or {}, output_dir,
            )

            n_windows = len(output_df)
            return AnalysisResult(
                name=self.name,
                outputs=[output_file] + plot_outputs,
                summary={
                    "n_percentiles": len(percentiles),
                    "n_frequencies": len(work_matrix.columns),
                    "percentiles": percentiles,
                    "window": window,
                    "n_windows": n_windows,
                    "plots_generated": len(plot_outputs),
                },
                warnings=plot_warnings,
            )

        except AnalysisModuleError:
            raise
        except Exception as exc:
            raise AnalysisModuleError(
                f"Spectral Percentiles analysis failed: {exc}"
            ) from exc

    def _generate_plots(
        self,
        output_df: pd.DataFrame,
        plot_cfg: dict,
        output_dir: str,
    ) -> tuple[list[str], list[str]]:
        """Produce spectral-percentile plots. Returns (output_paths, warnings)."""
        outputs: list[str] = []
        warnings: list[str] = []

        if not plot_cfg.get("enabled", False):
            return outputs, warnings

        try:
            from seasound.plotting.spectral_percentiles import (
                SpectralPercentilesPlotter,
            )
            import matplotlib.pyplot as plt
        except ImportError as exc:
            warnings.append(
                f"Spectral percentiles plotting requires matplotlib: {exc}"
            )
            logger.warning(warnings[-1])
            return outputs, warnings

        plotter = SpectralPercentilesPlotter(output_df)
        types = plot_cfg.get("types", ["curves"])
        output_format = plot_cfg.get("output_format", "png")
        dpi = plot_cfg.get("dpi", 300)

        for kind in types:
            kind_cfg = plot_cfg.get(kind, {}) or {}
            try:
                if kind == "curves":
                    fig = plotter.curves(**kind_cfg)
                else:
                    warnings.append(
                        f"Unknown spectral_percentiles plot type '{kind}'; skipped."
                    )
                    logger.warning(warnings[-1])
                    continue

                plot_path = os.path.join(
                    output_dir, f"spectral_percentiles_{kind}.{output_format}"
                )
                fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                outputs.append(plot_path)
                logger.info("Spectral percentiles plot: %s", plot_path)
            except Exception as exc: #pylint: disable=broad-except
                warnings.append(
                    f"Spectral percentiles plot '{kind}' failed: {exc}"
                )
                logger.warning(warnings[-1])

        return outputs, warnings


register_analysis("spectral_percentiles", SpectralPercentilesAnalysis)

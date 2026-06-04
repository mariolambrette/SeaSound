"""
TOB Levels analysis module.

Computes summary statistics (median, percentiles, max, mean, std) per TOB frequency band
over the entire or windowed deployment period.
"""

import os
import logging

import pandas as pd
import numpy as np

from seasound.analysis.base import (
    AnalysisModule, 
    AnalysisResult, 
    AnalysisModuleError
)
from seasound.analysis.registry import register_analysis


logger = logging.getLogger(__name__)


class TOBLevelsAnalysis(AnalysisModule):
    """
    TOB Levels analysis.
    
    Computes per-frequency summary statistics across time windows.
    Each window produces one row; statistics are columns.
    """

    name = "tob_levels"


    def validate_config(self, cfg: dict) -> None:
        """
        Validate TOB Levels configuration.
        
        Required keys:
        - statistics: list of {"median", "mean", "std", "max", "min", or percentiles as "p5", "p95"}
        
        Optional keys:
        - window_s: window size for statistics; null = entire deployment
        - freq_range: [freq_min_hz, freq_max_hz] or null for all
        - output_format: "csv" (default)
        """
        errors = []

        if "statistics" not in cfg:
            errors.append("tob_levels.config.statistics is required")
        else:
            stats_list = cfg["statistics"]
            if not isinstance(stats_list, list):
                errors.append(
                    f"tob_levels.config.statistics must be a list; "
                    f"got {type(stats_list).__name__}"
                )
            elif not stats_list:
                errors.append("tob_levels.config.statistics cannot be empty")
            else:
                valid_str_stats = {"median", "mean", "std", "max", "min"}
                for stat in stats_list:
                    if not isinstance(stat, str):
                        errors.append(
                            f"tob_levels.config.statistics must contain strings; "
                            f"found {type(stat).__name__}"
                        )
                    elif stat not in valid_str_stats and not (
                        stat.startswith("p") and stat[1:].isdigit()
                    ):
                        errors.append(
                            f"tob_levels.config.statistics contains invalid stat '{stat}'. "
                            f"Valid: {valid_str_stats} or percentiles like 'p5', 'p95'"
                        )

        window_s = cfg.get("window_s")
        if window_s is not None:
            if not isinstance(window_s, (int, float)) or window_s <= 0:
                errors.append(
                    f"tob_levels.config.window_s must be positive numeric or null; "
                    f"got {window_s}"
                )

        freq_range = cfg.get("freq_range")
        if freq_range is not None:
            if not isinstance(freq_range, (list, tuple)) or len(freq_range) != 2:
                errors.append(
                    f"tob_levels.config.freq_range must be [freq_min, freq_max] or null; "
                    f"got {freq_range}"
                )
            elif freq_range[0] >= freq_range[1]:
                errors.append(
                    "tob_levels.config.freq_range must have min < max"
                )

        output_format = cfg.get("output_format", "csv")
        if output_format not in {"csv"}:
            errors.append(
                f"tob_levels.config.output_format must be 'csv'; got '{output_format}'"
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
        Execute TOB Levels analysis.
        """
        self.validate_config(cfg)
        self._validate_base_matrix(base_matrix)

        try:
            # Filter frequencies
            freq_range = cfg.get("freq_range")
            work_matrix = self._filter_frequencies(base_matrix, freq_range)

            # Compute statistics
            window_s = cfg.get("window_s")
            stats_list = cfg.get("statistics", ["median"])

            def _compute_stats(slice_df: pd.DataFrame) -> pd.DataFrame:
                results: dict[str, pd.Series] = {}

                for stat_name in stats_list:
                    if stat_name == "median":
                        results[stat_name] = slice_df.median(axis=0)
                    elif stat_name == "mean":
                        # energy-correct mean for SPL
                        results[stat_name] = self._compute_linear_mean(slice_df)
                    elif stat_name == "std":
                        results[stat_name] = slice_df.std(axis=0)
                    elif stat_name == "max":
                        results[stat_name] = slice_df.max(axis=0)
                    elif stat_name == "min":
                        results[stat_name] = slice_df.min(axis=0)
                    elif stat_name.startswith("p") and stat_name[1:].isdigit():
                        p = int(stat_name[1:])
                        results[stat_name] = slice_df.quantile(p / 100.0, axis=0)
                    else:
                        raise AnalysisModuleError(f"Unknown statistic: {stat_name}")

                out = pd.DataFrame(results)
                out.index.name = "frequency_band"
                return out

            if window_s is None:
                output_df = _compute_stats(work_matrix)
            else:
                window_rule = f"{int(window_s)}s"
                window_frames = []

                for _, group in work_matrix.resample(window_rule):
                    if group.empty:
                        continue

                    if not isinstance(group.index, pd.DatetimeIndex):
                        raise AnalysisModuleError(
                            "TOB Levels windowing requires a DatetimeIndex"
                        )

                    ws = group.index.min()
                    if not isinstance(ws, pd.Timestamp):
                        raise AnalysisModuleError(
                            "Could not determine window start timestamp"
                        )

                    gstats = (
                        _compute_stats(group)
                        .reset_index()
                        .assign(
                            window_start=ws,
                            window_end=ws + pd.Timedelta(seconds=int(window_s)),
                        )
                    )
                    gstats = gstats[[
                        "window_start", 
                        "window_end",
                        "frequency_band", 
                        *stats_list,
                    ]]
                    window_frames.append(gstats)

                if not window_frames:
                    raise AnalysisModuleError("No non-empty windows produced for TOB Levels")

                output_df = pd.concat(window_frames, ignore_index=True)

            # Write output
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "tob_levels.csv")
            output_df.to_csv(output_file)
            logger.info("TOB Levels output: %s", output_file)

            return AnalysisResult(
                name=self.name,
                outputs=[output_file],
                summary={
                    "n_statistics": len(stats_list),
                    "n_frequencies": len(output_df),
                    "statistics": stats_list,
                    "data_points_analyzed": len(work_matrix),
                },
                warnings=[],
            )

        except AnalysisModuleError:
            raise
        except Exception as e:
            raise AnalysisModuleError(f"TOB Levels analysis failed: {e}") from e


    # --- Helper functions ---
    def _compute_linear_mean(self, work_matrix: pd.DataFrame) -> pd.Series:
        """Compute mean SPL via linear power averaging, then convert back to dB."""
        linear_power = np.power(10.0, work_matrix / 10.0)
        mean_linear = linear_power.mean(axis=0)  # one value per frequency column
        return 10.0 * np.log10(mean_linear)


register_analysis("tob_levels", TOBLevelsAnalysis)

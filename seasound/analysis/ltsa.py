"""
LTSA (Long-Term Spectral Average) analysis module for SeaSound.

Resamples a 1-second base matrix to a coarser time resolution and computes
summary statistics per frequency band.
"""

import os
import logging
from typing import Literal

import pandas as pd
import numpy as np

from seasound.analysis.base import (
    AnalysisModule,
    AnalysisResult, 
    AnalysisModuleError,
)
from seasound.analysis.registry import register_analysis

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
        
        if errors:
            raise ValueError("\n".join(errors))
        
    
    def run(
        self,
        base_matrix: pd.DataFrame,
        cfg: dict,
        output_dir: str,
    ) -> AnalysisResult:
        """
        Exeute LTSA analysis.

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
            
            # Write output
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, "ltsa.csv")
            aggregated.to_csv(output_file)
            logger.info(f"LTSA output: {output_file} ({len(aggregated)} rows)")

            return AnalysisResult(
                name=self.name,
                outputs=[output_file],
                summary={
                    "n_time_bins": len(aggregated),
                    "n_frequencies": len(aggregated.columns),
                    "time_resolution": time_resolution,
                    "statistic": statistic,
                    "time_range": f"{aggregated.index.min()} to {aggregated.index.max()}",
                },
                warnings=[],
            )

        except AnalysisModuleError:
            raise
        except Exception as exc:
            raise AnalysisModuleError(f"LTSA analysis failed: {exc}")


    # --- Helper methods ---
    def _compute_linear_mean(
        self, work_matrix: pd.DataFrame, 
        time_resolution: str,
    ) -> pd.DataFrame:
        """Compute mean in linear power domain, then convert back to dB."""
        linear_power = 10 ** (work_matrix / 10)
        aggregated_linear = linear_power.resample(time_resolution).mean()
        return aggregated_linear.apply(lambda col: 10 * np.log10(col))


register_analysis("ltsa", LTSAAnalysis)
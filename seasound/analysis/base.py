"""Base class and common utilities for SeaSound analysis modules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class AnalysisModuleError(Exception):
    """Raised when an analysis module encounters an unrecoverable error."""

@dataclass
class AnalysisResult:
    """Container for results produced by an analysis module."""
    name: str
    outputs: list[str] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


class AnalysisModule(ABC):
    """Abstract base class for all SeaSound analysis modules."""

    name: str

    # Substrates this analysis needs the loader to produce (refactor §7).
    # The default covers every base-matrix analysis (ltsa, tob_levels,
    # spectral_percentiles, the detectors); modules that read the STFT
    # override it (see SpectrogramAnalysis), and modules whose need is
    # conditional on config override ``required_substrates`` instead
    # (see EventDetectionAnalysis).
    REQUIRES: frozenset = frozenset({"base_matrix"})

    @abstractmethod
    def validate_config(self, cfg: dict) -> None:
        """Validate configuration dict for this module."""

    @abstractmethod
    def run(
        self,
        base_matrix: pd.DataFrame,
        cfg: dict,
        output_dir: str,
    ) -> AnalysisResult:
        """Execute the analysis on a base matrix."""

    def required_substrates(self, module_cfg: dict | None = None) -> set[str]:
        """Substrates this analysis needs produced for the given config.

        Defaults to ``REQUIRES``. Override when the requirement is
        conditional on the module's own config (e.g. event detection
        only needs the STFT when its annotated spectrogram is enabled).
        """
        return set(self.REQUIRES)

    # --- Helper methods for common operations ---

    def _validate_base_matrix(self, base_matrix: pd.DataFrame) -> None:
        """Check that base_matrix has expected structure."""
        if base_matrix is None or base_matrix.empty:
            raise AnalysisModuleError("Base matrix is empty.")

        if not isinstance(base_matrix.index, pd.DatetimeIndex):
            raise AnalysisModuleError(
                "Base matrix index must be DateTimeIndex."
            )

        freq_cols = self._get_frequency_columns(base_matrix)
        if not freq_cols:
            raise AnalysisModuleError(
                "No frequency columns found in base matrix."
            )

    def _get_frequency_columns(self, base_matrix: pd.DataFrame) -> list[str]:
        """Return all frequency band column names (ending in Hz)."""
        return [
            c for c in base_matrix.columns
            if isinstance(c, str) and c.endswith("Hz")
        ]

    def _get_frequency_value(self, column_name: str) -> float:
        """Extract numeric frequency from column name."""
        if not column_name.endswith("Hz"):
            raise ValueError(f"Invalid frequency column name: {column_name}")
        return float(column_name[:-2])

    def _filter_frequencies(
        self,
        base_matrix: pd.DataFrame,
        freq_range: tuple[float, float] | None,
    ) -> pd.DataFrame:
        """Filter base_matrix to only include frequencies in range."""
        if freq_range is None:
            return base_matrix.copy()

        freq_min, freq_max = freq_range
        freq_cols = self._get_frequency_columns(base_matrix)

        selected = []
        for col in freq_cols:
            try:
                frq = self._get_frequency_value(col)
                if freq_min <= frq <= freq_max:
                    selected.append(col)
            except ValueError:
                pass

        if not selected:
            raise AnalysisModuleError(
                f"No frequencies found in range [{freq_min}, {freq_max}] Hz"
            )

        return base_matrix[selected].copy()

    def set_runtime_context(
        self,
        context: dict,
    ) -> None:
        """Attach runtime context provided by the pipeline."""
        self._runtime_context = dict(context) # pylint: disable=attribute-defined-outside-init

    def _get_runtime_context(self) -> dict:
        """Access runtime context provided by the pipeline."""
        return getattr(self, "_runtime_context", {})

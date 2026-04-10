"""Base class for SeaSound analysis modules."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd


@dataclass
class AnalysisResult:
    name: str
    outputs: list[str]
    summary: dict
    warnings: list[str]


class AnalysisModule(ABC):
    name: str

    @abstractmethod
    def validate_config(self, cfg: dict) -> None:
        ...

    @abstractmethod
    def run(
        self,
        base_matrix: pd.DataFrame,
        cfg: dict,
        output_dir: str,
    ) -> AnalysisResult:
        ...
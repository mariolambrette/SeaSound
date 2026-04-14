"""Artifacts to store the outputs of data loading as python objects."""

from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd


@dataclass
class SegmentArtifact:
    """In-memory outputs for one processed segment/channel."""
    source_file: str
    channel: int
    serial: str | None
    datetime_start: datetime | None
    calibrated: bool
    base_matrix: pd.DataFrame
    cache_paths: list[str] = field(default_factory=list)


@dataclass
class LoadingOutput:
    """Aggregate results from loading."""
    artifacts: list[SegmentArtifact]

    @property
    def base_matrices(self) -> list[pd.DataFrame]:
        """Convenience method to get the base matrices from all artifacts."""
        return [
            a.base_matrix for a in self.artifacts if a.base_matrix is not None
        ]

"""Registry of SeaSound analysis modules."""

from seasound.analysis.base import AnalysisModule

ANALYSIS_REGISTRY: dict[str, type[AnalysisModule]] = {}


def register_analysis(name: str, cls: type[AnalysisModule]) -> None:
    ANALYSIS_REGISTRY[name] = cls


def get_analysis(name: str) -> AnalysisModule:
    if name not in ANALYSIS_REGISTRY:
        available = ", ".join(sorted(ANALYSIS_REGISTRY.keys()))
        raise ValueError(f"Unknown analysis '{name}'. Available: {available}")
    return ANALYSIS_REGISTRY[name]()
"""
SeaSound plotting package.

Pure plot-generation classes producing matplotlib Figures from analysis
outputs. No I/O is performed inside the plotting layer — callers (analysis
modules, CLI, report generator) are responsible for saving figures to disk.

Public API:
    LTSAPlotter                  — plots for LTSA analysis output
    SpectralPercentilesPlotter   — plots for spectral percentiles output
    PlotStyle                    — shared style defaults
"""

from seasound.plotting._style import PlotStyle
from seasound.plotting.ltsa import LTSAPlotter
from seasound.plotting.spectral_percentiles import SpectralPercentilesPlotter
from seasound.plotting.event_detection import (
    BandThresholdDiagnosticPlotter,
    EventSpectrogramPlotter,
    annotate_events,
)

__all__ = [
    "PlotStyle",
    "LTSAPlotter",
    "SpectralPercentilesPlotter",
    "BandThresholdDiagnosticPlotter",
    "EventSpectrogramPlotter",
    "annotate_events",
]

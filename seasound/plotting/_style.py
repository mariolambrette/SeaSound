"""Shared visual style defaults for SeaSound plots."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PlotStyle:
    """
    Defaults applied to SeaSound plots.

    Frozen so instances are safe to share as a module-level default.
    Override individual fields with dataclasses.replace().
    """

    figsize: tuple[float, float] = (14, 8)
    save_dpi: int = 300
    cmap: str = "viridis"
    gap_color: str = "white"          # masked-cell colour in heatmaps
    title_size: int = 14
    label_size: int = 11
    tick_size: int = 10
    line_width: float = 1.5
    grid_alpha: float = 0.3
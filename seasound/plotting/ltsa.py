"""
LTSA plot generator.

Produces matplotlib Figures from the wide-format DataFrame written by
LTSAAnalysis (DateTimeIndex with one column per TOB band).
"""

import logging
from copy import copy

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from seasound.plotting._common import (
    filter_frequency_range,
    frequency_values,
    get_frequency_columns,
    hz_to_nearest_band,
    reindex_with_gaps,
)
from seasound.plotting._style import PlotStyle

logger = logging.getLogger(__name__)

class LTSAPlotter:
    """
    Plot generator for LTSA analysis output.

    The DataFrame must be in the wide schema written by ``LTSAAnalysis``:
    a DateTimeIndex with one column per TOB band (column names ending in
    'Hz').

    Parameters
    ----------
    df : pd.DataFrame
        LTSA result. DateTimeIndex; columns ending in 'Hz' carry SPL in dB
        re 1 µPa.
    style : PlotStyle, optional
        Visual style overrides. Defaults to a default PlotStyle().
    """

    def __init__(self, df: pd.DataFrame, *, style: PlotStyle | None = None):
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "LTSAPlotter requires a DataFrame with a DatetimeIndex."
            )
        freq_cols = get_frequency_columns(df)
        if not freq_cols:
            raise ValueError(
                "LTSAPlotter: no frequency columns (ending in 'Hz') found."
            )
        self.df = df
        self.style = style or PlotStyle()
        self._freq_cols = freq_cols

    # ------------------------------------------------------------------
    # Public plot methods
    # ------------------------------------------------------------------

    def heatmap(
        self,
        *,
        freq_range: tuple[float, float] | None = None,
        db_range: tuple[float, float] | None = None,
        colormap: str | None = None,
        broadband_strip: bool = False,
        preserve_time_gaps: bool = True,
        title: str | None = None,
    ) -> Figure:
        """
        Render the LTSA as a time-frequency heatmap.

        Parameters
        ----------
        freq_range : (fmin, fmax) in Hz, optional
            Restrict the frequency axis. Default: all bands.
        db_range : (vmin, vmax) in dB, optional
            Colour-scale limits. Default: matplotlib auto-scale.
        colormap : str, optional
            Matplotlib colormap name. Default: style.cmap.
        broadband_strip : bool
            If True, add a thin panel above the heatmap showing the
            broadband SPL time series (mean across bands, computed in
            linear power then converted back to dB).
        preserve_time_gaps : bool
            If True, missing time bins are rendered as blank columns
            (essential for duty-cycled deployments).
        title : str, optional
            Figure title. Defaults to "LTSA".
        """
        from matplotlib.gridspec import GridSpec

        df = filter_frequency_range(self.df, freq_range)
        freq_cols = get_frequency_columns(df)
        if not freq_cols:
            raise ValueError(
                f"heatmap: no frequency bands left after applying "
                f"freq_range={freq_range}."
            )

        df = df[freq_cols]
        if preserve_time_gaps:
            df = reindex_with_gaps(df)

        spec = df.T.values  # shape: (n_freq, n_time)
        if preserve_time_gaps:
            spec = np.ma.masked_invalid(spec)

        freqs = frequency_values(freq_cols)
        times = df.index

        cmap = copy(plt.get_cmap(colormap or self.style.cmap))
        if preserve_time_gaps:
            cmap.set_bad(color=self.style.gap_color, alpha=1.0)

        # --- Figure layout ---
        # Explicit GridSpec puts the colorbar in its own column so it doesn't
        # shrink the heatmap relative to the broadband strip. constrained_layout
        # (rather than tight_layout) handles this composition correctly.
        if broadband_strip:
            fig = plt.figure(
                figsize=self.style.figsize, constrained_layout=True,
            )
            gs = GridSpec(
                2, 2,
                figure=fig,
                width_ratios=[40, 1],     # main column : colorbar column
                height_ratios=[1, 5],     # strip : heatmap
            )
            ax_strip = fig.add_subplot(gs[0, 0])
            ax = fig.add_subplot(gs[1, 0], sharex=ax_strip)
            cax = fig.add_subplot(gs[1, 1])
            self._draw_broadband_strip(ax_strip, df)
        else:
            fig = plt.figure(
                figsize=self.style.figsize, constrained_layout=True,
            )
            gs = GridSpec(
                1, 2,
                figure=fig,
                width_ratios=[40, 1],
            )
            ax = fig.add_subplot(gs[0, 0])
            cax = fig.add_subplot(gs[0, 1])
            ax_strip = None

        # --- Heatmap ---
        im = ax.imshow(
            spec,
            aspect="auto",
            cmap=cmap,
            vmin=db_range[0] if db_range else None,
            vmax=db_range[1] if db_range else None,
            origin="lower",
        )

        # Frequency axis (TOB index → labelled in Hz)
        n_freq_ticks = min(10, len(freqs))
        freq_idx = np.linspace(0, len(freqs) - 1, n_freq_ticks, dtype=int)
        ax.set_yticks(freq_idx)
        ax.set_yticklabels(
            [f"{freqs[i]:.0f}" for i in freq_idx],
            fontsize=self.style.tick_size,
        )
        ax.set_ylabel("Frequency (Hz)", fontsize=self.style.label_size)

        # Time axis
        n_time_ticks = min(10, len(times))
        time_idx = np.linspace(0, len(times) - 1, n_time_ticks, dtype=int)
        ax.set_xticks(time_idx)
        ax.set_xticklabels(
            [str(times[i])[:16] for i in time_idx],
            rotation=45, fontsize=self.style.tick_size, ha="right",
        )
        ax.set_xlabel("Time (UTC)", fontsize=self.style.label_size)

        # Title placement (above strip if present, else above heatmap)
        top_ax = ax_strip if broadband_strip else ax
        top_ax.set_title(title or "LTSA", fontsize=self.style.title_size) # type: ignore

        # Colorbar — in its own GridSpec cell so it doesn't shrink the heatmap
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("SPL (dB re 1 µPa)", fontsize=self.style.label_size)
        cbar.ax.tick_params(labelsize=self.style.tick_size)

        # No tight_layout — constrained_layout=True handles spacing including
        # the colorbar and gridspec cells.
        return fig

    def band_timeseries(
        self,
        *,
        bands_hz: list[float],
        db_range: tuple[float, float] | None = None,
        preserve_time_gaps: bool = True,
        title: str | None = None,
    ) -> Figure:
        """
        Plot SPL time series for a set of TOB bands.

        Parameters
        ----------
        bands_hz : list of float
            Target frequencies in Hz. Each is snapped to the nearest TOB
            column; duplicates after snapping are dropped.
        db_range : (vmin, vmax) in dB, optional
            Y-axis limits.
        preserve_time_gaps : bool
            If True, missing time bins render as line breaks.
        title : str, optional
            Figure title.
        """
        if not bands_hz:
            raise ValueError("bands_hz must contain at least one frequency.")

        df = self.df[self._freq_cols]
        if preserve_time_gaps:
            df = reindex_with_gaps(df)

        # Snap each requested Hz to nearest band; preserve order, drop dupes.
        seen: set[str] = set()
        selected: list[str] = []
        for hz in bands_hz:
            col = hz_to_nearest_band(hz, self._freq_cols)
            if col not in seen:
                seen.add(col)
                selected.append(col)

        fig, ax = plt.subplots(figsize=self.style.figsize)
        for col in selected:
            actual_hz = float(col[:-2])
            ax.plot(
                df.index, df[col].values, # type: ignore
                linewidth=self.style.line_width,
                label=f"{actual_hz:g} Hz",
            )

        ax.set_xlabel("Time (UTC)", fontsize=self.style.label_size)
        ax.set_ylabel("SPL (dB re 1 µPa)", fontsize=self.style.label_size)
        if db_range:
            ax.set_ylim(db_range)
        ax.set_title(
            title or "LTSA — selected bands",
            fontsize=self.style.title_size,
        )
        ax.legend(fontsize=self.style.tick_size, frameon=False)
        ax.grid(alpha=self.style.grid_alpha)
        ax.tick_params(labelsize=self.style.tick_size)
        fig.autofmt_xdate()
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_broadband_strip(self, ax, df: pd.DataFrame) -> None:
        """
        Plot broadband SPL above the heatmap.

        X-coordinates are integer indices (0..n-1) to match the imshow
        coordinate system used by the heatmap, with which this axes shares
        an X-axis. Time labels appear on the heatmap below.

        Mean is computed in linear power and converted back to dB so that
        a few loud bands don't dominate the average (matches the LTSA
        module's own linear-mean convention).
        """
        with np.errstate(invalid="ignore", divide="ignore"):
            power = 10 ** (df.values / 10.0)
            broadband_power = np.nanmean(power, axis=1)
            broadband_db = 10 * np.log10(broadband_power)

        ax.plot(
            range(len(df)), broadband_db,
            color="k", linewidth=self.style.line_width,
        )
        ax.set_ylabel(
            "Broadband\nSPL (dB)", fontsize=self.style.label_size,
        )
        ax.tick_params(labelbottom=False, labelsize=self.style.tick_size)
        ax.grid(alpha=self.style.grid_alpha)

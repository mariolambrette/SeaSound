"""
Event-detection plotters.

Two responsibilities:

1. annotate_events / EventSpectrogramPlotter — overlay bounding-box
   annotations on an STFT spectrogram for any detector's events.
2. BandThresholdDiagnosticPlotter (chunk 4) — per-band value/baseline/
   threshold panels for the band_threshold detector.

All pure: figures returned, callers save.
"""

import logging
from copy import copy

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from seasound.plotting._common import (
    filter_frequency_range,
    frequency_values,
    get_frequency_columns,
    reindex_with_gaps,
)
from seasound.plotting._style import PlotStyle
from seasound.utils.spectral import tob_band_edges

logger = logging.getLogger(__name__)


# Built-in defaults: callers can override via per-detector style overrides.
DEFAULT_DETECTOR_STYLES: dict[str, dict] = {
    "band_threshold":            {"edge_color": "red"},
    "adaptive_threshold_legacy": {"edge_color": "orange",
                                  "face_color": "orange", "alpha": 0.15},
    "anomaly":                   {"edge_color": "deepskyblue"},
}

DEFAULT_ANNOTATION_STYLE: dict = {
    "edge_color": "red",
    "face_color": "none",
    "line_width": 1.5,
    "alpha":      0.8,
    "label":      False,
}


def _band_edges_for_hz(band_hz: float) -> tuple[float, float]:
    """Lower and upper edge of the TOB band centred at ``band_hz``."""
    edges = tob_band_edges(np.array([band_hz]))
    return float(edges[0, 0]), float(edges[0, 1])


def _resolved_style(
    detector: str,
    annotation_styles: dict[str, dict] | None,
) -> dict:
    """
    Resolve the final style dict for an event from this detector.

    Precedence (low to high):
        DEFAULT_ANNOTATION_STYLE  ←  built-in DEFAULT_DETECTOR_STYLES[det]
                                  ←  user annotation_styles[det]
    """
    style = dict(DEFAULT_ANNOTATION_STYLE)
    style.update(DEFAULT_DETECTOR_STYLES.get(detector, {}))
    if annotation_styles:
        style.update(annotation_styles.get(detector, {}))
    return style

# ---------------------------------------------------------------------------
# annotate_events: detector-agnostic overlay function
# ---------------------------------------------------------------------------

def annotate_events(
    ax,
    events_df: pd.DataFrame,
    *,
    annotation_styles: dict[str, dict] | None = None,
    show_legend: bool = True,
    legend_loc: str = "upper right",
) -> None:
    """
    Draw event annotations onto an existing Axes.

    Each row in ``events_df`` is interpreted as follows:

    - If a ``band_hz`` column is present and non-null for the row, the
      event is drawn as a rectangle spanning ``[start_time, end_time]``
      × the TOB band edges around ``band_hz``.
    - Otherwise the event is treated as broadband and drawn as a vertical
      span across the full Y range.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes. Datetime X data and Hz Y data are assumed (i.e.
        produced by EventSpectrogramPlotter via pcolormesh).
    events_df : pd.DataFrame
        Must contain at least ``detector``, ``start_time``, ``end_time``.
        ``band_hz`` is optional.
    annotation_styles : dict[str, dict], optional
        Per-detector style overrides, e.g.
        ``{"band_threshold": {"edge_color": "blue"}}``.
        Keys not provided fall back to built-in defaults.
    show_legend : bool
        If True and more than one detector contributed events, draw a
        small legend with one entry per detector.
    """
    if events_df is None or events_df.empty:
        return

    events_df = events_df.copy()
    events_df["start_time"] = pd.to_datetime(events_df["start_time"])
    events_df["end_time"] = pd.to_datetime(events_df["end_time"])
    has_band = "band_hz" in events_df.columns

    detectors_seen: dict[str, dict] = {}

    for _, ev in events_df.iterrows():
        det = str(ev["detector"])
        style = _resolved_style(det, annotation_styles)
        detectors_seen.setdefault(det, style)

        x0 = mdates.date2num(ev["start_time"])
        x1 = mdates.date2num(ev["end_time"])
        width = x1 - x0

        if has_band and pd.notna(ev.get("band_hz")):
            band_hz = float(ev["band_hz"])
            y0, y1 = _band_edges_for_hz(band_hz)
            rect = Rectangle(
                (x0, y0), width, y1 - y0, #type: ignore
                edgecolor=style["edge_color"],
                facecolor=style["face_color"],
                linewidth=style["line_width"],
                alpha=style["alpha"],
            )
            ax.add_patch(rect)
            if style.get("label"):
                ax.text(
                    x0, y1, f" {ev.get('event_id', '')}",
                    fontsize=7, color=style["edge_color"],
                    va="top", ha="left",
                )
        else:
            ax.axvspan(
                ev["start_time"], ev["end_time"],
                edgecolor=style["edge_color"],
                facecolor=(
                    style["face_color"]
                    if style["face_color"] != "none"
                    else style["edge_color"]
                ),
                linewidth=style["line_width"],
                alpha=min(style["alpha"], 0.25),
            )

    if show_legend and len(detectors_seen) > 1:
        handles = [
            Patch(
                facecolor="none",
                edgecolor=s["edge_color"],
                linewidth=s["line_width"],
                label=det,
            )
            for det, s in detectors_seen.items()
        ]
        ax.legend(handles=handles, loc=legend_loc, frameon=False, fontsize=8)


# ---------------------------------------------------------------------------
# EventSpectrogramPlotter
# ---------------------------------------------------------------------------

class EventSpectrogramPlotter:
    """
    Annotated STFT spectrogram.

    Renders the spectrogram as a pcolormesh with real datetime/Hz
    coordinates and overlays event annotations from one or more
    detectors. Pcolormesh is used (rather than imshow) so that Rectangle
    and axvspan annotations land at the correct (datetime, Hz)
    coordinates without index conversion. The visual is therefore not
    identical to ``SpectrogramAnalysis``'s imshow-based output — this is
    intentional.

    Parameters
    ----------
    stft_matrix : pd.DataFrame
        STFT-derived matrix with DateTimeIndex and Hz-named columns
        (the output of ``build_stft_matrix``).
    events_by_detector : dict[str, pd.DataFrame]
        Mapping detector name → events DataFrame. Empty is fine — the
        plot then shows the spectrogram with no overlays.
    style : PlotStyle, optional
    """

    def __init__(
        self,
        stft_matrix: pd.DataFrame,
        events_by_detector: dict[str, pd.DataFrame],
        *,
        style: PlotStyle | None = None,
    ):
        if not isinstance(stft_matrix.index, pd.DatetimeIndex):
            raise ValueError(
                "EventSpectrogramPlotter requires stft_matrix to have a "
                "DatetimeIndex."
            )
        freq_cols = get_frequency_columns(stft_matrix)
        if not freq_cols:
            raise ValueError(
                "EventSpectrogramPlotter: stft_matrix has no frequency "
                "columns (names ending in 'Hz')."
            )
        self.stft_matrix = stft_matrix
        self.events_by_detector = events_by_detector or {}
        self.style = style or PlotStyle()

    def plot(
        self,
        *,
        freq_range: tuple[float, float] | None = None,
        db_range: tuple[float, float] | None = None,
        colormap: str | None = None,
        preserve_time_gaps: bool = True,
        annotation_styles: dict[str, dict] | None = None,
        annotation_label: bool = False,
        title: str | None = None,
    ) -> Figure:
        """
        Produce the annotated spectrogram.

        Parameters
        ----------
        freq_range : (fmin, fmax) in Hz, optional
        db_range : (vmin, vmax) in dB, optional
        colormap : str, optional
            Matplotlib colormap name. Defaults to ``style.cmap``.
        preserve_time_gaps : bool
            If True, missing time bins render as blank cells.
        annotation_styles : dict[str, dict], optional
            Per-detector style overrides; falls through to built-in
            defaults for detectors not listed.
        annotation_label : bool
            Convenience: when True, event ids are drawn at the corner of
            each rectangle.
        title : str, optional
        """
        from matplotlib.gridspec import GridSpec

        df = filter_frequency_range(self.stft_matrix, freq_range)
        freq_cols = get_frequency_columns(df)
        if not freq_cols:
            raise ValueError(
                f"EventSpectrogramPlotter: no frequency bands left after "
                f"applying freq_range={freq_range}."
            )
        df = df[freq_cols]
        if preserve_time_gaps:
            df = reindex_with_gaps(df)

        freqs = np.asarray(frequency_values(freq_cols), dtype=float)
        times = df.index
        C = ( #pylint: disable=invalid-name
            np.ma.masked_invalid(df.T.values)
            if preserve_time_gaps else df.T.values
        )

        cmap = copy(plt.get_cmap(colormap or self.style.cmap))
        if preserve_time_gaps:
            cmap.set_bad(color=self.style.gap_color, alpha=1.0)

        # Colorbar in its own GridSpec column (avoids tight_layout warnings)
        fig = plt.figure(
            figsize=self.style.figsize, constrained_layout=True,
        )
        gs = GridSpec(1, 2, figure=fig, width_ratios=[40, 1])
        ax = fig.add_subplot(gs[0, 0])
        cax = fig.add_subplot(gs[0, 1])

        mesh = ax.pcolormesh(
            times, freqs, C,
            cmap=cmap,
            vmin=db_range[0] if db_range else None,
            vmax=db_range[1] if db_range else None,
            shading="auto",
        )
        ax.set_xlabel("Time (UTC)", fontsize=self.style.label_size)
        ax.set_ylabel("Frequency (Hz)", fontsize=self.style.label_size)
        ax.tick_params(labelsize=self.style.tick_size)
        ax.xaxis_date()
        fig.autofmt_xdate()
        ax.set_title(
            title or "Spectrogram with detected events",
            fontsize=self.style.title_size,
        )

        cbar = fig.colorbar(mesh, cax=cax)
        cbar.set_label(
            "SPL (dB re 1 µPa)", fontsize=self.style.label_size,
        )
        cbar.ax.tick_params(labelsize=self.style.tick_size)

        # Overlay events
        events = self._collect_events(freq_range=freq_range)
        if not events.empty:
            styles = dict(annotation_styles or {})
            if annotation_label:
                # Apply label=True to every detector (simplest path)
                for det in events["detector"].unique():
                    styles.setdefault(det, {})
                    styles[det] = {**styles[det], "label": True}
            annotate_events(
                ax, events,
                annotation_styles=styles,
                show_legend=True,
            )

        return fig

    def _collect_events(
        self,
        *,
        freq_range: tuple[float, float] | None,
    ) -> pd.DataFrame:
        """Concatenate events from all detectors; filter band events by freq_range."""
        frames = []
        for det, df in self.events_by_detector.items():
            if df is None or df.empty:
                continue
            d = df.copy()
            d["detector"] = det
            frames.append(d)
        if not frames:
            return pd.DataFrame()
        merged = pd.concat(frames, ignore_index=True)

        if freq_range is not None and "band_hz" in merged.columns:
            fmin, fmax = freq_range
            in_range = merged["band_hz"].between(fmin, fmax)
            no_band = merged["band_hz"].isna()
            merged = merged[in_range | no_band]
        return merged

# ---------------------------------------------------------------------------
# BandThresholdDiagnosticPlotter
# ---------------------------------------------------------------------------

def _pool_series(
    series: pd.Series,
    target: int,
    agg: str = "max",
) -> pd.Series:
    """
    Downsample a Series to approximately ``target`` points by fixed-size
    block pooling.

    ``agg`` is the aggregation method passed to ``groupby`` ("max" or
    "mean" expected). Returns a Series whose index is taken from the
    first element of each block.
    """
    n = len(series)
    if n <= target:
        return series
    block = -(-n // target)         # ceil(n / target)
    n_blocks = -(-n // block)
    groups = np.arange(n) // block
    pooled = getattr(series.groupby(groups), agg)()
    return pd.Series(
        pooled.values,
        index=series.index[::block][:n_blocks],
    )


def _nearest_band_column(target_hz: float, columns: list[str]) -> str:
    freqs = frequency_values(columns)
    return columns[int(np.argmin([abs(f - target_hz) for f in freqs]))]


class BandThresholdDiagnosticPlotter:
    """
    Per-band diagnostic plot for the band_threshold detector.

    Each panel shows, for one TOB band:

    - the per-sample SPL value (solid black line),
    - the rolling threshold (dashed red),
    - the rolling baseline (dotted grey),
    - red fill where ``value > threshold`` (raw flag mask),
    - light-red vertical spans at the start/end of every merged event for
      the band (post min_duration_s / merge_gap_s).

    Long deployments are downsampled to ``max_points_per_panel`` (default
    5000) using max-pooling for value/threshold and mean-pooling for
    baseline. Events themselves are drawn at their actual timestamps.

    Parameters
    ----------
    diagnostics : BandThresholdDiagnostics
        As returned by ``BandThresholdDetector.detect_with_diagnostics``.
    events_df : pd.DataFrame
        The band_threshold events DataFrame (matched per panel via the
        ``band_hz`` column).
    style : PlotStyle, optional
    """

    def __init__(
        self,
        diagnostics,  # BandThresholdDiagnostics; not imported to avoid cycle
        events_df: pd.DataFrame,
        *,
        style: PlotStyle | None = None,
    ):
        # Duck-typed validation so we don't need to import the dataclass
        # at import time (avoids any chance of circular imports).
        for attr in ("values", "baseline", "threshold"):
            if not hasattr(diagnostics, attr):
                raise ValueError(
                    f"BandThresholdDiagnosticPlotter: diagnostics is "
                    f"missing attribute '{attr}'."
                )
            df = getattr(diagnostics, attr)
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError(
                    f"BandThresholdDiagnosticPlotter: diagnostics.{attr} "
                    f"must have a DatetimeIndex."
                )

        self.values = diagnostics.values
        self.baseline = diagnostics.baseline
        self.threshold = diagnostics.threshold

        events_df = (
            events_df.copy()
            if events_df is not None
            else pd.DataFrame()
        )
        if not events_df.empty:
            events_df["start_time"] = pd.to_datetime(events_df["start_time"])
            events_df["end_time"] = pd.to_datetime(events_df["end_time"])
        self.events_df = events_df

        self.style = style or PlotStyle()
        self._band_cols = list(self.values.columns)

    def per_band(
        self,
        *,
        bands_hz: list[float],
        max_points_per_panel: int = 5000,
        title: str | None = None,
    ) -> Figure:
        """
        Render one panel per band in ``bands_hz``.

        Bands are snapped to the nearest available diagnostic column;
        duplicates after snapping are dropped (input order preserved).
        """
        if not bands_hz:
            raise ValueError("bands_hz must contain at least one frequency.")

        seen: set[str] = set()
        selected: list[str] = []
        for hz in bands_hz:
            col = _nearest_band_column(hz, self._band_cols)
            if col not in seen:
                seen.add(col)
                selected.append(col)

        n = len(selected)
        figsize = (self.style.figsize[0], max(3.0, 2.5 * n))
        fig, axes = plt.subplots(
            n, 1,
            figsize=figsize, sharex=True, squeeze=False,
            constrained_layout=True,
        )
        axes = axes.ravel()

        for i, col in enumerate(selected):
            self._draw_panel(axes[i], col, max_points_per_panel)

        axes[-1].set_xlabel("Time (UTC)", fontsize=self.style.label_size)
        fig.autofmt_xdate()
        fig.suptitle(
            title or "Band-threshold diagnostic",
            fontsize=self.style.title_size,
        )
        return fig

    def _draw_panel(
        self, ax, band_col: str, max_points_per_panel: int,
    ) -> None:
        band_hz = float(band_col[:-2])

        value_ds = _pool_series(
            self.values[band_col], max_points_per_panel, "max",
        )
        thresh_ds = _pool_series(
            self.threshold[band_col], max_points_per_panel, "max",
        )
        base_ds = _pool_series(
            self.baseline[band_col], max_points_per_panel, "mean",
        )

        ax.plot(
            value_ds.index, value_ds.values,
            color="black", linewidth=self.style.line_width,
            label="value",
        )
        ax.plot(
            thresh_ds.index, thresh_ds.values,
            color="red", linewidth=self.style.line_width,
            linestyle="--", label="threshold",
        )
        ax.plot(
            base_ds.index, base_ds.values,
            color="grey", linewidth=self.style.line_width * 0.8,
            linestyle=":", label="baseline",
        )

        # Per-sample exceedance shading
        with np.errstate(invalid="ignore"):
            exceed = (
                (value_ds.values > thresh_ds.values) #type: ignore
                & np.isfinite(value_ds.values)
            )
        ax.fill_between(
            value_ds.index, thresh_ds.values, value_ds.values,
            where=exceed,
            color="red", alpha=0.15,
            label="exceedance",
        )

        # Merged events for this band
        if not self.events_df.empty and "band_hz" in self.events_df.columns:
            mask = np.isclose(
                self.events_df["band_hz"].to_numpy(dtype=float), band_hz,
            )
            for _, ev in self.events_df[mask].iterrows():
                ax.axvspan(
                    ev["start_time"], ev["end_time"],
                    color="red", alpha=0.08,
                )

        ax.set_ylabel(
            f"{band_hz:g} Hz\nSPL (dB)",
            fontsize=self.style.label_size,
        )
        ax.tick_params(labelsize=self.style.tick_size)
        ax.grid(alpha=self.style.grid_alpha)
        # Legend only on the top panel
        if ax is ax.figure.axes[0]:
            ax.legend(
                fontsize=self.style.tick_size,
                frameon=False, loc="upper right",
            )

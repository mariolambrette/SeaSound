"""
Spectral-percentiles plot generator.

Reads the CSV schema written by SpectralPercentilesAnalysis:

- "full" mode: one row, columns of the form '<freq>Hz_p<N>'.
- Windowed mode: 'window_start', 'window_end' columns plus '<freq>Hz_p<N>'
  columns; one row per window.

Produces percentile-vs-frequency curve plots. Windowed inputs yield a grid
of panels (one per window), subsampled evenly across the time range when
the number of windows exceeds max_panels.
"""

import logging
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

from seasound.plotting._common import ( #pylint: disable=unused-import
    compute_grid_dims,
    format_time_label,
    subsample_evenly,
    _validate_plot_block,
)
from seasound.plotting._style import PlotStyle

logger = logging.getLogger(__name__)


_PERCENTILE_COL_RE = re.compile(r"^(?P<f>.+?)Hz_p(?P<p>\d+)$")


def _parse_percentile_columns(
    columns: list[str],
) -> dict[int, list[tuple[float, str]]]:
    """
    Parse '<freq>Hz_p<N>' columns into a mapping
    {percentile -> [(freq_hz, column_name), ...]} sorted by frequency.

    Columns that don't match the expected pattern are silently ignored.
    """
    by_p: dict[int, list[tuple[float, str]]] = {}
    for col in columns:
        if not isinstance(col, str):
            continue
        m = _PERCENTILE_COL_RE.match(col)
        if not m:
            continue
        freq = float(m.group("f"))
        pct = int(m.group("p"))
        by_p.setdefault(pct, []).append((freq, col))
    return {p: sorted(items, key=lambda t: t[0]) for p, items in by_p.items()}


class SpectralPercentilesPlotter:
    """
    Plot generator for spectral_percentiles analysis output.

    Detects the schema automatically: if 'window_start' and 'window_end'
    columns are present, the DataFrame is treated as windowed; otherwise
    as single-deployment ('full').

    Parameters
    ----------
    df : pd.DataFrame
        Spectral percentiles result.
    style : PlotStyle, optional
        Visual style overrides.
    """

    def __init__(self, df: pd.DataFrame, *, style: PlotStyle | None = None):
        cols = list(df.columns)
        self.windowed = "window_start" in cols and "window_end" in cols

        pct_cols = [c for c in cols if c not in ("window_start", "window_end")]
        self._percentile_map = _parse_percentile_columns(pct_cols)
        if not self._percentile_map:
            raise ValueError(
                "SpectralPercentilesPlotter: no percentile columns found. "
                "Expected columns of the form '<freq>Hz_p<N>'."
            )

        # Coerce window timestamps if applicable so they're plottable.
        if self.windowed:
            df = df.copy()
            df["window_start"] = pd.to_datetime(df["window_start"])
            df["window_end"] = pd.to_datetime(df["window_end"])

        self.df = df
        self.style = style or PlotStyle()

    # ------------------------------------------------------------------
    # Public plot methods
    # ------------------------------------------------------------------

    def curves(
        self,
        *,
        freq_range: tuple[float, float] | None = None,
        db_range: tuple[float, float] | None = None,
        shaded_band: bool = False,
        shaded_percentiles: tuple[int, int] = (5, 95),
        log_freq: bool = True,
        max_panels: int = 16,
        title: str | None = None,
    ) -> Figure:
        """
        Plot percentile vs frequency curves.

        For a 'full'-window dataset this produces one panel.
        For a windowed dataset it produces a grid of panels (one per
        window). If the number of windows exceeds ``max_panels``, the
        windows are evenly subsampled across the time range.

        Parameters
        ----------
        freq_range : (fmin, fmax) in Hz, optional
            Restrict the frequency axis.
        db_range : (vmin, vmax) in dB, optional
            Y-axis limits.
        shaded_band : bool
            If True, shade the area between ``shaded_percentiles``.
        shaded_percentiles : (lo, hi)
            Which two percentiles bound the shaded region.
        log_freq : bool
            Use a logarithmic frequency axis.
        max_panels : int
            Maximum number of panels for windowed data (default 16).
        title : str, optional
            Figure title.
        """
        if self.windowed:
            return self._curves_grid(
                freq_range=freq_range,
                db_range=db_range,
                shaded_band=shaded_band,
                shaded_percentiles=shaded_percentiles,
                log_freq=log_freq,
                max_panels=max_panels,
                title=title,
            )
        return self._curves_single(
            freq_range=freq_range,
            db_range=db_range,
            shaded_band=shaded_band,
            shaded_percentiles=shaded_percentiles,
            log_freq=log_freq,
            title=title,
        )

    # ------------------------------------------------------------------
    # Internal: single-panel ('full') mode
    # ------------------------------------------------------------------

    def _curves_single(
        self, *, freq_range, db_range, shaded_band, shaded_percentiles,
        log_freq, title,
    ) -> Figure:
        fig, ax = plt.subplots(figsize=self.style.figsize)
        self._draw_curves(
            ax, self.df.iloc[0],
            freq_range=freq_range,
            db_range=db_range,
            shaded_band=shaded_band,
            shaded_percentiles=shaded_percentiles,
            log_freq=log_freq,
            legend=True,
        )
        ax.set_xlabel("Frequency (Hz)", fontsize=self.style.label_size)
        ax.set_ylabel("SPL (dB re 1 µPa)", fontsize=self.style.label_size)
        ax.set_title(
            title or "Spectral percentiles",
            fontsize=self.style.title_size,
        )
        ax.grid(alpha=self.style.grid_alpha)
        fig.tight_layout()
        return fig

    # ------------------------------------------------------------------
    # Internal: grid mode for windowed data
    # ------------------------------------------------------------------

    def _curves_grid(
        self, *, freq_range, db_range, shaded_band, shaded_percentiles,
        log_freq, max_panels, title,
    ) -> Figure:
        n_total = len(self.df)
        all_indices = list(range(n_total))
        sel = subsample_evenly(all_indices, max_panels)
        n = len(sel)
        n_rows, n_cols = compute_grid_dims(n)

        # Scale figure height with the grid row count to keep aspect ratios sane.
        height_scale = max(1.0, n_rows / 2.0)
        figsize = (self.style.figsize[0], self.style.figsize[1] * height_scale)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=figsize,
            sharex=True, sharey=True,
            squeeze=False,
        )

        for i, row_i in enumerate(sel):
            ax = axes[i // n_cols, i % n_cols]
            self._draw_curves(
                ax, self.df.iloc[row_i],
                freq_range=freq_range,
                db_range=db_range,
                shaded_band=shaded_band,
                shaded_percentiles=shaded_percentiles,
                log_freq=log_freq,
                legend=(i == 0),
            )
            window_label = format_time_label(self.df.iloc[row_i]["window_start"])
            ax.set_title(window_label, fontsize=self.style.tick_size)
            ax.grid(alpha=self.style.grid_alpha)

        # Hide unused axes in the trailing positions
        for j in range(n, n_rows * n_cols):
            axes[j // n_cols, j % n_cols].axis("off")

        fig.supxlabel("Frequency (Hz)", fontsize=self.style.label_size)
        fig.supylabel("SPL (dB re 1 µPa)", fontsize=self.style.label_size)

        suptitle = title or "Spectral percentiles (windowed)"
        if n < n_total:
            suptitle += (
                f"  [{n} of {n_total} windows shown, evenly sampled]"
            )
            logger.info(
                "SpectralPercentilesPlotter: subsampled %d windows to %d "
                "panels.", n_total, n,
            )
        fig.suptitle(suptitle, fontsize=self.style.title_size)
        fig.tight_layout(rect=(0.02, 0.02, 1.0, 0.96))
        return fig

    # ------------------------------------------------------------------
    # Internal: shared single-axes drawer
    # ------------------------------------------------------------------

    def _draw_curves(
        self, ax, row, *, freq_range, db_range, shaded_band,
        shaded_percentiles, log_freq, legend,
    ) -> None:
        """Draw all percentile curves for one DataFrame row onto an Axes."""
        sorted_pcts = sorted(self._percentile_map.keys())

        if shaded_band:
            lo, hi = shaded_percentiles
            if lo not in self._percentile_map or hi not in self._percentile_map:
                logger.warning(
                    "shaded_percentiles=%s not in computed percentiles %s; "
                    "skipping shaded band.",
                    shaded_percentiles, sorted_pcts,
                )
            else:
                pairs_lo = self._percentile_map[lo]
                pairs_hi = self._percentile_map[hi]
                fr, vlo = self._series_for(row, pairs_lo, freq_range)
                _, vhi = self._series_for(row, pairs_hi, freq_range)
                ax.fill_between(
                    fr, vlo, vhi,
                    alpha=0.15, color="grey",
                    label=f"P{lo}–P{hi}",
                )

        for p in sorted_pcts:
            pairs = self._percentile_map[p]
            fr, vals = self._series_for(row, pairs, freq_range)
            ax.plot(
                fr, vals,
                linewidth=self.style.line_width,
                label=f"P{p}",
            )

        if log_freq:
            ax.set_xscale("log")
        if db_range:
            ax.set_ylim(db_range)
        ax.tick_params(labelsize=self.style.tick_size)
        if legend:
            ax.legend(fontsize=self.style.tick_size, frameon=False, loc="best")

    @staticmethod
    def _series_for(
        row,
        pairs: list[tuple[float, str]],
        freq_range: tuple[float, float] | None,
    ) -> tuple[list[float], list[float]]:
        """Return (frequencies, values) for one row, optionally freq-filtered."""
        if freq_range is None:
            freqs = [f for f, _ in pairs]
            vals = [row[c] for _, c in pairs]
            return freqs, vals
        fmin, fmax = freq_range
        freqs: list[float] = []
        vals: list[float] = []
        for f, c in pairs:
            if fmin <= f <= fmax:
                freqs.append(f)
                vals.append(row[c])
        return freqs, vals

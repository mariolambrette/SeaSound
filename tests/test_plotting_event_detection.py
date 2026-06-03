"""Tests for event-detection plotters."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.patches import Rectangle

from seasound.plotting.event_detection import (
    EventSpectrogramPlotter,
    _band_edges_for_hz,
    annotate_events,
)
from seasound.analysis.event_detection import BandThresholdDiagnostics
from seasound.plotting.event_detection import BandThresholdDiagnosticPlotter


@pytest.fixture
def stft_matrix():
    """6h of 10-second STFT frames, 50 freq bins from 50 to 20000 Hz."""
    idx = pd.date_range("2024-01-01", periods=6 * 360, freq="10s")
    freqs = np.geomspace(50, 20000, 50)
    cols = [f"{f:.1f}Hz" for f in freqs]
    rng = np.random.default_rng(0)
    data = 70 + 8 * rng.standard_normal((len(idx), len(cols)))
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.fixture
def band_events():
    """Two per-band events from band_threshold."""
    return pd.DataFrame({
        "detector":   ["band_threshold", "band_threshold"],
        "event_id":   [1, 2],
        "start_time": pd.to_datetime(
            ["2024-01-01 01:00", "2024-01-01 03:30"]
        ),
        "end_time":   pd.to_datetime(
            ["2024-01-01 01:10", "2024-01-01 03:50"]
        ),
        "band_hz":    [1000.0, 4000.0],
    })


@pytest.fixture
def broadband_events():
    return pd.DataFrame({
        "detector":   ["adaptive_threshold_legacy"],
        "event_id":   [1],
        "start_time": pd.to_datetime(["2024-01-01 02:00"]),
        "end_time":   pd.to_datetime(["2024-01-01 02:15"]),
    })


class TestBandEdges:
    def test_band_edges_ratio_matches_iec(self):
        """Upper/lower ratio should match 10^(1/10) ≈ third-octave (IEC)."""
        lo, hi = _band_edges_for_hz(1000.0)
        assert hi / lo == pytest.approx(10 ** (1 / 10), rel=1e-6)


class TestAnnotateEvents:
    def test_band_event_draws_rectangle(self, band_events):
        fig, ax = plt.subplots()
        ax.set_ylim(50, 20000)
        ax.set_xlim(
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01 06:00"),
        )
        annotate_events(ax, band_events)
        rects = [p for p in ax.patches if isinstance(p, Rectangle)]
        assert len(rects) == 2
        plt.close(fig)

    def test_broadband_event_draws_axvspan(self, broadband_events):
        fig, ax = plt.subplots()
        ax.set_ylim(50, 20000)
        annotate_events(ax, broadband_events)
        assert len(ax.patches) >= 1
        plt.close(fig)

    def test_empty_events_is_noop(self):
        fig, ax = plt.subplots()
        annotate_events(ax, pd.DataFrame())
        assert len(ax.patches) == 0
        plt.close(fig)

    def test_legend_when_multiple_detectors(
        self, band_events, broadband_events,
    ):
        fig, ax = plt.subplots()
        ax.set_ylim(50, 20000)
        merged = pd.concat([band_events, broadband_events], ignore_index=True)
        annotate_events(ax, merged, show_legend=True)
        assert ax.get_legend() is not None
        plt.close(fig)

    def test_no_legend_when_single_detector(self, band_events):
        fig, ax = plt.subplots()
        ax.set_ylim(50, 20000)
        annotate_events(ax, band_events, show_legend=True)
        assert ax.get_legend() is None
        plt.close(fig)

    def test_style_override(self, band_events):
        fig, ax = plt.subplots()
        ax.set_ylim(50, 20000)
        annotate_events(
            ax, band_events,
            annotation_styles={"band_threshold": {"edge_color": "lime"}},
        )
        rects = [p for p in ax.patches if isinstance(p, Rectangle)]
        assert rects[0].get_edgecolor()[:3] == pytest.approx(
            (0.0, 1.0, 0.0), abs=1e-6,
        )
        plt.close(fig)


class TestEventSpectrogramPlotter:
    def test_basic_plot(self, stft_matrix, band_events):
        plotter = EventSpectrogramPlotter(
            stft_matrix, {"band_threshold": band_events},
        )
        fig = plotter.plot(db_range=(40, 100))
        assert fig is not None
        plt.close(fig)

    def test_no_events_renders_spectrogram(self, stft_matrix):
        plotter = EventSpectrogramPlotter(stft_matrix, {})
        fig = plotter.plot()
        plt.close(fig)

    def test_freq_range_filters_events(self, stft_matrix, band_events):
        plotter = EventSpectrogramPlotter(
            stft_matrix, {"band_threshold": band_events},
        )
        fig = plotter.plot(freq_range=(500, 2000))
        rects = [
            p for p in fig.axes[0].patches if isinstance(p, Rectangle)
        ]
        # 1000 Hz event in, 4000 Hz event out
        assert len(rects) == 1
        plt.close(fig)

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame({"100.0Hz": [1, 2, 3]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            EventSpectrogramPlotter(df, {})

    def test_no_frequency_columns_raises(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="1h")
        df = pd.DataFrame({"foo": [1, 2, 3]}, index=idx)
        with pytest.raises(ValueError, match="frequency columns"):
            EventSpectrogramPlotter(df, {})


@pytest.fixture
def diagnostics_and_events():
    """Synthetic 1h of per-second diagnostics on 2 bands; one event."""
    n = 3600
    idx = pd.date_range("2024-01-01", periods=n, freq="1s")
    cols = ["1000.0Hz", "5000.0Hz"]
    rng = np.random.default_rng(1)
    values = pd.DataFrame(
        80 + rng.normal(0, 1, (n, 2)), index=idx, columns=cols,
    )
    values.iloc[600:660, 0] += 10  # Spike on 1000 Hz
    threshold = pd.DataFrame(
        np.full((n, 2), 85.0), index=idx, columns=cols,
    )
    baseline = pd.DataFrame(
        np.full((n, 2), 79.0), index=idx, columns=cols,
    )
    diag = BandThresholdDiagnostics(
        values=values, baseline=baseline, threshold=threshold,
    )
    events = pd.DataFrame({
        "detector":   ["band_threshold"],
        "event_id":   [1],
        "start_time": [idx[600]],
        "end_time":   [idx[659]],
        "band_hz":    [1000.0],
    })
    return diag, events


class TestBandThresholdDiagnosticPlotter:
    def test_per_band_returns_figure(self, diagnostics_and_events):
        diag, events = diagnostics_and_events
        plotter = BandThresholdDiagnosticPlotter(diag, events)
        fig = plotter.per_band(bands_hz=[1000, 5000])
        assert len([a for a in fig.axes if a.has_data()]) == 2
        plt.close(fig)

    def test_panel_has_value_threshold_baseline_lines(
        self, diagnostics_and_events,
    ):
        diag, events = diagnostics_and_events
        plotter = BandThresholdDiagnosticPlotter(diag, events)
        fig = plotter.per_band(bands_hz=[1000])
        ax = fig.axes[0]
        assert len(ax.lines) == 3
        plt.close(fig)

    def test_event_span_only_on_matching_band(
        self, diagnostics_and_events,
    ):
        diag, events = diagnostics_and_events
        plotter = BandThresholdDiagnosticPlotter(diag, events)
        fig = plotter.per_band(bands_hz=[1000, 5000])
        # axvspan adds a Polygon to ax.patches
        n_spans_1k = len(fig.axes[0].patches)
        n_spans_5k = len(fig.axes[1].patches)
        assert n_spans_1k > n_spans_5k
        plt.close(fig)

    def test_missing_diagnostics_attr_raises(self):
        class Stub:
            pass
        with pytest.raises(ValueError, match="missing attribute"):
            BandThresholdDiagnosticPlotter(Stub(), pd.DataFrame())

    def test_empty_bands_raises(self, diagnostics_and_events):
        diag, events = diagnostics_and_events
        plotter = BandThresholdDiagnosticPlotter(diag, events)
        with pytest.raises(ValueError, match="at least one"):
            plotter.per_band(bands_hz=[])

    def test_long_series_downsamples(self):
        n = 100_000
        idx = pd.date_range("2024-01-01", periods=n, freq="1s")
        cols = ["1000.0Hz"]
        diag = BandThresholdDiagnostics(
            values=pd.DataFrame(
                np.full((n, 1), 80.0), index=idx, columns=cols,
            ),
            threshold=pd.DataFrame(
                np.full((n, 1), 85.0), index=idx, columns=cols,
            ),
            baseline=pd.DataFrame(
                np.full((n, 1), 79.0), index=idx, columns=cols,
            ),
        )
        plotter = BandThresholdDiagnosticPlotter(diag, pd.DataFrame())
        fig = plotter.per_band(bands_hz=[1000], max_points_per_panel=2000)
        line = fig.axes[0].lines[0]
        assert len(line.get_xdata()) <= 2500 #type: ignore
        plt.close(fig)

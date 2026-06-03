"""Tests for seasound.plotting.spectral_percentiles.SpectralPercentilesPlotter."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from seasound.plotting.spectral_percentiles import SpectralPercentilesPlotter


@pytest.fixture
def full_df():
    """Single-row schema (window='full')."""
    bands = [100.0, 500.0, 1000.0, 5000.0, 10000.0]
    pcts = [5, 25, 50, 75, 95]
    data = {}
    for f in bands:
        for p in pcts:
            # Pick values that monotonically increase with percentile
            data[f"{f}Hz_p{p}"] = [60 + p * 0.3]
    return pd.DataFrame(data)


@pytest.fixture
def windowed_df():
    """24 windows (e.g. 1h windows over a day) × the same band/pct grid."""
    n_windows = 24
    bands = [100.0, 1000.0, 10000.0]
    pcts = [5, 50, 95]
    starts = pd.date_range("2024-01-01", periods=n_windows, freq="1h")
    ends = starts + pd.Timedelta("1h")
    data = {"window_start": starts, "window_end": ends}
    rng = np.random.default_rng(0)
    for f in bands:
        for p in pcts:
            data[f"{f}Hz_p{p}"] = 60 + p * 0.3 + rng.normal(0, 1, n_windows) # type: ignore
    return pd.DataFrame(data)


@pytest.fixture
def long_windowed_df():
    """720 windows — larger than max_panels=16."""
    n_windows = 720
    bands = [100.0, 1000.0, 10000.0]
    pcts = [5, 50, 95]
    starts = pd.date_range("2024-01-01", periods=n_windows, freq="1h")
    ends = starts + pd.Timedelta("1h")
    data = {"window_start": starts, "window_end": ends}
    for f in bands:
        for p in pcts:
            data[f"{f}Hz_p{p}"] = np.full(n_windows, 60 + p * 0.3) # type: ignore
    return pd.DataFrame(data)


class TestSpectralPercentilesPlotterInit:
    def test_full_schema_detected(self, full_df): # pylint: disable=redefined-outer-name
        plotter = SpectralPercentilesPlotter(full_df)
        assert plotter.windowed is False

    def test_windowed_schema_detected(self, windowed_df): # pylint: disable=redefined-outer-name
        plotter = SpectralPercentilesPlotter(windowed_df)
        assert plotter.windowed is True

    def test_no_percentile_columns_raises(self):
        df = pd.DataFrame({"window_start": [], "window_end": []})
        with pytest.raises(ValueError, match="no percentile columns"):
            SpectralPercentilesPlotter(df)

    def test_percentile_map_sorted_by_frequency(self, full_df): # pylint: disable=redefined-outer-name
        plotter = SpectralPercentilesPlotter(full_df)
        for _, pairs in plotter._percentile_map.items(): # pylint: disable=protected-access
            freqs = [f for f, _ in pairs]
            assert freqs == sorted(freqs)


class TestCurvesFullMode:
    def test_basic_curves(self, full_df): # pylint: disable=redefined-outer-name
        plotter = SpectralPercentilesPlotter(full_df)
        fig = plotter.curves()
        ax = fig.axes[0]
        # 5 percentiles = 5 lines
        assert len(ax.lines) == 5
        plt.close(fig)

    def test_shaded_band_adds_collection(self, full_df): # pylint: disable=redefined-outer-name
        plotter = SpectralPercentilesPlotter(full_df)
        fig = plotter.curves(shaded_band=True, shaded_percentiles=(5, 95))
        ax = fig.axes[0]
        # fill_between adds a PolyCollection
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_log_freq_axis(self, full_df): # pylint: disable=redefined-outer-name 
        plotter = SpectralPercentilesPlotter(full_df)
        fig = plotter.curves(log_freq=True)
        assert fig.axes[0].get_xscale() == "log"
        plt.close(fig)

    def test_freq_range_filters(self, full_df): # pylint: disable=redefined-outer-name
        plotter = SpectralPercentilesPlotter(full_df)
        fig = plotter.curves(freq_range=(400, 6000))
        # Each line should only have data for bands in (400, 6000) = 500, 1000, 5000 = 3 points
        ax = fig.axes[0]
        for line in ax.lines:
            assert len(line.get_xdata()) == 3 # type: ignore
        plt.close(fig)

    def test_shaded_percentile_missing_warns_and_skips(self, full_df, caplog): # pylint: disable=redefined-outer-name
        plotter = SpectralPercentilesPlotter(full_df)
        with caplog.at_level("WARNING"):
            fig = plotter.curves(shaded_band=True, shaded_percentiles=(1, 99))
        assert any("shaded_percentiles" in r.message for r in caplog.records)
        plt.close(fig)


class TestCurvesWindowedMode:
    def test_grid_within_max_panels(self, windowed_df): # pylint: disable=redefined-outer-name
        plotter = SpectralPercentilesPlotter(windowed_df)
        fig = plotter.curves(max_panels=16)
        # 24 windows > 16 panels: should subsample to 16
        # 16 panels in a 4x4 grid
        ax_axes = [a for a in fig.axes if a.has_data()]
        assert len(ax_axes) == 16
        plt.close(fig)

    def test_grid_fewer_than_max(self, windowed_df): # pylint: disable=redefined-outer-name
        plotter = SpectralPercentilesPlotter(windowed_df)
        fig = plotter.curves(max_panels=30)
        ax_axes = [a for a in fig.axes if a.has_data()]
        assert len(ax_axes) == 24
        plt.close(fig)

    def test_long_deployment_subsamples_to_max(self, long_windowed_df): # pylint: disable=redefined-outer-name
        plotter = SpectralPercentilesPlotter(long_windowed_df)
        fig = plotter.curves(max_panels=16)
        ax_axes = [a for a in fig.axes if a.has_data()]
        assert len(ax_axes) == 16
        # Suptitle should mention subsampling
        assert "of 720 windows" in fig._suptitle.get_text() # type: ignore # pylint: disable=protected-access
        plt.close(fig)

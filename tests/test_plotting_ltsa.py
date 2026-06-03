"""Tests for seasound.plotting.ltsa.LTSAPlotter."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from seasound.plotting.ltsa import LTSAPlotter


@pytest.fixture
def ltsa_df():
    """24 hourly rows × 5 TOB bands. Uniform 80 dB."""
    idx = pd.date_range("2024-01-01", periods=24, freq="1h")
    cols = ["100.0Hz", "500.0Hz", "1000.0Hz", "5000.0Hz", "10000.0Hz"]
    data = np.full((24, 5), 80.0)
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.fixture
def ltsa_df_with_gap():
    """24 hourly rows with a 5-row gap."""
    idx = pd.DatetimeIndex(
        list(pd.date_range("2024-01-01", periods=10, freq="1h"))
        + list(pd.date_range("2024-01-01 15:00", periods=14, freq="1h"))
    )
    cols = ["100.0Hz", "1000.0Hz", "10000.0Hz"]
    data = np.full((24, 3), 80.0)
    return pd.DataFrame(data, index=idx, columns=cols)


class TestLTSAPlotterInit:
    def test_valid_dataframe_constructs(self, ltsa_df): # pylint: disable=redefined-outer-name
        plotter = LTSAPlotter(ltsa_df)
        assert plotter.df is ltsa_df

    def test_non_datetime_index_raises(self):
        df = pd.DataFrame({"100.0Hz": [1, 2, 3]})
        with pytest.raises(ValueError, match="DatetimeIndex"):
            LTSAPlotter(df)

    def test_no_frequency_columns_raises(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="1h")
        df = pd.DataFrame({"foo": [1, 2, 3]}, index=idx)
        with pytest.raises(ValueError, match="frequency columns"):
            LTSAPlotter(df)


class TestLTSAHeatmap:
    def test_basic_heatmap_returns_figure(self, ltsa_df): # pylint: disable=redefined-outer-name
        plotter = LTSAPlotter(ltsa_df)
        fig = plotter.heatmap()
        assert fig is not None
        assert len(fig.axes) >= 1  # at least heatmap; colorbar adds another
        plt.close(fig)

    def test_heatmap_with_broadband_strip(self, ltsa_df): # pylint: disable=redefined-outer-name
        plotter = LTSAPlotter(ltsa_df)
        fig = plotter.heatmap(broadband_strip=True)
        # 2 panels + 1 colorbar = 3 axes
        assert len(fig.axes) >= 2
        plt.close(fig)

    def test_heatmap_freq_range(self, ltsa_df): # pylint: disable=redefined-outer-name
        plotter = LTSAPlotter(ltsa_df)
        fig = plotter.heatmap(freq_range=(400, 6000))
        plt.close(fig)

    def test_heatmap_db_range_applied(self, ltsa_df): # pylint: disable=redefined-outer-name
        plotter = LTSAPlotter(ltsa_df)
        fig = plotter.heatmap(db_range=(70, 90))
        # imshow image vmin/vmax should reflect the requested range
        im = fig.axes[0].images[0]
        assert im.get_clim() == (70.0, 90.0)
        plt.close(fig)

    def test_heatmap_with_gaps(self, ltsa_df_with_gap): # pylint: disable=redefined-outer-name
        plotter = LTSAPlotter(ltsa_df_with_gap)
        fig = plotter.heatmap(preserve_time_gaps=True)
        plt.close(fig)

    def test_freq_range_empty_raises(self, ltsa_df): # pylint: disable=redefined-outer-name
        plotter = LTSAPlotter(ltsa_df)
        with pytest.raises(ValueError, match="no frequency bands"):
            plotter.heatmap(freq_range=(1e9, 1e10))

    def test_title_applied(self, ltsa_df): # pylint: disable=redefined-outer-name
        plotter = LTSAPlotter(ltsa_df)
        fig = plotter.heatmap(title="My Deployment")
        # Title is on the top axis
        titles = [a.get_title() for a in fig.axes]
        assert "My Deployment" in titles
        plt.close(fig)


class TestLTSABandTimeseries:
    def test_basic_band_timeseries(self, ltsa_df): # pylint: disable=redefined-outer-name
        plotter = LTSAPlotter(ltsa_df)
        fig = plotter.band_timeseries(bands_hz=[125, 1000, 8000])
        ax = fig.axes[0]
        assert len(ax.lines) == 3
        plt.close(fig)

    def test_duplicate_bands_collapse(self, ltsa_df): # pylint: disable=redefined-outer-name
        """Requesting 950 Hz and 1050 Hz should both snap to 1000 Hz once."""
        plotter = LTSAPlotter(ltsa_df)
        fig = plotter.band_timeseries(bands_hz=[950, 1050])
        ax = fig.axes[0]
        assert len(ax.lines) == 1
        plt.close(fig)

    def test_empty_bands_raises(self, ltsa_df): # pylint: disable=redefined-outer-name
        plotter = LTSAPlotter(ltsa_df)
        with pytest.raises(ValueError, match="at least one"):
            plotter.band_timeseries(bands_hz=[])

    def test_db_range_applied(self, ltsa_df): # pylint: disable=redefined-outer-name
        plotter = LTSAPlotter(ltsa_df)
        fig = plotter.band_timeseries(bands_hz=[1000], db_range=(60, 100))
        assert fig.axes[0].get_ylim() == (60.0, 100.0)
        plt.close(fig)

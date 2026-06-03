"""Tests for seasound.plotting._common helpers."""

import numpy as np
import pandas as pd
import pytest

from seasound.plotting._common import (
    compute_grid_dims,
    filter_frequency_range,
    frequency_values,
    get_frequency_columns,
    hz_to_nearest_band,
    reindex_with_gaps,
    subsample_evenly,
)


class TestFrequencyHelpers:
    def test_get_frequency_columns_filters_correctly(self):
        df = pd.DataFrame(
            columns=["window_start", "100.0Hz", "1000.0Hz", "metadata"]
        )
        assert get_frequency_columns(df) == ["100.0Hz", "1000.0Hz"]

    def test_frequency_values_parses_floats(self):
        assert frequency_values(["100.0Hz", "1000.5Hz"]) == [100.0, 1000.5]

    def test_hz_to_nearest_band_picks_closest(self):
        cols = ["100.0Hz", "1000.0Hz", "10000.0Hz"]
        assert hz_to_nearest_band(900, cols) == "1000.0Hz"
        assert hz_to_nearest_band(50, cols) == "100.0Hz"
        assert hz_to_nearest_band(20000, cols) == "10000.0Hz"

    def test_hz_to_nearest_band_empty_raises(self):
        with pytest.raises(ValueError):
            hz_to_nearest_band(1000, [])


class TestFilterFrequencyRange:
    def test_none_returns_unchanged(self):
        df = pd.DataFrame({"100.0Hz": [1], "1000.0Hz": [2]})
        out = filter_frequency_range(df, None)
        assert list(out.columns) == ["100.0Hz", "1000.0Hz"]

    def test_filters_inclusive(self):
        df = pd.DataFrame(
            {"100.0Hz": [1], "1000.0Hz": [2], "10000.0Hz": [3]}
        )
        out = filter_frequency_range(df, (500, 5000))
        assert list(out.columns) == ["1000.0Hz"]

    def test_preserves_non_frequency_columns(self):
        df = pd.DataFrame(
            {"window_start": ["t"], "100.0Hz": [1], "1000.0Hz": [2]}
        )
        out = filter_frequency_range(df, (500, 5000))
        assert list(out.columns) == ["window_start", "1000.0Hz"]


class TestReindexWithGaps:
    def test_no_gaps_returns_equivalent(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="1h")
        df = pd.DataFrame({"100.0Hz": range(5)}, index=idx)
        out = reindex_with_gaps(df)
        assert len(out) == 5
        assert not out.isna().any().any()

    def test_gap_filled_with_nan(self):
        idx = pd.DatetimeIndex(
            [
                "2024-01-01 00:00",
                "2024-01-01 01:00",
                "2024-01-01 04:00",
                "2024-01-01 05:00",
            ]
        )
        df = pd.DataFrame({"100.0Hz": [1, 2, 3, 4]}, index=idx)
        out = reindex_with_gaps(df)
        assert len(out) == 6
        assert out["100.0Hz"].isna().sum() == 2

    def test_non_datetime_raises(self):
        df = pd.DataFrame({"100.0Hz": [1, 2]})
        with pytest.raises(TypeError):
            reindex_with_gaps(df)


class TestGridDims:
    @pytest.mark.parametrize(
        "n,expected",
        [
            (1, (1, 1)),
            (2, (1, 2)),
            (3, (2, 2)),
            (4, (2, 2)),
            (9, (3, 3)),
            (12, (3, 4)),
            (16, (4, 4)),
            (24, (5, 5)),
        ],
    )
    def test_grid_dims(self, n, expected):
        assert compute_grid_dims(n) == expected

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            compute_grid_dims(0)


class TestSubsampleEvenly:
    def test_k_ge_n_returns_all(self):
        assert subsample_evenly([1, 2, 3], 5) == [1, 2, 3]

    def test_k_one_returns_first(self):
        assert subsample_evenly([1, 2, 3, 4], 1) == [1]

    def test_includes_first_and_last(self):
        out = subsample_evenly(list(range(100)), 5)
        assert out[0] == 0
        assert out[-1] == 99
        assert len(out) == 5

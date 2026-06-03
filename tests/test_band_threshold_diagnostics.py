"""Tests for BandThresholdDetector.detect_with_diagnostics."""

import numpy as np
import pandas as pd
import pytest

from seasound.analysis.event_detection import (
    BandThresholdDetector,
    BandThresholdDiagnostics,
)


@pytest.fixture
def small_base_matrix():
    """1 hour of 1-Hz data, 3 bands, one obvious spike on 1000 Hz."""
    n = 3600
    idx = pd.date_range("2024-01-01", periods=n, freq="1s")
    cols = ["100.0Hz", "1000.0Hz", "10000.0Hz"]
    rng = np.random.default_rng(0)
    data = 80 + rng.normal(0, 1, (n, 3))
    data[1800:1860, 1] += 15
    return pd.DataFrame(data, index=idx, columns=cols)


def test_detect_unchanged_returns_only_events(small_base_matrix):
    """detect() still returns just a DataFrame (back-compat)."""
    det = BandThresholdDetector()
    result = det.detect(
        small_base_matrix, {"baseline_window_hours": 0.25}, {},
    )
    assert isinstance(result, pd.DataFrame)


def test_detect_with_diagnostics_returns_tuple(small_base_matrix):
    """detect_with_diagnostics returns (events, BandThresholdDiagnostics)."""
    det = BandThresholdDetector()
    events, diag = det.detect_with_diagnostics(
        small_base_matrix, {"baseline_window_hours": 0.25}, {},
    )
    assert isinstance(events, pd.DataFrame)
    assert isinstance(diag, BandThresholdDiagnostics)


def test_diagnostics_shape_matches_input(small_base_matrix):
    det = BandThresholdDetector()
    _, diag = det.detect_with_diagnostics(
        small_base_matrix, {"baseline_window_hours": 0.25}, {},
    )
    assert isinstance(diag.values.index, pd.DatetimeIndex)
    assert diag.values.shape == diag.baseline.shape == diag.threshold.shape
    assert list(diag.values.columns) == list(diag.baseline.columns)


def test_diagnostics_none_when_no_usable_bands():
    """Empty/tiny input → diagnostics is None."""
    det = BandThresholdDetector()
    tiny = pd.DataFrame(
        {"100.0Hz": [80.0, 80.0]},
        index=pd.date_range("2024-01-01", periods=2, freq="1s"),
    )
    _, diag = det.detect_with_diagnostics(
        tiny, {"baseline_window_hours": 1}, {},
    )
    # 2 rows can't satisfy a 1-hour window with the default coverage filter
    assert diag is None


def test_diagnostics_columns_are_usable_bands_only(small_base_matrix):
    """The diagnostic columns match the bands that survived coverage filtering."""
    det = BandThresholdDetector()
    _, diag = det.detect_with_diagnostics(
        small_base_matrix, {"baseline_window_hours": 0.25}, {},
    )
    # All three bands should be usable in this synthetic fixture
    assert set(diag.values.columns) == {
        "100.0Hz", "1000.0Hz", "10000.0Hz",
    }

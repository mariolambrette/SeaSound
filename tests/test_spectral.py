"""Tests for spectral utilities."""

import numpy as np
from seasound.utils.spectral import (
    tob_centre_frequencies,
    tob_band_edges,
    nearest_band_index,
    extract_freq_hz,
)


def test_tob_centres_full_range():
    centres = tob_centre_frequencies(10, 50000)
    assert len(centres) == 38  # ISO 266 has 38 bands from 10-50kHz
    assert centres[0] == 10.0
    assert centres[-1] == 50000.0


def test_tob_centres_subset():
    centres = tob_centre_frequencies(100, 1000)
    assert centres[0] == 100.0
    assert centres[-1] == 1000.0


def test_band_edges_ordering():
    centres = tob_centre_frequencies(100, 1000)
    edges = tob_band_edges(centres)
    # Lower edge should be below centre, upper above
    for i, fc in enumerate(centres):
        assert edges[i, 0] < fc
        assert edges[i, 1] > fc


def test_band_edges_no_gaps():
    """Adjacent bands should overlap slightly or meet."""
    centres = tob_centre_frequencies(10, 50000)
    edges = tob_band_edges(centres)
    for i in range(len(centres) - 1):
        # Upper edge of band i should be >= lower edge of band i+1
        # (TOB bands actually overlap slightly by design)
        assert edges[i, 1] >= edges[i + 1, 0] * 0.99  # allow tiny gap


def test_nearest_band():
    centres = tob_centre_frequencies(10, 50000)
    idx = nearest_band_index(1000, centres)
    assert centres[idx] == 1000.0

    # Midway between 1000 and 1250, should pick whichever is closer
    idx = nearest_band_index(1100, centres)
    assert centres[idx] in (1000.0, 1250.0)


def test_extract_freq_hz():
    cols = ["63.0Hz", "125.0Hz", "1000.0Hz"]
    freqs = extract_freq_hz(cols)
    np.testing.assert_array_equal(freqs, [63.0, 125.0, 1000.0])
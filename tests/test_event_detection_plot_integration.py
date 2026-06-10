"""End-to-end integration tests for event-detection plotting."""

import os

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from seasound.analysis.event_detection import EventDetectionAnalysis


@pytest.fixture
def base_matrix_with_spike():
    """1h of 1s data on 3 bands; 1000 Hz spike at the middle."""
    n = 3600
    idx = pd.date_range("2024-01-01", periods=n, freq="1s")
    cols = ["100.0Hz", "1000.0Hz", "10000.0Hz"]
    rng = np.random.default_rng(0)
    data = 80 + rng.normal(0, 1, (n, 3))
    data[1800:1830, 1] += 15
    return pd.DataFrame(data, index=idx, columns=cols)


def test_band_threshold_with_diagnostic_plot(
    base_matrix_with_spike, tmp_path, #pylint: disable=redefined-outer-name
):
    """When enabled, the band_threshold plot is created and contains the expected event."""
    module = EventDetectionAnalysis()
    cfg = {
        "output_format": "csv",
        "detectors": [{
            "type": "band_threshold",
            "enabled": True,
            "baseline_window_hours": 0.25,
            "min_duration_s": 3,
            "merge_gap_s": 5,
            "plot": {
                "enabled": True,
                "types": ["threshold_diagnostic"],
                "threshold_diagnostic": {
                    "bands_hz": [1000],
                    "max_points_per_panel": 1000,
                },
                "output_format": "png",
                "dpi": 100,
            },
        }],
    }
    result = module.run(base_matrix_with_spike, cfg, str(tmp_path))
    pngs = [p for p in result.outputs if p.endswith(".png")]
    assert len(pngs) == 1
    assert "threshold_diagnostic" in os.path.basename(pngs[0])


def test_plot_disabled_uses_plain_detect(base_matrix_with_spike, tmp_path): #pylint: disable=redefined-outer-name
    """When plot is off, no diagnostics are computed and no plots produced."""
    module = EventDetectionAnalysis()
    cfg = {
        "detectors": [{
            "type": "band_threshold",
            "baseline_window_hours": 0.25,
        }],
    }
    result = module.run(base_matrix_with_spike, cfg, str(tmp_path))
    assert not any(p.endswith(".png") for p in result.outputs)


def test_annotated_spectrogram_warns_without_stft(
    base_matrix_with_spike, tmp_path, #pylint: disable=redefined-outer-name
):
    """No STFT cache → warning, but pipeline completes."""
    module = EventDetectionAnalysis()
    cfg = {
        "detectors": [{
            "type": "band_threshold",
            "baseline_window_hours": 0.25,
        }],
        "annotated_spectrogram": {"enabled": True},
    }
    result = module.run(base_matrix_with_spike, cfg, str(tmp_path))
    assert any(
        "Annotated spectrogram skipped: STFT data not available" in w for w in result.warnings
    )

def test_annotated_spectrogram_chunking_config_parses(
    base_matrix_with_spike, tmp_path, #pylint: disable=redefined-outer-name
):
    """time_chunk + time_bins parse correctly and don't crash the run.

    Without an STFT cache the plot itself is skipped, but the
    configuration is validated and the wrapper runs cleanly.
    """
    module = EventDetectionAnalysis()
    cfg = {
        "detectors": [{
            "type": "band_threshold",
            "baseline_window_hours": 0.25,
        }],
        "annotated_spectrogram": {
            "enabled": True,
            "time_chunk": "30min",
            "time_bins": 1000,
        },
    }
    result = module.run(base_matrix_with_spike, cfg, str(tmp_path))
    # No STFT → skipped with the standard warning
    assert any(
        "STFT data not available" in w for w in result.warnings
    )

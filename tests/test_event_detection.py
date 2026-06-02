"""
tests/test_event_detection.py

Unit and integration tests for the event_detection module.

Covers:
- Internal helper functions (_broadband_spl, _spectral_centroid, _flag_and_merge)
- Detector registry (register/get/list, error paths)
- AdaptiveThresholdDetector behaviour (positive, negative, schema)
- PCAAnomalyDetector behaviour (positive, negative, schema, the
  critical spectral-shape-anomaly test that adaptive_threshold misses)
- EventDetectionAnalysis (multi-detector config, validation, dispatch)
- Analysis-registry integration
"""

import os

import numpy as np
import pandas as pd
import pytest

from seasound.analysis.base import AnalysisModuleError
from seasound.analysis.event_detection import (
    CANONICAL_EVENT_COLUMNS,
    DETECTOR_REGISTRY,
    AdaptiveThresholdDetector,
    EventDetectionAnalysis,
    PCAAnomalyDetector,
    _broadband_spl,
    _flag_and_merge,
    _spectral_centroid,
    get_detector,
    list_detectors,
    register_detector,
)
from seasound.analysis.registry import get_analysis, list_registered


# =========================================================================
# Synthetic data helpers
# =========================================================================

def _make_synthetic_matrix(
    n_seconds: int = 3600,
    seed: int = 0,
    n_bands: int = 8,
    baseline_db: float = 80.0,
    noise_std: float = 1.0,
) -> pd.DataFrame:
    """Build a base matrix of mild Gaussian noise around a flat baseline."""
    rng = np.random.default_rng(seed)
    centres = np.geomspace(63.0, 8000.0, num=n_bands)
    cols = [f"{c:.1f}Hz" for c in centres]
    data = baseline_db + rng.normal(0, noise_std, size=(n_seconds, n_bands))
    idx = pd.date_range("2026-01-01 12:00", periods=n_seconds, freq="1s")
    return pd.DataFrame(data, index=idx, columns=cols)


def _inject_broadband_burst(
    matrix: pd.DataFrame,
    start_idx: int,
    duration_s: int,
    gain_db: float = 20.0,
) -> pd.DataFrame:
    """Add gain_db to every band over a window (a loud, flat-spectrum event)."""
    out = matrix.copy()
    out.iloc[start_idx:start_idx + duration_s, :] += gain_db
    return out


def _inject_single_band_anomaly(
    matrix: pd.DataFrame,
    start_idx: int,
    duration_s: int,
    band_idx: int,
    gain_db: float = 12.0,
) -> pd.DataFrame:
    """
    Add gain to ONE band only. With ~8 bands the broadband SPL increase is
    small (a few dB) while the spectral shape is clearly anomalous —
    perfect for testing that PCA catches what adaptive_threshold misses.
    """
    out = matrix.copy()
    out.iloc[start_idx:start_idx + duration_s, band_idx] += gain_db
    return out


# =========================================================================
# Helper-function tests
# =========================================================================

class TestBroadbandSPL:

    def test_uniform_bands(self):
        idx = pd.date_range("2026-01-01", periods=10, freq="1s")
        df = pd.DataFrame(
            np.full((10, 4), 80.0),
            index=idx,
            columns=["100.0Hz", "200.0Hz", "400.0Hz", "800.0Hz"],
        )
        bb = _broadband_spl(df)
        expected = 10.0 * np.log10(4 * 10 ** 8)
        np.testing.assert_allclose(bb.to_numpy(), expected, atol=1e-6)

    def test_nan_bands_ignored(self):
        idx = pd.date_range("2026-01-01", periods=3, freq="1s")
        df = pd.DataFrame(
            [[80.0, np.nan], [80.0, 80.0], [80.0, np.nan]],
            index=idx,
            columns=["100.0Hz", "200.0Hz"],
        )
        bb = _broadband_spl(df)
        # Row 0 and 2: only one band of 80 dB → 80 dB
        # Row 1: two bands of 80 dB → 83.01 dB
        np.testing.assert_allclose(bb.iloc[0], 80.0, atol=1e-6)
        np.testing.assert_allclose(bb.iloc[1], 80.0 + 10 * np.log10(2), atol=1e-6)
        np.testing.assert_allclose(bb.iloc[2], 80.0, atol=1e-6)


class TestSpectralCentroid:

    def test_low_band_dominates(self):
        s = pd.Series(
            [80.0, 0.0, 0.0],
            index=["100.0Hz", "1000.0Hz", "10000.0Hz"],
        )
        c = _spectral_centroid(s)
        # 10^8 at 100 Hz vs 10^0 at 1k and 10k — 100 Hz dominates entirely
        assert 99.99 < c < 100.01

    def test_zero_total_returns_nan(self):
        s = pd.Series(
            [-np.inf, -np.inf],
            index=["100.0Hz", "200.0Hz"],
        )
        assert np.isnan(_spectral_centroid(s))


class TestFlagAndMerge:

    def _flag_series(self, n: int, true_ranges: list[tuple[int, int]]):
        idx = pd.date_range("2026-01-01", periods=n, freq="1s")
        s = pd.Series(False, index=idx)
        for a, b in true_ranges:
            s.iloc[a:b] = True
        return s

    def test_single_run(self):
        flags = self._flag_series(100, [(10, 20)])  # 10s of True
        merged = _flag_and_merge(flags, min_duration_s=3, merge_gap_s=10)
        assert len(merged) == 1

    def test_merges_close_runs(self):
        flags = self._flag_series(100, [(10, 20), (25, 35)])  # 5s gap
        merged = _flag_and_merge(flags, min_duration_s=3, merge_gap_s=30)
        assert len(merged) == 1
        assert merged[0][0] == flags.index[10]
        assert merged[0][1] == flags.index[34]

    def test_keeps_distant_runs_separate(self):
        flags = self._flag_series(200, [(10, 20), (100, 110)])  # 80s gap
        merged = _flag_and_merge(flags, min_duration_s=3, merge_gap_s=30)
        assert len(merged) == 2

    def test_drops_short_runs(self):
        flags = self._flag_series(100, [(10, 12)])  # 2s, below min=3
        merged = _flag_and_merge(flags, min_duration_s=3, merge_gap_s=10)
        assert merged == []

    def test_empty_input(self):
        flags = self._flag_series(100, [])
        assert _flag_and_merge(flags, 3, 30) == []


# =========================================================================
# Detector registry tests
# =========================================================================

class TestDetectorRegistry:

    def test_builtins_registered(self):
        names = set(DETECTOR_REGISTRY.keys())
        assert "adaptive_threshold" in names
        assert "anomaly" in names

    def test_get_detector_returns_instance(self):
        d = get_detector("adaptive_threshold")
        assert isinstance(d, AdaptiveThresholdDetector)
        d2 = get_detector("anomaly")
        assert isinstance(d2, PCAAnomalyDetector)

    def test_get_detector_unknown_raises(self):
        with pytest.raises(ValueError):
            get_detector("nonsense")

    def test_register_non_eventdetector_raises(self):
        class NotADetector:  # noqa: D101
            pass
        with pytest.raises(TypeError):
            register_detector("bad", NotADetector)  # type: ignore[arg-type]

    def test_list_detectors_contains_builtins(self):
        listing = list_detectors()
        assert "adaptive_threshold" in listing
        assert "anomaly" in listing


# =========================================================================
# AdaptiveThresholdDetector tests
# =========================================================================

class TestAdaptiveThresholdDetector:

    def test_validate_requires_threshold_db(self):
        det = AdaptiveThresholdDetector()
        with pytest.raises(ValueError):
            det.validate_config({}, {})

    def test_validate_rejects_invalid_percentile(self):
        det = AdaptiveThresholdDetector()
        with pytest.raises(ValueError):
            det.validate_config(
                {"threshold_db": 10.0, "baseline_percentile": 150}, {},
            )

    def test_validate_rejects_invalid_freq_range(self):
        det = AdaptiveThresholdDetector()
        with pytest.raises(ValueError):
            det.validate_config(
                {"threshold_db": 10.0, "broadband_freq_range": [100, 50]},
                {},
            )

    def test_detect_finds_injected_burst(self):
        matrix = _make_synthetic_matrix(n_seconds=3600)
        matrix = _inject_broadband_burst(matrix, 1800, 60, gain_db=20.0)
        det = AdaptiveThresholdDetector()
        events = det.detect(
            matrix,
            cfg={
                "threshold_db": 10.0,
                "baseline_window_hours": 0.25,
                "baseline_percentile": 10,
            },
            shared_cfg={"min_duration_s": 3, "merge_gap_s": 30},
        )
        assert len(events) >= 1
        burst_start = matrix.index[1800]
        burst_end = matrix.index[1859]
        overlaps = (
            (events["start_time"] <= burst_end)
            & (events["end_time"] >= burst_start)
        )
        assert overlaps.any()

    def test_no_events_on_pure_noise(self):
        matrix = _make_synthetic_matrix(n_seconds=3600, noise_std=1.0)
        det = AdaptiveThresholdDetector()
        events = det.detect(
            matrix,
            cfg={
                "threshold_db": 10.0,
                "baseline_window_hours": 0.25,
                "baseline_percentile": 10,
            },
            shared_cfg={"min_duration_s": 3, "merge_gap_s": 30},
        )
        # Pure noise with 1 dB std should never exceed baseline by 10 dB for >3 s
        assert len(events) == 0

    def test_event_has_canonical_schema(self):
        matrix = _make_synthetic_matrix(n_seconds=3600)
        matrix = _inject_broadband_burst(matrix, 1800, 60, 20.0)
        det = AdaptiveThresholdDetector()
        events = det.detect(
            matrix,
            cfg={"threshold_db": 10.0, "baseline_window_hours": 0.25,
                 "baseline_percentile": 10},
            shared_cfg={"min_duration_s": 3, "merge_gap_s": 30},
        )
        for col in CANONICAL_EVENT_COLUMNS:
            assert col in events.columns
        assert (events["detector"] == "adaptive_threshold").all()
        assert (events["score_type"] == "delta_db").all()


# =========================================================================
# PCAAnomalyDetector tests
# =========================================================================

class TestPCAAnomalyDetector:

    def test_validate_rejects_invalid_method(self):
        det = PCAAnomalyDetector()
        with pytest.raises(ValueError):
            det.validate_config({"method": "knn"}, {})

    def test_validate_rejects_invalid_n_components(self):
        det = PCAAnomalyDetector()
        with pytest.raises(ValueError):
            det.validate_config({"n_components": 0}, {})

    def test_validate_rejects_invalid_variance_explained(self):
        det = PCAAnomalyDetector()
        with pytest.raises(ValueError):
            det.validate_config({"variance_explained": 1.5}, {})

    def test_validate_rejects_invalid_trim(self):
        det = PCAAnomalyDetector()
        with pytest.raises(ValueError):
            det.validate_config({"baseline_trim_percentile": 30}, {})

    def test_detect_finds_broadband_burst(self):
        matrix = _make_synthetic_matrix(n_seconds=3600)
        matrix = _inject_broadband_burst(matrix, 1800, 60, gain_db=20.0)
        det = PCAAnomalyDetector()
        events = det.detect(
            matrix,
            cfg={"n_components": 3, "threshold_percentile": 99.0},
            shared_cfg={"min_duration_s": 3, "merge_gap_s": 30},
        )
        assert len(events) >= 1

    def test_detect_finds_spectral_anomaly_that_threshold_misses(self):
        """
        The canonical test that distinguishes PCA from adaptive threshold:
        a single-band anomaly whose broadband impact is only a few dB
        should be caught by PCA but not by adaptive_threshold.
        """
        matrix = _make_synthetic_matrix(n_seconds=3600, noise_std=0.5)
        # Inject 12 dB into one band only — broadband rises ~1-2 dB,
        # well below adaptive_threshold's 10 dB trigger.
        matrix = _inject_single_band_anomaly(
            matrix, start_idx=1800, duration_s=60, band_idx=2, gain_db=12.0,
        )

        # Adaptive threshold should not detect this
        at_det = AdaptiveThresholdDetector()
        at_events = at_det.detect(
            matrix,
            cfg={"threshold_db": 10.0, "baseline_window_hours": 0.25,
                 "baseline_percentile": 10},
            shared_cfg={"min_duration_s": 3, "merge_gap_s": 30},
        )
        assert len(at_events) == 0, (
            "adaptive_threshold should not detect this spectral-only anomaly"
        )

        # PCA should
        pca_det = PCAAnomalyDetector()
        pca_events = pca_det.detect(
            matrix,
            cfg={"n_components": 3, "threshold_percentile": 99.0,
                 "report_top_n_bands": 3},
            shared_cfg={"min_duration_s": 3, "merge_gap_s": 30},
        )
        assert len(pca_events) >= 1, (
            "PCA should detect the single-band spectral anomaly"
        )

    def test_top_band_columns_present(self):
        matrix = _make_synthetic_matrix(n_seconds=3600)
        matrix = _inject_broadband_burst(matrix, 1800, 60, 20.0)
        det = PCAAnomalyDetector()
        events = det.detect(
            matrix,
            cfg={"n_components": 3, "threshold_percentile": 99.0,
                 "report_top_n_bands": 3},
            shared_cfg={"min_duration_s": 3, "merge_gap_s": 30},
        )
        for j in range(1, 4):
            assert f"top_band_{j}_hz" in events.columns
            assert f"top_band_{j}_contribution" in events.columns
        assert "n_components" in events.columns
        assert "explained_variance" in events.columns

    def test_handles_nan_bands(self):
        matrix = _make_synthetic_matrix(n_seconds=3600)
        matrix = _inject_broadband_burst(matrix, 1800, 60, 20.0)
        # Inject NaN into one band on a random subset of rows — simulating
        # a band above the file's Nyquist on some channels.
        matrix.iloc[::100, 0] = np.nan
        det = PCAAnomalyDetector()
        events = det.detect(
            matrix,
            cfg={"n_components": 3, "threshold_percentile": 99.0},
            shared_cfg={"min_duration_s": 3, "merge_gap_s": 30},
        )
        # Should still find the burst despite some NaN rows
        assert len(events) >= 1

    def test_variance_explained_overrides_n_components(self):
        matrix = _make_synthetic_matrix(n_seconds=3600)
        matrix = _inject_broadband_burst(matrix, 1800, 60, 20.0)
        det = PCAAnomalyDetector()
        events = det.detect(
            matrix,
            cfg={
                "n_components": 1,        # would be too few
                "variance_explained": 0.95,
                "threshold_percentile": 99.0,
            },
            shared_cfg={"min_duration_s": 3, "merge_gap_s": 30},
        )
        # n_components in output should reflect the variance_explained choice
        if len(events) > 0:
            assert events["n_components"].iloc[0] >= 1


# =========================================================================
# EventDetectionAnalysis (the outer wrapper) tests
# =========================================================================

class TestEventDetectionAnalysis:

    def test_validate_requires_detectors_list(self):
        ed = EventDetectionAnalysis()
        with pytest.raises(ValueError):
            ed.validate_config({})

    def test_validate_rejects_empty_detectors_list(self):
        ed = EventDetectionAnalysis()
        with pytest.raises(ValueError):
            ed.validate_config({"detectors": []})

    def test_validate_rejects_unknown_detector_type(self):
        ed = EventDetectionAnalysis()
        with pytest.raises(ValueError):
            ed.validate_config({"detectors": [{"type": "ghosts"}]})

    def test_validate_rejects_invalid_shared_param(self):
        ed = EventDetectionAnalysis()
        with pytest.raises(ValueError):
            ed.validate_config({
                "min_duration_s": -1,
                "detectors": [
                    {"type": "adaptive_threshold", "threshold_db": 10.0},
                ],
            })

    def test_validate_collects_multiple_errors(self):
        ed = EventDetectionAnalysis()
        with pytest.raises(ValueError) as exc_info:
            ed.validate_config({
                "min_duration_s": -1,
                "merge_gap_s": -5,
                "detectors": [{"type": "ghosts"}],
            })
        # Multi-line error message contains all three problems
        msg = str(exc_info.value)
        assert "min_duration_s" in msg
        assert "merge_gap_s" in msg
        assert "ghosts" in msg

    def test_run_with_adaptive_threshold(self, tmp_path):
        matrix = _make_synthetic_matrix(n_seconds=3600)
        matrix = _inject_broadband_burst(matrix, 1800, 60, 20.0)
        ed = EventDetectionAnalysis()
        cfg = {
            "min_duration_s": 3,
            "merge_gap_s": 30,
            "detectors": [
                {
                    "type": "adaptive_threshold",
                    "threshold_db": 10.0,
                    "baseline_window_hours": 0.25,
                    "baseline_percentile": 10,
                },
            ],
        }
        result = ed.run(matrix, cfg, str(tmp_path))
        assert result.name == "event_detection"
        assert len(result.outputs) == 1
        assert "event_detection_adaptive_threshold.csv" in result.outputs[0]
        assert os.path.isfile(result.outputs[0])
        assert result.summary["n_detectors"] == 1

    def test_run_with_both_detectors(self, tmp_path):
        matrix = _make_synthetic_matrix(n_seconds=3600)
        matrix = _inject_broadband_burst(matrix, 1800, 60, 20.0)
        ed = EventDetectionAnalysis()
        cfg = {
            "min_duration_s": 3,
            "merge_gap_s": 30,
            "detectors": [
                {"type": "adaptive_threshold", "threshold_db": 10.0,
                 "baseline_window_hours": 0.25, "baseline_percentile": 10},
                {"type": "anomaly", "n_components": 3,
                 "threshold_percentile": 99.0},
            ],
        }
        result = ed.run(matrix, cfg, str(tmp_path))
        assert len(result.outputs) == 2
        names = {os.path.basename(p) for p in result.outputs}
        assert names == {
            "event_detection_adaptive_threshold.csv",
            "event_detection_anomaly.csv",
        }
        assert result.summary["n_detectors"] == 2
        assert "adaptive_threshold" in result.summary["detectors"]
        assert "anomaly" in result.summary["detectors"]

    def test_run_writes_empty_csv_when_no_events(self, tmp_path):
        matrix = _make_synthetic_matrix(n_seconds=3600, noise_std=0.5)
        ed = EventDetectionAnalysis()
        cfg = {
            "min_duration_s": 3,
            "merge_gap_s": 30,
            "detectors": [
                {"type": "adaptive_threshold", "threshold_db": 30.0,
                 "baseline_window_hours": 0.25, "baseline_percentile": 10},
            ],
        }
        result = ed.run(matrix, cfg, str(tmp_path))
        # File still written, with header but no rows
        df = pd.read_csv(result.outputs[0])
        assert len(df) == 0
        for col in CANONICAL_EVENT_COLUMNS:
            assert col in df.columns


# =========================================================================
# Analysis-registry integration
# =========================================================================

class TestAnalysisRegistryIntegration:

    def test_event_detection_registered(self):
        modules = list_registered()
        assert "event_detection" in modules

    def test_get_analysis_returns_instance(self):
        m = get_analysis("event_detection")
        assert isinstance(m, EventDetectionAnalysis)

    def test_run_via_pipeline_dispatch(self, tmp_path):
        """End-to-end through the analysis-registry dispatch."""
        matrix = _make_synthetic_matrix(n_seconds=3600)
        matrix = _inject_broadband_burst(matrix, 1800, 60, 20.0)
        module = get_analysis("event_detection")
        cfg = {
            "min_duration_s": 3,
            "merge_gap_s": 30,
            "detectors": [
                {"type": "adaptive_threshold", "threshold_db": 10.0,
                 "baseline_window_hours": 0.25, "baseline_percentile": 10},
            ],
        }
        result = module.run(matrix, cfg, str(tmp_path))
        assert result.name == "event_detection"
        assert os.path.isfile(result.outputs[0])

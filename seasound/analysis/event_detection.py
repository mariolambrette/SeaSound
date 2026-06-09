"""
seasound/analysis/event_detection.py

Event detection analysis module with pluggable detector algorithms.
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type

import numpy as np
import pandas as pd

from seasound.analysis.base import (
    AnalysisModule,
    AnalysisModuleError,
    AnalysisResult,
)
from seasound.analysis.registry import register_analysis

logger = logging.getLogger(__name__)


CANONICAL_EVENT_COLUMNS: list[str] = [
    "detector",
    "event_id",
    "start_time",
    "end_time",
    "duration_s",
    "score",
    "score_type",
]


class EventDetector(ABC):
    """Base class for event-detection algorithms."""

    name: str

    # Detectors run on the base matrix (refactor §7/D5). ``detect``
    # accepts an optional ``stft`` accessor for forward compatibility —
    # a future detector may consume the STFT — but no current detector
    # does. A subclass that needs it adds "stft" here.
    REQUIRES: frozenset = frozenset({"base_matrix"})

    @abstractmethod
    def validate_config(self, cfg: dict, shared_cfg: dict) -> None:
        """Validate the detector-specific configuration."""

    @abstractmethod
    def detect(
        self,
        base_matrix: pd.DataFrame,
        cfg: dict,
        shared_cfg: dict,
        stft=None,
    ) -> pd.DataFrame:
        """Run detection on the base matrix.

        ``stft`` is an optional windowed STFT accessor for detectors
        that declare ``"stft"`` in ``REQUIRES``; it is ``None`` for the
        base-matrix detectors that ship today.
        """


DETECTOR_REGISTRY: dict[str, Type[EventDetector]] = {}


def register_detector(name: str, cls: Type[EventDetector]) -> None:
    """Register a detector class under `name`."""
    if not issubclass(cls, EventDetector):
        raise TypeError(
            f"Detector '{name}' must inherit from EventDetector; got {cls}"
        )
    if name in DETECTOR_REGISTRY:
        logger.warning(
            "Duplicate detector registration: '%s'. "
            "Overwriting %s with %s",
            name,
            DETECTOR_REGISTRY[name].__name__,
            cls.__name__,
        )
    DETECTOR_REGISTRY[name] = cls
    logger.debug("Registered detector: '%s' (%s)", name, cls.__name__)


def get_detector(name: str) -> EventDetector:
    """Instantiate a detector by name. Raises ValueError if unknown."""
    if name not in DETECTOR_REGISTRY:
        available = ", ".join(sorted(DETECTOR_REGISTRY.keys()))
        raise ValueError(
            f"Unknown detector '{name}'. Available: {available}"
        )
    return DETECTOR_REGISTRY[name]()


def list_detectors() -> dict[str, str]:
    """Return mapping of detector name -> class name (for diagnostics)."""
    return {n: c.__name__ for n, c in DETECTOR_REGISTRY.items()}


def _get_frequency_value(column_name: str) -> float:
    """Parse Hz from a column name like '1000.0Hz'."""
    if not column_name.endswith("Hz"):
        raise ValueError(f"Invalid frequency column: {column_name}")
    return float(column_name[:-2])


def _filter_freq_range(
    matrix: pd.DataFrame,
    freq_range: tuple[float, float] | list | None,
) -> pd.DataFrame:
    """Return a view of `matrix` with only columns whose Hz is in range."""
    if freq_range is None:
        return matrix
    fmin, fmax = float(freq_range[0]), float(freq_range[1])
    keep = [
        c for c in matrix.columns
        if isinstance(c, str)
        and c.endswith("Hz")
        and fmin <= _get_frequency_value(c) <= fmax
    ]
    if not keep:
        raise AnalysisModuleError(
            f"No frequency columns in range [{fmin}, {fmax}] Hz"
        )
    return matrix[keep]


def _broadband_spl(matrix: pd.DataFrame) -> pd.Series:
    """Sum band-level SPLs into a single broadband SPL per row."""
    linear = np.power(10.0, matrix.to_numpy(dtype=np.float64) / 10.0)
    linear = np.where(np.isfinite(linear), linear, 0.0)
    total = linear.sum(axis=1)
    total = np.where(total > 0, total, np.nan)
    return pd.Series(10.0 * np.log10(total), index=matrix.index)


def _spectral_centroid(spectrum_db: pd.Series) -> float:
    """Energy-weighted mean frequency of a single spectrum."""
    freqs = np.array(
        [_get_frequency_value(c) for c in spectrum_db.index],
        dtype=np.float64,
    )
    values = spectrum_db.to_numpy(dtype=np.float64)
    linear = np.power(10.0, values / 10.0)
    linear = np.where(np.isfinite(linear), linear, 0.0)
    total = linear.sum()
    if total <= 0 or not np.isfinite(total):
        return float("nan")
    return float((freqs * linear).sum() / total)


def _flag_and_merge(
    flags: pd.Series,
    min_duration_s: int,
    merge_gap_s: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Convert a boolean flag series into a list of (start, end) event tuples."""
    flags = flags.fillna(False).astype(bool)
    if not flags.any():
        return []

    arr = flags.to_numpy()
    idx = flags.index

    diff = np.diff(arr.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    if len(starts) == 0:
        return []

    runs = [(idx[int(s)], idx[int(e)]) for s, e in zip(starts, ends)]

    merged: list[tuple[pd.Timestamp, pd.Timestamp]] = [runs[0]]
    for s, e in runs[1:]:
        gap_s = (s - merged[-1][1]).total_seconds() - 1.0
        if gap_s <= merge_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    final = [
        (s, e) for s, e in merged
        if (e - s).total_seconds() + 1.0 >= min_duration_s
    ]
    return final


def _flag_and_merge_array(
    flags_arr: np.ndarray,
    index: pd.DatetimeIndex,
    min_duration_s: int,
    merge_gap_s: int,
) -> list[tuple[int, int]]:
    """Vectorised flag-and-merge that returns integer index pairs."""
    if not flags_arr.any():
        return []

    diff = np.diff(flags_arr.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1) - 1
    if len(starts) == 0:
        return []

    timestamps = index.to_numpy()
    merged: list[tuple[int, int]] = [(int(starts[0]), int(ends[0]))]
    for s, e in zip(starts[1:], ends[1:]):
        prev_end_ts = timestamps[merged[-1][1]]
        gap_s = (timestamps[s] - prev_end_ts) / np.timedelta64(1, "s") - 1.0
        if gap_s <= merge_gap_s:
            merged[-1] = (merged[-1][0], int(e))
        else:
            merged.append((int(s), int(e)))

    final: list[tuple[int, int]] = []
    for s, e in merged:
        dur = (timestamps[e] - timestamps[s]) / np.timedelta64(1, "s") + 1.0
        if dur >= min_duration_s:
            final.append((s, e))
    return final


def _build_event_row(
    detector_name: str,
    event_id: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    base_matrix: pd.DataFrame,
    score: float,
    score_type: str,
    extras: dict | None = None,
) -> dict:
    """Build a canonical event record dict from one (start, end) window."""
    window = base_matrix.loc[start:end]
    if window.empty:
        return {}

    broadband = _broadband_spl(window)
    if broadband.dropna().empty:
        peak_spl = float("nan")
        leq = float("nan")
        centroid = float("nan")
    else:
        peak_spl = float(broadband.max())
        linear = np.power(10.0, broadband.dropna().to_numpy() / 10.0)
        leq = float(10.0 * np.log10(linear.mean()))
        peak_idx = broadband.idxmax()
        peak_row = window.loc[peak_idx]
        if isinstance(peak_row, pd.DataFrame):
            peak_row = peak_row.iloc[0]
        centroid = _spectral_centroid(peak_row)

    row = {
        "detector": detector_name,
        "event_id": int(event_id),
        "start_time": start,
        "end_time": end,
        "duration_s": float((end - start).total_seconds() + 1.0),
        "peak_spl_dB": peak_spl,
        "leq_dB": leq,
        "spectral_centroid_hz": centroid,
        "score": float(score),
        "score_type": score_type,
    }
    if extras:
        row.update(extras)
    return row


def _resolve_base_resolution_s(detector: EventDetector) -> float:
    """Read base_resolution_s from the runtime context."""
    ctx = (
        getattr(detector, "_get_runtime_context", lambda: {})()
        if hasattr(detector, "_get_runtime_context")
        else {}
    )
    pipeline_cfg = ctx.get("pipeline_config")
    if pipeline_cfg is None:
        return 1.0
    return float(getattr(pipeline_cfg.pipeline, "base_resolution_s", 1))


def _validate_plot_block(
    module_name: str,
    cfg: dict,
    valid_types: set[str],
    errors: list[str],
) -> None:
    """Validate a `plot:` block. Accumulates errors instead of raising."""
    plot_cfg = cfg.get("plot")
    if plot_cfg is None:
        return
    if not isinstance(plot_cfg, dict):
        errors.append(
            f"{module_name}.plot must be a mapping or null; "
            f"got {type(plot_cfg).__name__}"
        )
        return

    enabled = plot_cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        errors.append(f"{module_name}.plot.enabled must be a boolean")

    types = plot_cfg.get("types", [])
    if not isinstance(types, list):
        errors.append(
            f"{module_name}.plot.types must be a list; "
            f"got {type(types).__name__}"
        )
    else:
        for t in types:
            if t not in valid_types:
                errors.append(
                    f"{module_name}.plot.types entries must be in "
                    f"{sorted(valid_types)}; got '{t}'"
                )

    fmt = plot_cfg.get("output_format", "png")
    if fmt not in {"png", "pdf"}:
        errors.append(
            f"{module_name}.plot.output_format must be 'png' or 'pdf'; "
            f"got '{fmt}'"
        )

    dpi = plot_cfg.get("dpi", 300)
    if not isinstance(dpi, int) or dpi <= 0:
        errors.append(
            f"{module_name}.plot.dpi must be a positive integer; got {dpi}"
        )


@dataclass
class BandThresholdDiagnostics:
    """Per-band rolling statistics from a BandThresholdDetector run."""
    values:    pd.DataFrame
    baseline:  pd.DataFrame
    threshold: pd.DataFrame

class BandThresholdDetector(EventDetector):
    """Per-band rolling-percentile event detector."""

    name = "band_threshold"

    DEFAULTS = {
        "freq_range_hz": None,
        "baseline_window_hours": 4.0,
        "baseline_percentile": 10.0,
        "threshold_percentile": 99.0,
        "min_band_coverage": 0.5,
        "min_duration_s": 3,
        "merge_gap_s": 30,
    }

    def validate_config(self, cfg: dict, shared_cfg: dict) -> None:
        errors: list[str] = []
        prefix = "event_detection.detectors[band_threshold]"

        bw_hours = cfg.get(
            "baseline_window_hours", self.DEFAULTS["baseline_window_hours"]
        )
        if not isinstance(bw_hours, (int, float)) or bw_hours <= 0:
            errors.append(
                f"{prefix}.baseline_window_hours must be positive; "
                f"got {bw_hours}"
            )

        bp = cfg.get(
            "baseline_percentile", self.DEFAULTS["baseline_percentile"]
        )
        if not isinstance(bp, (int, float)) or not 0 < bp < 100:
            errors.append(
                f"{prefix}.baseline_percentile must be in (0, 100); got {bp}"
            )

        tp = cfg.get(
            "threshold_percentile", self.DEFAULTS["threshold_percentile"]
        )
        if not isinstance(tp, (int, float)) or not 0 < tp < 100:
            errors.append(
                f"{prefix}.threshold_percentile must be in (0, 100); got {tp}"
            )

        if isinstance(bp, (int, float)) and isinstance(tp, (int, float)):
            if tp <= bp:
                errors.append(
                    f"{prefix}.threshold_percentile ({tp}) must be greater "
                    f"than baseline_percentile ({bp})"
                )

        cov = cfg.get(
            "min_band_coverage", self.DEFAULTS["min_band_coverage"]
        )
        if not isinstance(cov, (int, float)) or not 0 < cov <= 1:
            errors.append(
                f"{prefix}.min_band_coverage must be in (0, 1]; got {cov}"
            )

        min_dur = cfg.get(
            "min_duration_s", self.DEFAULTS["min_duration_s"]
        )
        if not isinstance(min_dur, (int, float)) or min_dur <= 0:
            errors.append(
                f"{prefix}.min_duration_s must be positive; got {min_dur}"
            )

        merge_gap = cfg.get(
            "merge_gap_s", self.DEFAULTS["merge_gap_s"]
        )
        if not isinstance(merge_gap, (int, float)) or merge_gap < 0:
            errors.append(
                f"{prefix}.merge_gap_s must be >= 0; got {merge_gap}"
            )

        freq_range = cfg.get("freq_range_hz")
        if freq_range is not None:
            ok = (
                isinstance(freq_range, (list, tuple))
                and len(freq_range) == 2
                and freq_range[0] < freq_range[1]
            )
            if not ok:
                errors.append(
                    f"{prefix}.freq_range_hz must be [fmin, fmax] with "
                    f"fmin < fmax; got {freq_range}"
                )

        if errors:
            raise ValueError("\n".join(errors))

    def detect(
        self,
        base_matrix: pd.DataFrame,
        cfg: dict,
        shared_cfg: dict,
        stft=None,
    ) -> pd.DataFrame:
        """Run detection. Returns events only."""
        events_df, _ = self._run(
            base_matrix, cfg, shared_cfg, return_diagnostics=False,
        )
        return events_df

    def detect_with_diagnostics(
        self,
        base_matrix: pd.DataFrame,
        cfg: dict,
        shared_cfg: dict,
    ) -> tuple[pd.DataFrame, BandThresholdDiagnostics | None]:
        """Run detection and additionally return diagnostic time series."""
        return self._run(
            base_matrix, cfg, shared_cfg, return_diagnostics=True,
        )

    def _run(
        self,
        base_matrix: pd.DataFrame,
        cfg: dict,
        shared_cfg: dict,
        *,
        return_diagnostics: bool,
    ) -> tuple[pd.DataFrame, BandThresholdDiagnostics | None]:
        """Shared implementation of detect / detect_with_diagnostics."""
        freq_range = cfg.get(
            "freq_range", self.DEFAULTS.get("freq_range")
        )
        bw_hours = float(cfg.get(
            "baseline_window_hours",
            self.DEFAULTS["baseline_window_hours"],
        ))
        bp = float(cfg.get(
            "baseline_percentile", self.DEFAULTS["baseline_percentile"],
        ))
        tp = float(cfg.get(
            "threshold_percentile", self.DEFAULTS["threshold_percentile"],
        ))
        min_cov = float(cfg.get(
            "min_band_coverage", self.DEFAULTS["min_band_coverage"],
        ))
        min_dur = int(cfg.get(
            "min_duration_s",
            shared_cfg.get("min_duration_s", self.DEFAULTS["min_duration_s"]),
        ))
        merge_gap = int(cfg.get(
            "merge_gap_s",
            shared_cfg.get("merge_gap_s", self.DEFAULTS["merge_gap_s"]),
        ))

        extras_columns = [
            "band_hz", "band_index",
            "peak_value_dB", "mean_value_dB",
            "baseline_dB", "threshold_dB",
            "delta_peak_dB", "percentile_at_peak",
        ]
        empty = pd.DataFrame(
            columns=CANONICAL_EVENT_COLUMNS + extras_columns
        )

        working = _filter_freq_range(base_matrix, freq_range)
        if working.empty:
            logger.warning(
                "band_threshold: no bands in working frequency range"
            )
            return empty, None

        base_res_s = _resolve_base_resolution_s(self)
        window_rows = max(2, int(bw_hours * 3600 / base_res_s))
        min_periods = max(2, int(window_rows * min_cov))

        logger.info(
            "band_threshold: %d bands; rolling window = %d rows "
            "(%s h at %s s/row); min_periods = %d; "
            "baseline p%g, threshold p%g; min_duration_s = %s, "
            "merge_gap_s = %s",
            len(working.columns), window_rows, bw_hours, base_res_s,
            min_periods, bp, tp, min_dur, merge_gap,
        )

        rolling = working.rolling(
            window=window_rows, min_periods=min_periods,
        )
        baseline_df = rolling.quantile(bp / 100.0)
        threshold_df = rolling.quantile(tp / 100.0)

        coverage_per_band = threshold_df.notna().sum(axis=0) / max(
            1, len(threshold_df)
        )
        usable_bands = [
            c for c in working.columns if coverage_per_band[c] > 0
        ]
        skipped_bands = [
            c for c in working.columns if c not in usable_bands
        ]
        if skipped_bands:
            logger.warning(
                "band_threshold: %d band(s) skipped due to insufficient "
                "coverage: %s%s",
                len(skipped_bands),
                skipped_bands[:5],
                " ..." if len(skipped_bands) > 5 else "",
            )
        if not usable_bands:
            logger.warning(
                "band_threshold: no usable bands; no events emitted"
            )
            return empty, None

        records: list[dict] = []
        next_event_id = 1
        index = working.index
        if not isinstance(index, pd.DatetimeIndex):
            raise AnalysisModuleError(
                "band_threshold requires a DatetimeIndex on the base matrix"
            )

        for band_idx, band_name in enumerate(usable_bands):
            band_values = working[band_name].to_numpy(dtype=np.float64)
            band_baseline = baseline_df[band_name].to_numpy(dtype=np.float64)
            band_threshold = threshold_df[band_name].to_numpy(dtype=np.float64)

            valid = np.isfinite(band_values) & np.isfinite(band_threshold)
            flags = valid & (band_values > band_threshold)

            event_windows = _flag_and_merge_array(
                flags, index, min_dur, merge_gap,
            )
            if not event_windows:
                continue

            band_hz = _get_frequency_value(band_name)

            for s_idx, e_idx in event_windows:
                window_values = band_values[s_idx:e_idx + 1]
                window_baseline = band_baseline[s_idx:e_idx + 1]
                window_threshold = band_threshold[s_idx:e_idx + 1]

                with np.errstate(all="ignore"):
                    peak_value = float(np.nanmax(window_values))
                    mean_value = float(np.nanmean(window_values))
                    mean_baseline = float(np.nanmean(window_baseline))
                    mean_threshold = float(np.nanmean(window_threshold))

                local_peak_idx = int(np.nanargmax(window_values))
                global_peak_idx = s_idx + local_peak_idx

                lo = max(0, global_peak_idx - window_rows + 1)
                hi = global_peak_idx + 1
                rolling_slice = band_values[lo:hi]
                rolling_slice = rolling_slice[np.isfinite(rolling_slice)]
                if rolling_slice.size > 0:
                    percentile_at_peak = float(
                        100.0 * (rolling_slice <= peak_value).sum()
                        / rolling_slice.size
                    )
                else:
                    percentile_at_peak = float("nan")

                delta_peak = peak_value - mean_threshold
                start_ts = index[s_idx]
                end_ts = index[e_idx]

                records.append({
                    "detector": self.name,
                    "event_id": next_event_id,
                    "start_time": start_ts,
                    "end_time": end_ts,
                    "duration_s": float(
                        (end_ts - start_ts).total_seconds() + 1.0
                    ),
                    "score": delta_peak,
                    "score_type": "band_delta_db",
                    "band_hz": band_hz,
                    "band_index": band_idx,
                    "peak_value_dB": peak_value,
                    "mean_value_dB": mean_value,
                    "baseline_dB": mean_baseline,
                    "threshold_dB": mean_threshold,
                    "delta_peak_dB": delta_peak,
                    "percentile_at_peak": percentile_at_peak,
                })
                next_event_id += 1

        diagnostics: BandThresholdDiagnostics | None = None
        if return_diagnostics:
            diagnostics = BandThresholdDiagnostics(
                values=working[usable_bands].copy(),
                baseline=baseline_df[usable_bands].copy(),
                threshold=threshold_df[usable_bands].copy(),
            )

        if not records:
            return empty, diagnostics

        df = pd.DataFrame(records)
        df = df.sort_values(
            ["start_time", "band_hz"]
        ).reset_index(drop=True)
        df["event_id"] = np.arange(1, len(df) + 1, dtype=int)

        logger.info(
            "band_threshold: emitted %d (band, event) row(s) across %d band(s)",
            len(df), df["band_hz"].nunique(),
        )

        return df, diagnostics


class AdaptiveThresholdLegacyDetector(EventDetector):
    """Legacy broadband-energy event detector."""

    name = "adaptive_threshold_legacy"

    REQUIRED_KEYS = {"threshold_db"}
    DEFAULTS = {
        "broadband_freq_range": None,
        "baseline_window_hours": 1.0,
        "baseline_percentile": 10.0,
        "min_duration_s": 3,
        "merge_gap_s": 30,
    }

    def validate_config(self, cfg: dict, shared_cfg: dict) -> None:
        errors: list[str] = []
        prefix = "event_detection.detectors[adaptive_threshold_legacy]"

        for key in self.REQUIRED_KEYS - set(cfg.keys()):
            errors.append(f"{prefix}: missing required key '{key}'")

        threshold_db = cfg.get("threshold_db")
        if threshold_db is not None and not isinstance(
            threshold_db, (int, float)
        ):
            errors.append(
                f"{prefix}.threshold_db must be numeric; "
                f"got {type(threshold_db).__name__}"
            )

        bw_hours = cfg.get(
            "baseline_window_hours", self.DEFAULTS["baseline_window_hours"]
        )
        if not isinstance(bw_hours, (int, float)) or bw_hours <= 0:
            errors.append(
                f"{prefix}.baseline_window_hours must be positive; "
                f"got {bw_hours}"
            )

        bp = cfg.get(
            "baseline_percentile", self.DEFAULTS["baseline_percentile"]
        )
        if not isinstance(bp, (int, float)) or not 0 < bp < 100:
            errors.append(
                f"{prefix}.baseline_percentile must be in (0, 100); got {bp}"
            )

        for key in ("min_duration_s", "merge_gap_s"):
            value = cfg.get(key, shared_cfg.get(key, self.DEFAULTS[key]))
            ok = isinstance(value, (int, float)) and (
                value > 0 if key == "min_duration_s" else value >= 0
            )
            if not ok:
                errors.append(
                    f"{prefix}.{key} (or shared) must be "
                    f"{'positive' if key == 'min_duration_s' else '>= 0'}; "
                    f"got {value}"
                )

        freq_range = cfg.get("broadband_freq_range")
        if freq_range is not None:
            ok = (
                isinstance(freq_range, (list, tuple))
                and len(freq_range) == 2
                and freq_range[0] < freq_range[1]
            )
            if not ok:
                errors.append(
                    f"{prefix}.broadband_freq_range must be [fmin, fmax] "
                    f"with fmin < fmax; got {freq_range}"
                )

        if errors:
            raise ValueError("\n".join(errors))

    def detect(
        self,
        base_matrix: pd.DataFrame,
        cfg: dict,
        shared_cfg: dict,
        stft=None,
    ) -> pd.DataFrame:
        freq_range = cfg.get(
            "broadband_freq_range",
            self.DEFAULTS["broadband_freq_range"],
        )
        working = _filter_freq_range(base_matrix, freq_range)

        broadband = _broadband_spl(working)

        bw_hours = float(cfg.get(
            "baseline_window_hours", self.DEFAULTS["baseline_window_hours"]
        ))
        bp = float(cfg.get(
            "baseline_percentile", self.DEFAULTS["baseline_percentile"]
        ))

        base_res_s = _resolve_base_resolution_s(self)
        window_rows = max(2, int(bw_hours * 3600 / base_res_s))
        min_periods_required = max(1, window_rows // 4)
        baseline = (
            broadband
            .rolling(window=window_rows, min_periods=min_periods_required)
            .quantile(bp / 100.0)
        )
        logger.info(
            "adaptive_threshold_legacy: rolling baseline window = "
            "%s rows (%s h at %s s/row); min_periods = %s",
            window_rows,
            bw_hours,
            base_res_s,
            min_periods_required,
        )

        threshold_db = float(cfg["threshold_db"])
        flags = (broadband > (baseline + threshold_db)).fillna(False)

        min_dur = int(cfg.get(
            "min_duration_s",
            shared_cfg.get("min_duration_s", self.DEFAULTS["min_duration_s"]),
        ))
        merge_gap = int(cfg.get(
            "merge_gap_s",
            shared_cfg.get("merge_gap_s", self.DEFAULTS["merge_gap_s"]),
        ))
        windows = _flag_and_merge(flags, min_dur, merge_gap)

        extras_columns = [
            "peak_spl_dB",
            "leq_dB",
            "spectral_centroid_hz",
            "baseline_spl_dB",
            "delta_peak_dB",
        ]
        records: list[dict] = []
        for i, (start, end) in enumerate(windows, start=1):
            event_bb = broadband.loc[start:end]
            event_base = baseline.loc[start:end]
            delta = event_bb - event_base
            score = float(np.nanmax(delta.to_numpy()))
            extras = {
                "baseline_spl_dB": float(np.nanmean(event_base.to_numpy())),
                "delta_peak_dB": score,
            }
            row = _build_event_row(
                detector_name=self.name,
                event_id=i,
                start=start,
                end=end,
                base_matrix=base_matrix,
                score=score,
                score_type="delta_db",
                extras=extras,
            )
            if row:
                records.append(row)

        if not records:
            return pd.DataFrame(
                columns=CANONICAL_EVENT_COLUMNS + extras_columns
            )
        return pd.DataFrame(records)


class PCAAnomalyDetector(EventDetector):
    """Detects spectral anomalies via PCA reconstruction error."""

    name = "anomaly"

    DEFAULTS = {
        "method": "pca",
        "n_components": 5,
        "variance_explained": None,
        "freq_range_hz": None,
        "baseline_trim_percentile": 95.0,
        "threshold_percentile": 99.5,
        "report_top_n_bands": 5,
        "min_duration_s": 3,
        "merge_gap_s": 30,
    }

    def validate_config(self, cfg: dict, shared_cfg: dict) -> None:
        errors: list[str] = []
        prefix = "event_detection.detectors[anomaly]"

        method = cfg.get("method", self.DEFAULTS["method"])
        if method != "pca":
            errors.append(
                f"{prefix}.method '{method}' not supported in this version; "
                f"only 'pca' is implemented"
            )

        n_comp = cfg.get("n_components", self.DEFAULTS["n_components"])
        var_exp = cfg.get(
            "variance_explained", self.DEFAULTS["variance_explained"]
        )
        if var_exp is None:
            if not isinstance(n_comp, int) or n_comp < 1:
                errors.append(
                    f"{prefix}.n_components must be a positive integer; "
                    f"got {n_comp}"
                )
        else:
            if not isinstance(var_exp, (int, float)) or not 0 < var_exp <= 1:
                errors.append(
                    f"{prefix}.variance_explained must be in (0, 1]; "
                    f"got {var_exp}"
                )

        trim = cfg.get(
            "baseline_trim_percentile",
            self.DEFAULTS["baseline_trim_percentile"],
        )
        if not isinstance(trim, (int, float)) or not 50 <= trim <= 100:
            errors.append(
                f"{prefix}.baseline_trim_percentile must be in [50, 100]; "
                f"got {trim}"
            )

        thresh = cfg.get(
            "threshold_percentile", self.DEFAULTS["threshold_percentile"]
        )
        if not isinstance(thresh, (int, float)) or not 0 < thresh < 100:
            errors.append(
                f"{prefix}.threshold_percentile must be in (0, 100); "
                f"got {thresh}"
            )

        top_n = cfg.get(
            "report_top_n_bands", self.DEFAULTS["report_top_n_bands"]
        )
        if not isinstance(top_n, int) or top_n < 0:
            errors.append(
                f"{prefix}.report_top_n_bands must be a non-negative integer; "
                f"got {top_n}"
            )

        for key in ("min_duration_s", "merge_gap_s"):
            value = cfg.get(key, shared_cfg.get(key, self.DEFAULTS[key]))
            ok = isinstance(value, (int, float)) and (
                value > 0 if key == "min_duration_s" else value >= 0
            )
            if not ok:
                errors.append(
                    f"{prefix}.{key} (or shared) must be "
                    f"{'positive' if key == 'min_duration_s' else '>= 0'}; "
                    f"got {value}"
                )

        freq_range = cfg.get("freq_range_hz")
        if freq_range is not None:
            ok = (
                isinstance(freq_range, (list, tuple))
                and len(freq_range) == 2
                and freq_range[0] < freq_range[1]
            )
            if not ok:
                errors.append(
                    f"{prefix}.freq_range_hz must be [fmin, fmax] with "
                    f"fmin < fmax; got {freq_range}"
                )

        if errors:
            raise ValueError("\n".join(errors))

    def detect(
        self,
        base_matrix: pd.DataFrame,
        cfg: dict,
        shared_cfg: dict,
        stft=None,
    ) -> pd.DataFrame:
        top_n = int(cfg.get(
            "report_top_n_bands", self.DEFAULTS["report_top_n_bands"]
        ))
        extras_columns = (
            [f"top_band_{j}_hz" for j in range(1, top_n + 1)]
            + [f"top_band_{j}_contribution" for j in range(1, top_n + 1)]
            + ["n_components", "explained_variance"]
        )
        empty = pd.DataFrame(columns=CANONICAL_EVENT_COLUMNS + extras_columns)

        freq_range = cfg.get("freq_range_hz", self.DEFAULTS["freq_range_hz"])
        working = _filter_freq_range(base_matrix, freq_range)

        working_clean = working.dropna(how="any")
        if working_clean.empty:
            raise AnalysisModuleError(
                "PCA anomaly detector: no complete rows after NaN removal"
            )

        X = working_clean.to_numpy(dtype=np.float64) #pylint: disable=invalid-name
        if X.shape[0] < 2:
            logger.warning(
                "PCA anomaly detector: fewer than 2 complete rows; "
                "returning no events"
            )
            return empty

        band_mean = X.mean(axis=0)
        band_std = X.std(axis=0, ddof=0)
        band_std = np.where(band_std < 1e-12, 1.0, band_std)
        Z = (X - band_mean) / band_std #pylint: disable=invalid-name

        intensity = np.linalg.norm(Z, axis=1)
        trim = float(cfg.get(
            "baseline_trim_percentile",
            self.DEFAULTS["baseline_trim_percentile"],
        ))
        cutoff = np.percentile(intensity, trim)
        baseline_mask = intensity <= cutoff
        Z_baseline = Z[baseline_mask] #pylint: disable=invalid-name
        if Z_baseline.shape[0] < 2:
            raise AnalysisModuleError(
                f"PCA anomaly detector: insufficient baseline samples "
                f"({Z_baseline.shape[0]}) after trimming at P{trim}"
            )

        Z_centred = Z_baseline - Z_baseline.mean(axis=0) #pylint: disable=invalid-name
        _, S, Vt = np.linalg.svd(Z_centred, full_matrices=False) #pylint: disable=invalid-name
        eigenvalues = (S ** 2) / max(1, Z_baseline.shape[0] - 1)
        total_var = float(eigenvalues.sum())

        var_exp_cfg = cfg.get(
            "variance_explained", self.DEFAULTS["variance_explained"]
        )
        if var_exp_cfg is not None:
            cum_ratio = np.cumsum(eigenvalues) / max(total_var, 1e-12)
            n_components = int(np.searchsorted(cum_ratio, var_exp_cfg) + 1)
            n_components = max(1, min(n_components, Vt.shape[0]))
        else:
            n_comp_cfg = int(cfg.get(
                "n_components", self.DEFAULTS["n_components"]
            ))
            n_components = max(1, min(n_comp_cfg, Vt.shape[0]))

        Vk = Vt[:n_components] #pylint: disable=invalid-name
        explained_var = float(
            eigenvalues[:n_components].sum() / max(total_var, 1e-12)
        )
        logger.info(
            "PCA anomaly: fitted %s components on %s baseline rows (%.1f%% variance explained)",
            n_components,
            f"{Z_baseline.shape[0]:,}",
            explained_var * 100,
        )

        projection = Z @ Vk.T
        reconstruction = projection @ Vk
        residual = Z - reconstruction
        score = np.linalg.norm(residual, axis=1)
        contribution = residual ** 2

        score_s = pd.Series(score, index=working_clean.index)
        contribution_df = pd.DataFrame(
            contribution,
            index=working_clean.index,
            columns=working_clean.columns,
        )

        thresh_p = float(cfg.get(
            "threshold_percentile", self.DEFAULTS["threshold_percentile"]
        ))
        threshold = float(np.percentile(score, thresh_p))
        flags_clean = score_s > threshold
        flags = flags_clean.reindex(base_matrix.index, fill_value=False)

        min_dur = int(cfg.get(
            "min_duration_s",
            shared_cfg.get("min_duration_s", self.DEFAULTS["min_duration_s"]),
        ))
        merge_gap = int(cfg.get(
            "merge_gap_s",
            shared_cfg.get("merge_gap_s", self.DEFAULTS["merge_gap_s"]),
        ))
        windows = _flag_and_merge(flags, min_dur, merge_gap)

        records: list[dict] = []
        for i, (start, end) in enumerate(windows, start=1):
            event_scores = score_s.loc[start:end].dropna()
            if event_scores.empty:
                continue
            peak_score = float(event_scores.max())

            event_contrib = contribution_df.loc[start:end].mean(axis=0)
            top_bands = event_contrib.nlargest(top_n)

            extras: dict = {}
            for j in range(1, top_n + 1):
                if j <= len(top_bands):
                    band_name = top_bands.index[j - 1]
                    extras[f"top_band_{j}_hz"] = _get_frequency_value(band_name)
                    extras[f"top_band_{j}_contribution"] = float(
                        top_bands.iloc[j - 1]
                    )
                else:
                    extras[f"top_band_{j}_hz"] = float("nan")
                    extras[f"top_band_{j}_contribution"] = float("nan")
            extras["n_components"] = int(n_components)
            extras["explained_variance"] = explained_var

            row = _build_event_row(
                detector_name=self.name,
                event_id=i,
                start=start,
                end=end,
                base_matrix=base_matrix,
                score=peak_score,
                score_type="reconstruction_error",
                extras=extras,
            )
            if row:
                records.append(row)

        if not records:
            return empty
        return pd.DataFrame(records)


class EventDetectionAnalysis(AnalysisModule):
    """Analysis-module wrapper that runs one or more event detectors."""

    name = "event_detection"

    # Detectors run on the base matrix; the annotated-spectrogram
    # sub-feature additionally needs the STFT, but only when it is
    # enabled — so the requirement is computed from config (refactor §7)
    # rather than declared as a static set.
    REQUIRES = frozenset({"base_matrix"})

    def required_substrates(self, module_cfg: dict | None = None) -> set[str]:
        substrates = set(self.REQUIRES)
        annotated = (module_cfg or {}).get("annotated_spectrogram") or {}
        if annotated.get("enabled", False):
            substrates.add("stft")
        return substrates

    def validate_config(self, cfg: dict) -> None:
        errors: list[str] = []

        output_format = cfg.get("output_format", "csv")
        if output_format != "csv":
            errors.append(
                f"event_detection.output_format must be 'csv'; "
                f"got '{output_format}'"
            )

        for key in ("min_duration_s", "merge_gap_s"):
            if key in cfg:
                value = cfg[key]
                ok = isinstance(value, (int, float)) and (
                    value > 0 if key == "min_duration_s" else value >= 0
                )
                if not ok:
                    errors.append(
                        f"event_detection.{key} (shared) must be "
                        f"{'positive' if key == 'min_duration_s' else '>= 0'};"
                        f" got {value}"
                    )

        detectors = cfg.get("detectors")
        if not isinstance(detectors, list) or not detectors:
            errors.append(
                "event_detection.detectors must be a non-empty list"
            )
        else:
            shared_cfg = {
                k: v for k, v in cfg.items() if k != "detectors"
            }
            for i, det_cfg in enumerate(detectors):
                if isinstance(det_cfg, dict) and not det_cfg.get(
                    "enabled", True
                ):
                    continue

                if not isinstance(det_cfg, dict):
                    errors.append(
                        f"event_detection.detectors[{i}] must be a dict; "
                        f"got {type(det_cfg).__name__}"
                    )
                    continue
                det_type = det_cfg.get("type")
                if det_type is None:
                    errors.append(
                        f"event_detection.detectors[{i}] missing 'type' key"
                    )
                    continue
                if det_type not in DETECTOR_REGISTRY:
                    available = ", ".join(sorted(DETECTOR_REGISTRY.keys()))
                    errors.append(
                        f"event_detection.detectors[{i}].type '{det_type}' "
                        f"is unknown. Available: {available}"
                    )
                    continue
                try:
                    detector = get_detector(det_type)
                    detector.validate_config(det_cfg, shared_cfg)
                except ValueError as exc:
                    errors.append(str(exc))

        for i, det_cfg in enumerate(cfg.get("detectors", []) or []):
            det_type = det_cfg.get("type", f"<index {i}>")
            if det_type == "band_threshold":
                prefix = f"event_detection.detectors[{i}, band_threshold]"
                _validate_plot_block(
                    module_name=prefix,
                    cfg=det_cfg,
                    valid_types={"threshold_diagnostic"},
                    errors=errors,
                )
                td = (det_cfg.get("plot") or {}).get("threshold_diagnostic", {}) or {}
                if "bands_hz" in td:
                    bands_hz = td["bands_hz"]
                    if not isinstance(bands_hz, list) or not bands_hz:
                        errors.append(
                            f"{prefix}.plot.threshold_diagnostic.bands_hz must be a "
                            f"non-empty list"
                        )
                    elif not all(isinstance(b, (int, float)) for b in bands_hz):
                        errors.append(
                            f"{prefix}.plot.threshold_diagnostic.bands_hz entries "
                            f"must be numeric"
                        )
                if "max_points_per_panel" in td:
                    mpp = td["max_points_per_panel"]
                    if not isinstance(mpp, int) or mpp <= 0:
                        errors.append(
                            f"{prefix}.plot.threshold_diagnostic."
                            f"max_points_per_panel must be a positive integer"
                        )

        ann = cfg.get("annotated_spectrogram")
        if ann is not None:
            if not isinstance(ann, dict):
                errors.append(
                    "event_detection.annotated_spectrogram must be a mapping or null"
                )
            else:
                if not isinstance(ann.get("enabled", False), bool):
                    errors.append(
                        "event_detection.annotated_spectrogram.enabled must be a boolean"
                    )
                include = ann.get("include_detectors")
                if include is not None and not (
                    isinstance(include, list)
                    and all(isinstance(x, str) for x in include)
                ):
                    errors.append(
                        "event_detection.annotated_spectrogram.include_detectors "
                        "must be null or a list of detector-type strings"
                    )
                fmt = ann.get("output_format", "png")
                if fmt not in {"png", "pdf"}:
                    errors.append(
                        f"event_detection.annotated_spectrogram.output_format must be "
                        f"'png' or 'pdf'; got '{fmt}'"
                    )
                time_chunk = ann.get("time_chunk")
                if time_chunk is not None:
                    if not isinstance(time_chunk, str):
                        errors.append(
                            f"event_detection.annotated_spectrogram.time_chunk "
                            f"must be a pandas offset alias string (e.g. '1h', "
                            f"'6h') or null; got "
                            f"{type(time_chunk).__name__}"
                        )
                    else:
                        try:
                            pd.tseries.frequencies.to_offset(time_chunk)
                        except (ValueError, TypeError):
                            errors.append(
                                f"event_detection.annotated_spectrogram.time_chunk "
                                f"'{time_chunk}' is not a valid pandas offset alias"
                            )

                time_bins = ann.get("time_bins", 12000)
                if time_bins is not None and (
                    not isinstance(time_bins, int) or time_bins <= 0
                ):
                    errors.append(
                        f"event_detection.annotated_spectrogram.time_bins must be "
                        f"a positive integer or null; got {time_bins}"
                    )

        if errors:
            raise ValueError("\n".join(errors))

    def run(
        self,
        base_matrix: pd.DataFrame,
        cfg: dict,
        output_dir: str,
    ) -> AnalysisResult:
        self.validate_config(cfg)
        self._validate_base_matrix(base_matrix)

        outputs: list[str] = []
        warnings: list[str] = []
        events_by_detector: dict[str, pd.DataFrame] = {}
        per_detector_summary: dict[str, dict] = {}

        shared_cfg = {k: v for k, v in cfg.items() if k != "detectors"}
        detector_configs = [
            d for d in cfg["detectors"] if d.get("enabled", True)
        ]

        if not detector_configs:
            logger.info("event_detection: all detectors disabled, skipping")
            return AnalysisResult(
                name=self.name,
                outputs=[],
                summary={"n_detectors": 0},
                warnings=["All detectors disabled in config"],
            )

        os.makedirs(output_dir, exist_ok=True)

        runtime_context = self._get_runtime_context()

        for det_cfg in detector_configs:
            det_type = det_cfg["type"]
            detector = get_detector(det_type)
            if hasattr(detector, "set_runtime_context"):
                detector.set_runtime_context(runtime_context) # type: ignore
            logger.info("Running event detector: %s", det_type)

            try:
                plot_cfg = det_cfg.get("plot") or {}
                plot_enabled = plot_cfg.get("enabled", False)

                if plot_enabled and hasattr(detector, "detect_with_diagnostics"):
                    events_df, diagnostics = detector.detect_with_diagnostics( #type: ignore
                        base_matrix, det_cfg, shared_cfg,
                    )
                else:
                    events_df = detector.detect(base_matrix, det_cfg, shared_cfg)
                    diagnostics = None
            except AnalysisModuleError:
                raise
            except Exception as exc:
                raise AnalysisModuleError(
                    f"Event detector '{det_type}' failed: {exc}"
                ) from exc

            n_events = len(events_df)
            output_file = os.path.join(
                output_dir, f"event_detection_{det_type}.csv"
            )
            events_df.to_csv(output_file, index=False)
            outputs.append(output_file)
            events_by_detector[det_type] = events_df

            plot_paths, plot_warns = self._generate_detector_plots(
                det_type=det_type,
                diagnostics=diagnostics,
                events_df=events_df,
                plot_cfg=plot_cfg,
                output_dir=output_dir,
            )
            outputs.extend(plot_paths)
            warnings.extend(plot_warns)

            logger.info(
                "  detector=%s: %d event row(s) → %s",
                det_type,
                n_events,
                output_file
            )
            per_detector_summary[det_type] = {
                "n_events": int(n_events),
                "output_file": output_file,
                "config": det_cfg,
            }

        ann_cfg = cfg.get("annotated_spectrogram") or {}
        if ann_cfg.get("enabled", False):
            ann_paths, ann_warns = self._generate_annotated_spectrogram(
                events_by_detector=events_by_detector,
                ann_cfg=ann_cfg,
                output_dir=output_dir,
            )
            outputs.extend(ann_paths)
            warnings.extend(ann_warns)

        return AnalysisResult(
            name=self.name,
            outputs=outputs,
            summary={
                "n_detectors": len(detector_configs),
                "detectors": per_detector_summary,
                "shared_config": shared_cfg,
                "total_events": int(
                    sum(s["n_events"] for s in per_detector_summary.values())
                ),
            },
            warnings=warnings,
        )

    def _generate_detector_plots(
        self,
        *,
        det_type: str,
        diagnostics,  # BandThresholdDiagnostics | None
        events_df: pd.DataFrame,
        plot_cfg: dict,
        output_dir: str,
    ) -> tuple[list[str], list[str]]:
        """Produce per-detector plots. Returns (paths, warnings)."""
        outputs: list[str] = []
        warnings: list[str] = []

        if not plot_cfg.get("enabled", False):
            return outputs, warnings

        try:
            import matplotlib.pyplot as plt
            from seasound.plotting.event_detection import (
                BandThresholdDiagnosticPlotter,
            )
        except ImportError as exc:
            msg = f"Event-detection plotting requires matplotlib: {exc}"
            warnings.append(msg)
            logger.warning(msg)
            return outputs, warnings

        types = plot_cfg.get("types", [])
        output_format = plot_cfg.get("output_format", "png")
        dpi = plot_cfg.get("dpi", 300)

        for kind in types:
            try:
                if kind == "threshold_diagnostic" and det_type == "band_threshold":
                    if diagnostics is None:
                        msg = (
                            f"band_threshold diagnostics unavailable "
                            f"(no usable bands); skipping {kind} plot."
                        )
                        warnings.append(msg)
                        logger.warning(msg)
                        continue
                    kind_cfg = plot_cfg.get(kind, {}) or {}
                    bands_hz = kind_cfg.get("bands_hz")
                    if not bands_hz:
                        msg = f"{kind} plot requires bands_hz; skipped."
                        warnings.append(msg)
                        logger.warning(msg)
                        continue
                    fig = BandThresholdDiagnosticPlotter(
                        diagnostics, events_df,
                    ).per_band(
                        bands_hz=bands_hz,
                        max_points_per_panel=kind_cfg.get(
                            "max_points_per_panel", 5000,
                        ),
                    )
                else:
                    msg = (
                        f"Plot type '{kind}' not supported for detector "
                        f"'{det_type}'; skipped."
                    )
                    warnings.append(msg)
                    logger.warning(msg)
                    continue

                plot_path = os.path.join(
                    output_dir,
                    f"event_detection_{det_type}_{kind}.{output_format}",
                )
                fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                outputs.append(plot_path)
                logger.info("Detector plot: %s", plot_path)
            except Exception as exc: #pylint: disable=broad-except
                msg = f"Plot '{kind}' for detector '{det_type}' failed: {exc}"
                warnings.append(msg)
                logger.warning(msg)

        return outputs, warnings


    def _generate_annotated_spectrogram(
        self,
        *,
        events_by_detector: dict[str, pd.DataFrame],
        ann_cfg: dict,
        output_dir: str,
    ) -> tuple[list[str], list[str]]:
        """Produce the cross-detector annotated spectrogram."""
        outputs: list[str] = []
        warnings: list[str] = []

        try:
            import matplotlib.pyplot as plt
            from seasound.core.stft import iter_stft_windows
            from seasound.core.output_layout import (
                resolve_spectrogram_output_dir,
            )
            from seasound.plotting.event_detection import (
                EventSpectrogramPlotter,
            )
        except ImportError as exc:
            msg = f"Annotated spectrogram requires matplotlib: {exc}"
            warnings.append(msg)
            logger.warning(msg)
            return outputs, warnings

        runtime_ctx = self._get_runtime_context()
        time_bins = ann_cfg.get("time_bins", 12000)
        time_chunk = ann_cfg.get("time_chunk")
        output_format = ann_cfg.get("output_format", "png")
        dpi = ann_cfg.get("dpi", 300)

        include = ann_cfg.get("include_detectors")
        if include is not None:
            events_subset = {
                k: v for k, v in events_by_detector.items() if k in include
            }
        else:
            events_subset = dict(events_by_detector)

        def _format_ts(ts: pd.Timestamp) -> str:
            return ts.strftime("%Y%m%dT%H%M%S")

        # Windowed STFT reads on the global downsampling grid (§8): one
        # figure per time_chunk window, each loading only its own native
        # frames. The per-window arrays match slicing the legacy
        # globally-downsampled matrix (gated, §9 test 7). target_dir is
        # created lazily on the first window so a store with no STFT
        # shards skips cleanly.
        target_dir: str | None = None
        for i, (window_start, window_end, stft_slice) in enumerate(
            iter_stft_windows(
                runtime_ctx,
                time_chunk=time_chunk,
                time_bins=time_bins,
                warnings=warnings,
            )
        ):
            if target_dir is None:
                target_dir = resolve_spectrogram_output_dir(
                    output_dir,
                    runtime_ctx.get("pipeline_config"),
                    which="annotated",
                )
                os.makedirs(target_dir, exist_ok=True)
            try:
                chunk_events = self._filter_events_to_window(
                    events_subset, window_start, window_end,
                )
                fig = EventSpectrogramPlotter(stft_slice, chunk_events).plot(
                    freq_range=ann_cfg.get("freq_range"),
                    db_range=ann_cfg.get("db_range"),
                    colormap=ann_cfg.get("colormap"),
                    preserve_time_gaps=ann_cfg.get("preserve_time_gaps", True),
                    annotation_styles=ann_cfg.get("annotation_styles"),
                    annotation_label=ann_cfg.get("annotation_label", False),
                )
                if time_chunk is None:
                    filename = f"annotated_spectrogram.{output_format}"
                else:
                    s = _format_ts(window_start)
                    e = _format_ts(window_end)
                    filename = (
                        f"annotated_spectrogram_{i:04d}_{s}_{e}.{output_format}"
                    )
                plot_path = os.path.join(target_dir, filename)
                fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                outputs.append(plot_path)
                logger.info("Annotated spectrogram: %s", plot_path)
            except Exception as exc: #pylint: disable=broad-exception-caught
                msg = (
                    f"Annotated spectrogram chunk "
                    f"({window_start} → {window_end}) failed: {exc}"
                )
                warnings.append(msg)
                logger.warning(msg)

        if target_dir is None:
            msg = (
                "Annotated spectrogram skipped: STFT data not available. "
                "Enable the spectrogram analysis or pre-compute STFT and re-run."
            )
            warnings.append(msg)
            logger.warning(msg)

        return outputs, warnings

    @staticmethod
    def _filter_events_to_window(
        events_by_detector: dict[str, pd.DataFrame],
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        """Return events filtered to those overlapping the window."""
        out: dict[str, pd.DataFrame] = {}
        for det, df in events_by_detector.items():
            if df is None or df.empty:
                out[det] = df
                continue
            starts = pd.to_datetime(df["start_time"])
            ends = pd.to_datetime(df["end_time"])
            mask = (ends >= window_start) & (starts <= window_end)
            out[det] = df[mask]
        return out


register_detector("band_threshold", BandThresholdDetector)
register_detector("adaptive_threshold_legacy", AdaptiveThresholdLegacyDetector)
register_detector("anomaly", PCAAnomalyDetector)
register_analysis("event_detection", EventDetectionAnalysis)

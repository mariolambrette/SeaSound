"""Internal helpers shared across SeaSound plotting modules."""

import logging
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_frequency_columns(df: pd.DataFrame) -> list[str]:
    """Return all frequency-band column names (those ending in 'Hz')."""
    return [c for c in df.columns if isinstance(c, str) and c.endswith("Hz")]


def frequency_values(columns: list[str]) -> list[float]:
    """Extract numeric frequencies from column names (e.g. '1000.0Hz' -> 1000.0)."""
    return [float(c[:-2]) for c in columns]


def hz_to_nearest_band(target_hz: float, columns: list[str]) -> str:
    """
    Return the column name whose centre frequency is closest to target_hz.

    Raises
    ------
    ValueError
        If columns is empty.
    """
    if not columns:
        raise ValueError("hz_to_nearest_band: columns list is empty.")
    freqs = frequency_values(columns)
    idx = int(np.argmin([abs(f - target_hz) for f in freqs]))
    return columns[idx]


def filter_frequency_range(
    df: pd.DataFrame,
    freq_range: tuple[float, float] | list[float] | None,
) -> pd.DataFrame:
    """
    Restrict columns to those whose frequency falls in freq_range (inclusive).

    Non-frequency columns are preserved unchanged and placed first in the
    returned DataFrame (matching the input order).
    """
    if freq_range is None:
        return df
    fmin, fmax = freq_range
    freq_cols = get_frequency_columns(df)
    keep = [c for c in freq_cols if fmin <= float(c[:-2]) <= fmax]
    non_freq = [c for c in df.columns if c not in freq_cols]
    return df[non_freq + keep]


def reindex_with_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reindex a DateTimeIndex DataFrame to a uniform step, NaN-filling any gaps.

    The step is inferred from the median of positive index differences.
    This is used so missing time periods render as blanks rather than being
    silently collapsed (important for duty-cycled deployments).
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("reindex_with_gaps requires a DatetimeIndex.")
    if len(df) < 2:
        return df

    diffs = df.index.to_series().diff().dropna()
    positive = diffs[diffs > pd.Timedelta(0)]
    if positive.empty:
        return df

    step = positive.median()
    if not isinstance(step, pd.Timedelta) or step == pd.Timedelta(0):
        return df

    full_index = pd.date_range(df.index.min(), df.index.max(), freq=step)
    return df.reindex(full_index)


def compute_grid_dims(n: int) -> tuple[int, int]:
    """
    Return (n_rows, n_cols) for the most square grid that holds n panels.

    Prefers width >= height (more columns than rows), which fits landscape
    figures better.

    Raises
    ------
    ValueError
        If n <= 0.
    """
    if n <= 0:
        raise ValueError(f"compute_grid_dims: n must be positive; got {n}")
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def subsample_evenly(items: list, k: int) -> list:
    """
    Return up to k items from `items`, evenly spaced (always includes the
    first and last when k >= 2).

    Used to subsample a long time series of windows down to a fixed panel
    count for the spectral_percentiles grid plot.
    """
    n = len(items)
    if k >= n:
        return list(items)
    if k == 1:
        return [items[0]]
    idx = np.linspace(0, n - 1, k).round().astype(int)
    return [items[i] for i in idx]


def format_time_label(ts) -> str:
    """Compact ISO-like timestamp for axis and panel labels."""
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M")

def _validate_plot_block(
    module_name: str,
    cfg: dict,
    valid_types: set[str],
    errors: list[str],
) -> None:
    """
    Validate a `plot:` block inside an analysis config.

    Accumulates errors into `errors` (matching the existing pattern). Each
    module passes its own ``module_name`` (for prefixed error messages) and
    the set of plot ``types`` it supports.
    """
    plot_cfg = cfg.get("plot")
    if plot_cfg is None:
        return  # Block absent; that's fine.
    if not isinstance(plot_cfg, dict):
        errors.append(
            f"{module_name}.config.plot must be a mapping or null; "
            f"got {type(plot_cfg).__name__}"
        )
        return

    enabled = plot_cfg.get("enabled", False)
    if not isinstance(enabled, bool):
        errors.append(f"{module_name}.config.plot.enabled must be a boolean")

    types = plot_cfg.get("types", [])
    if not isinstance(types, list):
        errors.append(
            f"{module_name}.config.plot.types must be a list; "
            f"got {type(types).__name__}"
        )
    else:
        for t in types:
            if t not in valid_types:
                errors.append(
                    f"{module_name}.config.plot.types entries must be in "
                    f"{sorted(valid_types)}; got '{t}'"
                )

    fmt = plot_cfg.get("output_format", "png")
    if fmt not in {"png", "pdf"}:
        errors.append(
            f"{module_name}.config.plot.output_format must be 'png' or 'pdf'; "
            f"got '{fmt}'"
        )

    dpi = plot_cfg.get("dpi", 300)
    if not isinstance(dpi, int) or dpi <= 0:
        errors.append(
            f"{module_name}.config.plot.dpi must be a positive integer; "
            f"got {dpi}"
        )

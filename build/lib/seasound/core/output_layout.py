"""
Output-directory layout helpers.

The spectrogram and annotated-spectrogram outputs share a
``spectrograms/`` folder under the run's output directory. When both
kinds are produced, they live in ``raw/`` and ``annotated/``
subfolders; when only one is produced, it sits directly inside
``spectrograms/``.
"""

import logging
import os

logger = logging.getLogger(__name__)


def resolve_spectrogram_output_dir(
    root_output_dir: str,
    pipeline_config,
    which: str,
) -> str:
    """
    Decide where spectrogram-class outputs should land.

    Layout:
    - Only one of {plain, annotated} enabled: ``<root>/spectrograms/``
    - Both enabled:                           ``<root>/spectrograms/<which>/``

    Parameters
    ----------
    root_output_dir : str
        The pipeline run's output directory.
    pipeline_config : PipelineConfig or None
        Full pipeline config. If None, falls back to the flat
        ``<root>/spectrograms/`` layout.
    which : str
        ``"raw"`` for plain spectrograms (SpectrogramAnalysis),
        ``"annotated"`` for the cross-detector annotated spectrogram.

    Returns
    -------
    str
        The path. The caller is responsible for ``os.makedirs(...)``.
    """
    if which not in ("raw", "annotated"):
        raise ValueError(
            f"resolve_spectrogram_output_dir: which must be 'raw' or "
            f"'annotated'; got {which!r}"
        )

    base = os.path.join(root_output_dir, "spectrograms")
    if pipeline_config is None:
        return base

    plain = _plain_spectrogram_enabled(pipeline_config)
    annotated = _annotated_spectrogram_enabled(pipeline_config)
    if plain and annotated:
        return os.path.join(base, which)
    return base


def _plain_spectrogram_enabled(pipeline_config) -> bool:
    """True if the plain ``spectrogram`` analysis is enabled."""
    spec = _lookup_analysis(pipeline_config, "spectrogram")
    return bool(spec.get("enabled", False)) if spec else False


def _annotated_spectrogram_enabled(pipeline_config) -> bool:
    """True if ``event_detection``'s ``annotated_spectrogram`` is enabled."""
    ed = _lookup_analysis(pipeline_config, "event_detection")
    if not ed or not ed.get("enabled", False):
        return False
    ed_cfg = ed.get("config", {}) or {}
    ann = ed_cfg.get("annotated_spectrogram") or {}
    return bool(ann.get("enabled", False))


def _lookup_analysis(pipeline_config, name: str) -> dict | None:
    """
    Look up one analysis's config block from PipelineConfig.

    The ``analyses`` container may be a dict-of-dicts (raw YAML shape)
    or a typed object (one attribute per analysis). Both are handled.
    Returns a dict with at least an ``enabled`` key, or None if the
    analysis is not present at all.
    """
    analyses = getattr(pipeline_config, "analyses", None)
    if analyses is None:
        return None

    if isinstance(analyses, dict):
        return analyses.get(name)

    obj = getattr(analyses, name, None)
    if obj is None:
        return None
    return {
        "enabled": getattr(obj, "enabled", False),
        "config": getattr(obj, "config", {}) or {},
    }

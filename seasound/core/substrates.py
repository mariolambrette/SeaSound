"""
seasound/core/substrates.py

Substrate dependency resolver (refactor plan §7).

Stage 1 has two producers — the ``base_matrix`` and the ``stft`` shards —
each a branch off the shared calibrated block. Rather than driving them
from two free-floating booleans, the producer set is derived from what
the *enabled* analyses declare they need (``AnalysisModule.REQUIRES`` /
``required_substrates``), then adjusted by explicit force overrides
(``pipeline.base_matrix_enabled`` / ``pipeline.stft_enabled``):

- ``None``  → the resolver decides from the enabled analyses' needs.
- ``True``  → force the producer on even when nothing needs it.
- ``False`` → force it off; an enabled analysis that needs it is then
  skipped with a warning (warn-and-skip), mirroring the existing
  graceful "STFT not available" behaviour in the annotated spectrogram.

Stage 5b wires :func:`resolve_producers` into the loop; the per-file
refinement (:func:`subtract_cached`) lets a run that previously produced
only base matrices add STFT on a re-run without recomputing anything.
"""

from __future__ import annotations

import logging

from seasound.analysis.registry import get_analysis
from seasound.core.config import PipelineConfig

logger = logging.getLogger(__name__)

BASE_MATRIX = "base_matrix"
STFT = "stft"
ALL_PRODUCERS = frozenset({BASE_MATRIX, STFT})


def _enabled_analyses(config: PipelineConfig):
    """Yield (name, entry) for each enabled analysis, mirroring the
    enable check in ``pipeline.run_analyses``."""
    for name, entry in (config.analyses or {}).items():
        if isinstance(entry, dict) and entry.get("enabled", False):
            yield name, entry


def _overrides(config: PipelineConfig) -> dict[str, bool | None]:
    return {
        BASE_MATRIX: config.pipeline.base_matrix_enabled,
        STFT: config.pipeline.stft_enabled,
    }


def required_producers(config: PipelineConfig) -> set[str]:
    """Union of substrates required by the enabled analyses, before any
    force override is applied. Unknown analysis names are ignored here
    (config validation reports them)."""
    needed: set[str] = set()
    for name, entry in _enabled_analyses(config):
        try:
            module = get_analysis(name)
        except ValueError:
            continue
        needed |= module.required_substrates(entry.get("config", {}))
    return needed


def resolve_producers(config: PipelineConfig) -> set[str]:
    """The producer set to run for this config: the union required by the
    enabled analyses, with each producer forced on/off by its override."""
    producers = required_producers(config)
    for producer, override in _overrides(config).items():
        if override is True:
            producers.add(producer)
        elif override is False:
            producers.discard(producer)
    return producers


def validate_substrates(config: PipelineConfig) -> list[str]:
    """Warnings for enabled analyses that need a force-disabled substrate.

    Warn-and-skip: the run continues and the affected analysis degrades
    gracefully (e.g. the spectrogram emits no figures). Returns the
    messages so the caller controls logging; nothing is logged here.
    """
    forced_off = {p for p, override in _overrides(config).items() if override is False}
    if not forced_off:
        return []

    warnings: list[str] = []
    for name, entry in _enabled_analyses(config):
        try:
            module = get_analysis(name)
        except ValueError:
            continue
        missing = module.required_substrates(entry.get("config", {})) & forced_off
        for substrate in sorted(missing):
            warnings.append(
                f"Analysis '{name}' requires the '{substrate}' substrate, "
                f"but pipeline.{substrate}_enabled is set to false; '{name}' "
                f"will be skipped or produce no output."
            )
    return warnings


def subtract_cached(resolved: set[str], cached: set[str]) -> set[str]:
    """Per-file refinement (§7): the producers actually run for a file are
    the resolved set minus whatever is already cached and complete for it,
    so enabling a new product re-streams each file for that product only."""
    return set(resolved) - set(cached)

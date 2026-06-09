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

Validation comes in two severities, matching the rule "a config *choice*
degrades gracefully; an actual *failure* stops loudly":

- :func:`validate_substrates` — *warnings* for the user choice of
  force-disabling a substrate an enabled analysis needs (warn-and-skip).
- :func:`validate_analyses_registered` / :func:`validate_resolved_coverage`
  — *errors* for an enabled analysis that does not resolve, or a needed
  substrate that goes missing for any reason other than being force-disabled.
  This is the insurance against the registry import-order fragility silently
  dropping a substrate (e.g. STFT) in a future refactor.
"""

from __future__ import annotations

import logging

from seasound.analysis.registry import get_analysis, list_registered
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


def validate_analyses_registered(config: PipelineConfig) -> list[str]:
    """Errors for enabled analyses that do not resolve to a registered,
    instantiable module.

    The hard-error counterpart to :func:`validate_substrates`. An enabled
    analysis that cannot be resolved contributes *no* substrate needs —
    :func:`required_producers` skips it via ``except ValueError`` — so
    Stage 1 could quietly run without a substrate the user asked for. That
    is the registry import-order failure mode (a module dropped from
    ``analysis/__init__`` self-registration, or a resolve call reached
    before the analyses are imported). Surfacing it as an error makes the
    failure loud instead of silent, and also catches a mistyped analysis
    name. ``list_registered()`` membership is checked first so the message
    can distinguish "not registered" from "registered but failed to
    instantiate". Returns messages; the caller logs/raises.
    """
    registered = set(list_registered())
    errors: list[str] = []
    for name, _entry in _enabled_analyses(config):
        if name not in registered:
            available = ", ".join(sorted(registered)) or "none"
            errors.append(
                f"Enabled analysis '{name}' is not registered "
                f"(available: {available}). Check the name for a typo, or "
                f"ensure its module is imported (seasound.analysis)."
            )
            continue
        # Registered, but instantiation can still fail — surface that too.
        try:
            get_analysis(name)
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def validate_resolved_coverage(
    config: PipelineConfig, resolved: set[str]
) -> list[str]:
    """Errors when an enabled, registered analysis needs a substrate that is
    neither in ``resolved`` nor explicitly force-disabled.

    Force-disabling a needed substrate is a user choice and warns
    (:func:`validate_substrates`); a substrate going missing for any *other*
    reason is a resolver/registration bug and errors. A belt-and-braces
    backstop to :func:`validate_analyses_registered`: that function catches
    an analysis that does not resolve at all, this one catches a resolved
    analysis whose declared substrate still failed to make the producer set.
    Returns messages; the caller logs/raises.
    """
    forced_off = {p for p, override in _overrides(config).items() if override is False}
    errors: list[str] = []
    for name, entry in _enabled_analyses(config):
        try:
            module = get_analysis(name)
        except ValueError:
            continue  # already reported by validate_analyses_registered
        needed = module.required_substrates(entry.get("config", {}))
        for substrate in sorted(needed - resolved - forced_off):
            errors.append(
                f"Analysis '{name}' requires the '{substrate}' substrate, but "
                f"it was not produced and is not force-disabled "
                f"(pipeline.{substrate}_enabled). This indicates a resolver or "
                f"registration error, not a configuration choice."
            )
    return errors


def subtract_cached(resolved: set[str], cached: set[str]) -> set[str]:
    """Per-file refinement (§7): the producers actually run for a file are
    the resolved set minus whatever is already cached and complete for it,
    so enabling a new product re-streams each file for that product only."""
    return set(resolved) - set(cached)

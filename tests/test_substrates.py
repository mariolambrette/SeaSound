"""
§9 test 6 — substrate dependency resolver (refactor plan §7).

Asserts that enabled-analysis combinations produce the correct producer
set, that the annotated-spectrogram sub-feature pulls in STFT only when
enabled, that force overrides add/remove producers, that a force-disabled
substrate needed by an enabled analysis emits a warn-and-skip warning,
and that per-file subtraction yields the missing products only.

Also covers the hard-error preflight added with config hardening:
validate_analyses_registered (an enabled analysis must resolve to a
registered, instantiable module) and validate_resolved_coverage (a needed
substrate missing for any reason other than being force-disabled is a bug,
not a config choice).
"""

import pytest

# Importing the modules registers them so get_analysis can resolve them.
import seasound.analysis.spectrogram  # noqa: F401
import seasound.analysis.event_detection  # noqa: F401
from seasound.analysis.base import AnalysisModule
from seasound.analysis.registry import (
    ANALYSIS_REGISTRY,
    get_analysis,
    register_analysis,
)
from seasound.core.config import PipelineConfig, ProcessingConfig
from seasound.core.substrates import (
    ALL_PRODUCERS,
    BASE_MATRIX,
    STFT,
    required_producers,
    resolve_producers,
    subtract_cached,
    validate_analyses_registered,
    validate_resolved_coverage,
    validate_substrates,
)


def _config(analyses=None, *, base_matrix_enabled=None, stft_enabled=None):
    return PipelineConfig(
        pipeline=ProcessingConfig(
            base_matrix_enabled=base_matrix_enabled,
            stft_enabled=stft_enabled,
        ),
        analyses=analyses or {},
    )


def _an(enabled=True, config=None):
    return {"enabled": enabled, "config": config or {}}


# --- producer set from enabled analyses -----------------------------------

def test_no_enabled_analyses_needs_nothing():
    assert resolve_producers(_config({})) == set()


def test_spectrogram_needs_stft_and_base():
    cfg = _config({"spectrogram": _an()})
    assert resolve_producers(cfg) == {STFT, BASE_MATRIX}


def test_event_detection_without_annotated_needs_base_only():
    cfg = _config({"event_detection": _an(config={"detectors": []})})
    assert resolve_producers(cfg) == {BASE_MATRIX}


def test_event_detection_with_annotated_pulls_in_stft():
    cfg = _config({
        "event_detection": _an(
            config={"annotated_spectrogram": {"enabled": True}, "detectors": []}
        )
    })
    assert resolve_producers(cfg) == {BASE_MATRIX, STFT}


def test_disabled_analysis_is_ignored():
    cfg = _config({"spectrogram": _an(enabled=False)})
    assert resolve_producers(cfg) == set()


def test_union_across_enabled_analyses():
    cfg = _config({
        "spectrogram": _an(),
        "event_detection": _an(config={"detectors": []}),
    })
    assert resolve_producers(cfg) == {STFT, BASE_MATRIX}


# --- force overrides -------------------------------------------------------

def test_force_on_stft_without_consumer():
    cfg = _config(
        {"event_detection": _an(config={"detectors": []})},
        stft_enabled=True,
    )
    assert resolve_producers(cfg) == {BASE_MATRIX, STFT}


def test_force_on_base_matrix_with_nothing_enabled():
    cfg = _config({}, base_matrix_enabled=True)
    assert resolve_producers(cfg) == {BASE_MATRIX}


def test_force_off_stft_removes_it():
    cfg = _config({"spectrogram": _an()}, stft_enabled=False)
    assert resolve_producers(cfg) == {BASE_MATRIX}


# --- validation (warn-and-skip) -------------------------------------------

def test_no_warning_when_nothing_forced_off():
    cfg = _config({"spectrogram": _an()})
    assert validate_substrates(cfg) == []


def test_force_off_required_substrate_warns():
    cfg = _config({"spectrogram": _an()}, stft_enabled=False)
    warnings = validate_substrates(cfg)
    assert len(warnings) == 1
    assert "spectrogram" in warnings[0]
    assert STFT in warnings[0]


def test_force_off_unneeded_substrate_does_not_warn():
    # event_detection without annotated needs only base_matrix, so a
    # force-off of stft is irrelevant to it.
    cfg = _config(
        {"event_detection": _an(config={"detectors": []})},
        stft_enabled=False,
    )
    assert validate_substrates(cfg) == []


def test_force_off_warns_for_conditional_annotated_need():
    cfg = _config(
        {"event_detection": _an(
            config={"annotated_spectrogram": {"enabled": True}, "detectors": []}
        )},
        stft_enabled=False,
    )
    warnings = validate_substrates(cfg)
    assert len(warnings) == 1
    assert "event_detection" in warnings[0] and STFT in warnings[0]


# --- per-file subtraction --------------------------------------------------

def test_subtract_cached_leaves_only_missing():
    assert subtract_cached({BASE_MATRIX, STFT}, {BASE_MATRIX}) == {STFT}


def test_subtract_cached_fully_cached_is_empty():
    assert subtract_cached({BASE_MATRIX}, {BASE_MATRIX}) == set()


def test_subtract_cached_nothing_cached_keeps_all():
    assert subtract_cached({BASE_MATRIX, STFT}, set()) == {BASE_MATRIX, STFT}


# --- invariant across every registered analysis ---------------------------

def test_all_registered_analyses_declare_known_substrates():
    """required_substrates for any registered analysis (empty config) is a
    non-empty subset of the known producers — scales to ltsa/tob_levels/
    spectral_percentiles in the full repo without naming them here."""
    assert ANALYSIS_REGISTRY, "no analyses registered"
    for name in ANALYSIS_REGISTRY:
        subs = get_analysis(name).required_substrates({})
        assert subs, f"{name} declares no substrates"
        assert subs <= set(ALL_PRODUCERS), f"{name} declares unknown substrate: {subs}"


# --- registered-analysis preflight (hard error) ---------------------------

def test_registered_enabled_analysis_has_no_errors():
    cfg = _config({"spectrogram": _an()})
    assert validate_analyses_registered(cfg) == []


def test_unregistered_enabled_analysis_errors():
    cfg = _config({"ghosts": _an()})
    errors = validate_analyses_registered(cfg)
    assert len(errors) == 1
    assert "ghosts" in errors[0]
    assert "not registered" in errors[0]


def test_unregistered_disabled_analysis_is_ignored():
    cfg = _config({"ghosts": _an(enabled=False)})
    assert validate_analyses_registered(cfg) == []


def test_registered_but_uninstantiable_analysis_errors():
    # get_analysis re-raises an instantiation failure as ValueError; the
    # preflight must surface it rather than silently dropping the analysis.
    class _BoomAnalysis(AnalysisModule):
        name = "boom"

        def __init__(self):
            raise RuntimeError("boom")

        def validate_config(self, cfg):  # pragma: no cover - never reached
            ...

        def run(self, base_matrix, cfg, output_dir):  # pragma: no cover
            ...

    register_analysis("boom", _BoomAnalysis)
    try:
        errors = validate_analyses_registered(_config({"boom": _an()}))
        assert len(errors) == 1
        assert "boom" in errors[0]
    finally:
        ANALYSIS_REGISTRY.pop("boom", None)


# --- resolved-coverage backstop (hard error) ------------------------------

def test_coverage_ok_when_all_required_resolved():
    cfg = _config({"spectrogram": _an()})
    assert validate_resolved_coverage(cfg, {BASE_MATRIX, STFT}) == []


def test_coverage_errors_when_required_substrate_missing():
    # spectrogram needs STFT; absent from the resolved set and not
    # force-disabled => resolver/registration bug, not a user choice.
    cfg = _config({"spectrogram": _an()})
    errors = validate_resolved_coverage(cfg, {BASE_MATRIX})
    assert len(errors) == 1
    assert "spectrogram" in errors[0] and STFT in errors[0]


def test_coverage_silent_when_missing_substrate_is_force_disabled():
    # Same missing STFT, but force-disabled: handled by validate_substrates
    # (warn-and-skip), so coverage stays silent.
    cfg = _config({"spectrogram": _an()}, stft_enabled=False)
    assert validate_resolved_coverage(cfg, {BASE_MATRIX}) == []

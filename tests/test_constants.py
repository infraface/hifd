"""Tests for hifd.constants."""

import pytest

from hifd import constants


def test_default_constants_values_match_paper():
    c = constants.DEFAULT_CONSTANTS
    assert c["Delta_age"] == 100.0
    assert c["tau_NME"] == 0.10
    assert c["theta_max_deg"] == 90.0
    assert c["tau_HR"] == 20.0
    assert c["tau_N"] == 5.0
    assert c["alpha"] == 0.5
    assert c["epsilon"] == 1.0e-6


def test_default_constants_are_immutable():
    with pytest.raises(TypeError):
        constants.DEFAULT_CONSTANTS["tau_HR"] = 1.0  # type: ignore[index]


def test_default_profiles_values_match_paper():
    p = constants.DEFAULT_PROFILES
    assert p["privacy_first"]["P"] == 0.50
    assert p["balanced"]["P"] == 0.20
    assert p["clinical"]["U3"] == 0.40
    # All profile weights sum to 1.0
    for name, w in p.items():
        s = sum(w.values())
        assert abs(s - 1.0) < 1e-9, f"profile {name} weights sum to {s}, not 1.0"


def test_default_profiles_are_immutable():
    with pytest.raises(TypeError):
        constants.DEFAULT_PROFILES["balanced"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        constants.DEFAULT_PROFILES["balanced"]["P"] = 0.9  # type: ignore[index]


def test_schema_version():
    assert constants.SCHEMA_VERSION == "1.0"


# ----- Public API contract ---------------------------------------------------

def test_top_level_public_api_present():
    import hifd
    for name in ("score_age", "score_categorical", "score_landmark",
                 "score_gaze", "score_bvp", "score_hr",
                 "privacy_score", "privacy_ensemble", "quality_score",
                 "U1", "U2", "U3",
                 "composite", "compose_profiles",
                 "run_pipeline", "compute_agreements", "aggregate", "compose",
                 "load_predictions", "validate_pair",
                 "DEFAULT_CONSTANTS", "DEFAULT_PROFILES", "SCHEMA_VERSION", "__version__"):
        assert hasattr(hifd, name), f"hifd.{name} missing"


def test_top_level___all___matches():
    import hifd
    assert "score_age" in hifd.__all__
    assert "composite" in hifd.__all__
    assert "DEFAULT_CONSTANTS" in hifd.__all__

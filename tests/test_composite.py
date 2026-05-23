"""Tests for hifd.composite."""

import pytest

from hifd import compose_profiles, composite, load_profiles_yaml
from hifd import constants as const


def test_composite_all_equal_returns_same_value():
    w = {"P": 0.2, "Q": 0.2, "U1": 0.2, "U2": 0.2, "U3": 0.2}
    assert composite(0.5, 0.5, 0.5, 0.5, 0.5, weights=w, epsilon=0.0) == pytest.approx(0.5)


def test_composite_drops_none_and_renormalizes():
    # Only P=0.5 present; profile weight 0.5 → weight renormalizes to 1, score=0.5.
    w = {"P": 0.5, "Q": 0.125, "U1": 0.125, "U2": 0.125, "U3": 0.125}
    assert composite(0.5, None, None, None, None, weights=w, epsilon=0.0) == pytest.approx(0.5)


def test_composite_all_none_returns_zero():
    w = {"P": 0.2, "Q": 0.2, "U1": 0.2, "U2": 0.2, "U3": 0.2}
    assert composite(None, None, None, None, None, weights=w, epsilon=0.0) == 0.0


def test_composite_matches_hand_computation():
    # Two components, equal weights.
    # HM = sum(w) / sum(w_k / x_k) with w renormalized.
    # P=0.4, Q=0.8, w={P:0.5,Q:0.5}, eps=0.
    # → 1 / (0.5/0.4 + 0.5/0.8) = 1 / (1.25 + 0.625) = 1 / 1.875 ≈ 0.5333…
    w = {"P": 0.5, "Q": 0.5}
    expected = 1.0 / (0.5 / 0.4 + 0.5 / 0.8)
    out = composite(0.4, 0.8, None, None, None, weights=w, epsilon=0.0)
    assert out == pytest.approx(expected)


def test_composite_zero_weight_sum_returns_zero():
    w = {"P": 0.0, "Q": 0.0, "U1": 0.0, "U2": 0.0, "U3": 0.0}
    assert composite(0.5, 0.5, 0.5, 0.5, 0.5, weights=w, epsilon=0.0) == 0.0


def test_compose_profiles_uses_default_when_none():
    out = compose_profiles(0.5, 0.5, 0.5, 0.5, 0.5, epsilon=0.0)
    assert set(out.keys()) == set(const.DEFAULT_PROFILES.keys())
    for v in out.values():
        assert v == pytest.approx(0.5)


def test_compose_profiles_accepts_custom_profiles():
    custom = {"only_p": {"P": 1.0}}
    out = compose_profiles(0.4, 0.8, None, None, None, profiles=custom, epsilon=0.0)
    assert out == {"only_p": pytest.approx(0.4)}


def test_load_profiles_yaml(tmp_path):
    f = tmp_path / "profiles.yaml"
    f.write_text("test: {P: 0.5, Q: 0.5}\n")
    out = load_profiles_yaml(str(f))
    assert out == {"test": {"P": 0.5, "Q": 0.5}}


def test_composite_default_epsilon_avoids_division_by_zero():
    """Spec mandates default epsilon=1e-6 so zero components don't blow up."""
    w = {"P": 0.5, "Q": 0.5}
    # P=0 with default epsilon — should NOT raise ZeroDivisionError.
    out = composite(0.0, 0.8, None, None, None, weights=w)
    # Result is dominated by P=0 → small but finite.
    assert out >= 0.0
    assert out < 0.01  # Effectively zero, but finite.

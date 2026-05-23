"""Composite HiFD score (weighted harmonic mean) and profile compositions."""

from __future__ import annotations

from collections.abc import Mapping

import yaml

from .constants import DEFAULT_PROFILES

Profile = Mapping[str, float]


def composite(
    P: float | None,
    Q: float | None,
    U1: float | None,
    U2: float | None,
    U3: float | None,
    weights: Profile,
    epsilon: float = 1.0e-6,
) -> float:
    """Weighted harmonic mean over the components that are not None.

    Weights for missing components are dropped and the remainder renormalized.
    Returns 0.0 if no components are present or all weights are zero.
    """
    components = {"P": P, "Q": Q, "U1": U1, "U2": U2, "U3": U3}
    active = {k: v for k, v in components.items() if v is not None}
    if not active:
        return 0.0

    w_active = {k: float(weights[k]) for k in active if k in weights}
    w_sum = sum(w_active.values())
    if w_sum <= 0.0:
        return 0.0
    w_active = {k: v / w_sum for k, v in w_active.items()}

    num = sum(w_active.values())
    den = sum(w_active[k] / (active[k] + epsilon) for k in w_active)
    return num / den


def compose_profiles(
    P: float | None,
    Q: float | None,
    U1: float | None,
    U2: float | None,
    U3: float | None,
    profiles: Mapping[str, Profile] | None = None,
    epsilon: float = 1.0e-6,
) -> dict[str, float]:
    """Compute composite score under each profile. ``profiles=None`` uses DEFAULT_PROFILES."""
    profs = profiles if profiles is not None else DEFAULT_PROFILES
    return {name: composite(P, Q, U1, U2, U3, w, epsilon) for name, w in profs.items()}


def load_profiles_yaml(path: str) -> dict[str, dict[str, float]]:
    """Read a YAML file mapping profile_name → weight dict."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected top-level mapping, got {type(raw).__name__}")
    return {name: dict(weights) for name, weights in raw.items()}

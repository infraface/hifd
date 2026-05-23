"""Per-task agreement primitives.

Each ``score_*`` returns a scalar in [0, 1] for one paired sample. Batch
variants take ndarray inputs and return ndarrays.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

# ----- 4.1 Age ---------------------------------------------------------------

def score_age(a_orig: float, a_deid: float, Delta_age: float = 100.0) -> float:
    return float(max(0.0, 1.0 - abs(a_orig - a_deid) / Delta_age))


def batch_score_age(
    a_orig: NDArray[np.floating[Any]],
    a_deid: NDArray[np.floating[Any]],
    Delta_age: float = 100.0,
) -> NDArray[np.floating[Any]]:
    return np.maximum(0.0, 1.0 - np.abs(a_orig - a_deid) / Delta_age)


# ----- 4.2 Categorical -------------------------------------------------------

def score_categorical(p_orig: ArrayLike, p_deid: ArrayLike) -> float:
    p_o = np.asarray(p_orig, dtype=np.float64)
    p_d = np.asarray(p_deid, dtype=np.float64)
    assert p_o.shape == p_d.shape, f"shape mismatch: {p_o.shape} vs {p_d.shape}"
    tv = 0.5 * float(np.sum(np.abs(p_o - p_d)))
    return 1.0 - tv


def batch_score_categorical(
    p_orig: NDArray[np.floating[Any]], p_deid: NDArray[np.floating[Any]]
) -> NDArray[np.floating[Any]]:
    tv = 0.5 * np.sum(np.abs(p_orig - p_deid), axis=1)
    return cast(NDArray[np.floating[Any]], 1.0 - tv)


# ----- 4.3 Landmark ----------------------------------------------------------

def score_landmark(
    L_orig: ArrayLike,
    L_deid: ArrayLike,
    left_eye: ArrayLike,
    right_eye: ArrayLike,
    tau_NME: float = 0.10,
) -> float:
    L_o = np.asarray(L_orig, dtype=np.float64)
    L_d = np.asarray(L_deid, dtype=np.float64)
    le = np.asarray(left_eye, dtype=np.float64)
    re = np.asarray(right_eye, dtype=np.float64)
    d_io = float(np.linalg.norm(le - re))
    if d_io < 1e-3:
        return 0.0
    nme = float(np.mean(np.linalg.norm(L_o - L_d, axis=1))) / d_io
    return max(0.0, 1.0 - nme / tau_NME)


# ----- 4.4 Gaze --------------------------------------------------------------

def score_gaze(
    g_orig: ArrayLike, g_deid: ArrayLike, theta_max_deg: float = 90.0
) -> float:
    g_o = np.asarray(g_orig, dtype=np.float64)
    g_d = np.asarray(g_deid, dtype=np.float64)
    n_o = float(np.linalg.norm(g_o))
    n_d = float(np.linalg.norm(g_d))
    if n_o < 1e-12 or n_d < 1e-12:
        return 0.0
    g_o = g_o / n_o
    g_d = g_d / n_d
    cos_sim = float(np.clip(np.dot(g_o, g_d), -1.0, 1.0))
    theta = float(np.degrees(np.arccos(cos_sim)))
    return max(0.0, 1.0 - theta / theta_max_deg)


# ----- 4.5 BVP (rPPG waveform) -----------------------------------------------

def score_bvp(b_orig: ArrayLike, b_deid: ArrayLike) -> float:
    b_o = np.asarray(b_orig, dtype=np.float64)
    b_d = np.asarray(b_deid, dtype=np.float64)
    T = min(len(b_o), len(b_d))
    b_o, b_d = b_o[:T], b_d[:T]
    if np.std(b_o) < 1e-9 or np.std(b_d) < 1e-9:
        return 0.0
    rho = float(np.corrcoef(b_o, b_d)[0, 1])
    if not np.isfinite(rho):
        return 0.0
    return max(0.0, rho)


# ----- 4.6 Heart rate --------------------------------------------------------

def score_hr(r_orig: float, r_deid: float, tau_HR: float = 20.0) -> float:
    return float(max(0.0, 1.0 - abs(r_orig - r_deid) / tau_HR))


def batch_score_hr(
    r_orig: NDArray[np.floating[Any]],
    r_deid: NDArray[np.floating[Any]],
    tau_HR: float = 20.0,
) -> NDArray[np.floating[Any]]:
    return np.maximum(0.0, 1.0 - np.abs(r_orig - r_deid) / tau_HR)

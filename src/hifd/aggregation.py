"""Utility-level aggregation: per-sample U1/U2/U3 and per-method aggregation."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

# ----- 5.1 Per-sample level scores -------------------------------------------

def U1(s_age: float, s_gender: float, s_eth: float, s_macro: float, s_lmk: float) -> float:
    return float(np.mean([s_age, s_gender, s_eth, s_macro, s_lmk]))


def U2(s_gaze: float, s_me: float | None = None) -> float:
    if s_me is not None:
        return 0.5 * (s_gaze + s_me)
    return s_gaze


def U3(bvp_scores: list[float], hr_scores: list[float]) -> float:
    per_est = 0.5 * (np.asarray(bvp_scores) + np.asarray(hr_scores))
    return float(np.mean(per_est))


# ----- Vectorized versions ---------------------------------------------------

def batch_U1(
    s_age: NDArray[np.floating[Any]],
    s_gender: NDArray[np.floating[Any]],
    s_eth: NDArray[np.floating[Any]],
    s_macro: NDArray[np.floating[Any]],
    s_lmk: NDArray[np.floating[Any]],
) -> NDArray[np.floating[Any]]:
    return cast(
        NDArray[np.floating[Any]],
        np.mean(np.stack([s_age, s_gender, s_eth, s_macro, s_lmk], axis=0), axis=0),
    )


def batch_U2(
    s_gaze: NDArray[np.floating[Any]], s_me: NDArray[np.floating[Any]] | None = None
) -> NDArray[np.floating[Any]]:
    if s_me is not None:
        return cast(NDArray[np.floating[Any]], 0.5 * (s_gaze + s_me))  # type: ignore[redundant-cast]
    return s_gaze.copy()


def batch_U3(
    bvp_per_est: list[NDArray[np.floating[Any]]],
    hr_per_est: list[NDArray[np.floating[Any]]],
) -> NDArray[np.floating[Any]]:
    combined = [0.5 * (b + h) for b, h in zip(bvp_per_est, hr_per_est, strict=False)]
    return cast(
        NDArray[np.floating[Any]], np.mean(np.stack(combined, axis=0), axis=0)
    )


# ----- 5.2 Per-method aggregation --------------------------------------------

_DIMENSION_MAP = {
    "age": "L1", "gender": "L1", "ethnicity": "L1", "macro_exp": "L1", "landmark": "L1",
    "gaze": "L2", "micro_exp": "L2",
    "bvp": "L3", "hr": "L3",
    "P": "Privacy", "Q": "Quality",
    "U1": "L1", "U2": "L2", "U3": "L3",
}


def aggregate_method_scores(
    per_sample: dict[str, NDArray[np.floating[Any]]], method_name: str
) -> pd.DataFrame:
    """Aggregate per-sample arrays into a method-level dataframe."""
    rows = [
        {
            "method": method_name,
            "dimension": _DIMENSION_MAP.get(name, "Other"),
            "sub_score": name,
            "value": float(np.mean(arr)),
            "n_samples": len(arr),
            "std": float(np.std(arr)),
        }
        for name, arr in per_sample.items()
    ]
    return pd.DataFrame(rows)

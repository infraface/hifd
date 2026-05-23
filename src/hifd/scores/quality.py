"""Quality score combining LPIPS (perceptual) and NIQE (no-reference)."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray


def quality_score(
    lpips: float, niqe: float, tau_N: float = 5.0, alpha: float = 0.5
) -> float:
    """Q = alpha * (1 - LPIPS_clipped) + (1 - alpha) * NIQE*."""
    perc = 1.0 - min(max(float(lpips), 0.0), 1.0)
    niqe_star = max(0.0, min(1.0, (tau_N - float(niqe)) / tau_N))
    return float(alpha * perc + (1.0 - alpha) * niqe_star)


def batch_quality_score(
    lpips_vals: NDArray[np.floating[Any]],
    niqe_vals: NDArray[np.floating[Any]],
    tau_N: float = 5.0,
    alpha: float = 0.5,
) -> NDArray[np.floating[Any]]:
    perc = 1.0 - np.clip(lpips_vals, 0.0, 1.0)
    niqe_star = np.clip((tau_N - niqe_vals) / tau_N, 0.0, 1.0)
    return cast(NDArray[np.floating[Any]], alpha * perc + (1.0 - alpha) * niqe_star)

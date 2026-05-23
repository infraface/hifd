"""Privacy scoring from face-recognition embeddings."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


def privacy_score(emb_orig: ArrayLike, emb_deid: ArrayLike) -> float:
    """Single-backbone, single-sample privacy:  P = 0.5 * (1 - cos_sim)."""
    e_o = np.asarray(emb_orig, dtype=np.float64)
    e_d = np.asarray(emb_deid, dtype=np.float64)
    e_o = e_o / (np.linalg.norm(e_o) + 1e-12)
    e_d = e_d / (np.linalg.norm(e_d) + 1e-12)
    cos_sim = float(np.clip(np.dot(e_o, e_d), -1.0, 1.0))
    return 0.5 * (1.0 - cos_sim)


def batch_privacy_score(
    emb_orig: NDArray[np.floating[Any]], emb_deid: NDArray[np.floating[Any]]
) -> NDArray[np.floating[Any]]:
    """Vectorized over N samples. Inputs shape (N, D)."""
    norm_o = np.linalg.norm(emb_orig, axis=1, keepdims=True) + 1e-12
    norm_d = np.linalg.norm(emb_deid, axis=1, keepdims=True) + 1e-12
    e_o = emb_orig / norm_o
    e_d = emb_deid / norm_d
    cos_sim = np.clip(np.sum(e_o * e_d, axis=1), -1.0, 1.0)
    return cast(NDArray[np.floating[Any]], 0.5 * (1.0 - cos_sim))


def privacy_ensemble(
    backbone_scores: dict[str, NDArray[np.floating[Any]]],
) -> NDArray[np.floating[Any]]:
    """Average per-sample privacy across recognition backbones."""
    stacked = np.stack(list(backbone_scores.values()), axis=0)  # (M, N)
    return cast(NDArray[np.floating[Any]], np.mean(stacked, axis=0))

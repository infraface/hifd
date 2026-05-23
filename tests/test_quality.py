"""Tests for hifd.scores.quality."""

import numpy as np
import pytest

from hifd.scores import quality as qu


def test_quality_score_perfect():
    # LPIPS=0 → perceptual term 1; NIQE=0 → niqe_star 1 → Q=1.
    assert qu.quality_score(lpips=0.0, niqe=0.0, tau_N=5.0, alpha=0.5) == pytest.approx(1.0)


def test_quality_score_floor():
    # LPIPS=1, NIQE>=tau_N → Q=0.
    assert qu.quality_score(lpips=1.0, niqe=5.0, tau_N=5.0, alpha=0.5) == pytest.approx(0.0)


def test_quality_score_alpha_zero_is_niqe_only():
    # alpha=0 → ignore LPIPS, niqe=2.5/5 → niqe_star=0.5.
    assert qu.quality_score(lpips=1.0, niqe=2.5, tau_N=5.0, alpha=0.0) == pytest.approx(0.5)


def test_quality_score_alpha_one_is_lpips_only():
    # alpha=1 → ignore NIQE, lpips=0.3 → perc=0.7.
    assert qu.quality_score(lpips=0.3, niqe=99.0, tau_N=5.0, alpha=1.0) == pytest.approx(0.7)


def test_quality_score_clamps_lpips_above_one():
    assert qu.quality_score(lpips=2.0, niqe=0.0, tau_N=5.0, alpha=0.5) == pytest.approx(0.5)


def test_batch_quality_score_matches_scalar():
    lp = np.array([0.0, 0.5, 1.0])
    nq = np.array([0.0, 2.5, 5.0])
    batch = qu.batch_quality_score(lp, nq, tau_N=5.0, alpha=0.5)
    scalar = np.array([
        qu.quality_score(float(lpips_val), float(niqe_val), tau_N=5.0, alpha=0.5)
        for lpips_val, niqe_val in zip(lp, nq, strict=False)
    ])
    np.testing.assert_allclose(batch, scalar)

"""Tests for hifd.scores.agreement."""

import numpy as np
import pytest

from hifd.scores import agreement as ag

# ----- score_age --------------------------------------------------------------

def test_score_age_identity():
    assert ag.score_age(30.0, 30.0) == 1.0


def test_score_age_max_disagreement_floor():
    # Larger than Delta_age → clamped at 0.
    assert ag.score_age(0.0, 200.0, Delta_age=100.0) == 0.0


def test_score_age_linear():
    assert ag.score_age(30.0, 40.0, Delta_age=100.0) == pytest.approx(0.9)


def test_score_age_batch_matches_scalar():
    a = np.array([30.0, 0.0, 50.0])
    b = np.array([30.0, 200.0, 60.0])
    batch = ag.batch_score_age(a, b, Delta_age=100.0)
    scalar = np.array([ag.score_age(x, y, Delta_age=100.0) for x, y in zip(a, b, strict=False)])
    np.testing.assert_allclose(batch, scalar)


# ----- score_categorical ------------------------------------------------------

def test_score_categorical_identity():
    p = np.array([0.2, 0.5, 0.3])
    assert ag.score_categorical(p, p) == pytest.approx(1.0)


def test_score_categorical_disjoint():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert ag.score_categorical(a, b) == pytest.approx(0.0)


def test_score_categorical_half_overlap():
    a = np.array([1.0, 0.0])
    b = np.array([0.5, 0.5])
    # TV = 0.5 * (|0.5| + |-0.5|) = 0.5 → score 0.5
    assert ag.score_categorical(a, b) == pytest.approx(0.5)


def test_score_categorical_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        ag.score_categorical(np.array([0.5, 0.5]), np.array([0.3, 0.3, 0.4]))


# ----- score_landmark ---------------------------------------------------------

def test_score_landmark_identity():
    pts = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
    left_eye = np.array([0.0, 0.0])
    right_eye = np.array([10.0, 0.0])
    assert ag.score_landmark(pts, pts, left_eye, right_eye, tau_NME=0.10) == 1.0


def test_score_landmark_zero_inter_ocular_returns_zero():
    pts = np.array([[10.0, 20.0]])
    eye = np.array([5.0, 5.0])
    assert ag.score_landmark(pts, pts, eye, eye, tau_NME=0.10) == 0.0


def test_score_landmark_above_threshold_clamps_to_zero():
    pts_o = np.array([[0.0, 0.0]])
    pts_d = np.array([[100.0, 0.0]])
    left_eye = np.array([0.0, 0.0])
    right_eye = np.array([10.0, 0.0])
    # NME = 100 / 10 = 10, threshold 0.10 → clamped to 0.
    assert ag.score_landmark(pts_o, pts_d, left_eye, right_eye, tau_NME=0.10) == 0.0


# ----- score_gaze -------------------------------------------------------------

def test_score_gaze_identity():
    g = np.array([0.0, 0.0, 1.0])
    assert ag.score_gaze(g, g) == pytest.approx(1.0)


def test_score_gaze_opposite():
    g1 = np.array([0.0, 0.0, 1.0])
    g2 = np.array([0.0, 0.0, -1.0])
    # 180 degrees, theta_max_deg 90 → clamped at 0.
    assert ag.score_gaze(g1, g2, theta_max_deg=90.0) == 0.0


def test_score_gaze_perpendicular():
    g1 = np.array([0.0, 0.0, 1.0])
    g2 = np.array([1.0, 0.0, 0.0])
    # 90 degrees / 90 = 1.0 → score 0.0.
    assert ag.score_gaze(g1, g2, theta_max_deg=90.0) == pytest.approx(0.0, abs=1e-9)


# ----- score_bvp --------------------------------------------------------------

def test_score_bvp_identity():
    rng = np.random.default_rng(0)
    s = rng.normal(size=200)
    assert ag.score_bvp(s, s) == pytest.approx(1.0)


def test_score_bvp_flat_returns_zero():
    s = np.zeros(100)
    rng = np.random.default_rng(0)
    other = rng.normal(size=100)
    assert ag.score_bvp(s, other) == 0.0


def test_score_bvp_anticorrelated_clamps_to_zero():
    rng = np.random.default_rng(0)
    s = rng.normal(size=200)
    assert ag.score_bvp(s, -s) == 0.0


def test_score_bvp_length_mismatch_truncates():
    rng = np.random.default_rng(0)
    s = rng.normal(size=200)
    longer = np.concatenate([s, rng.normal(size=20)])
    assert ag.score_bvp(s, longer) == pytest.approx(1.0)


# ----- score_hr ---------------------------------------------------------------

def test_score_hr_identity():
    assert ag.score_hr(72.0, 72.0) == 1.0


def test_score_hr_linear():
    assert ag.score_hr(72.0, 82.0, tau_HR=20.0) == pytest.approx(0.5)


def test_score_hr_above_threshold_clamps_to_zero():
    assert ag.score_hr(72.0, 200.0, tau_HR=20.0) == 0.0


def test_score_hr_batch_matches_scalar():
    a = np.array([72.0, 60.0])
    b = np.array([72.0, 200.0])
    batch = ag.batch_score_hr(a, b, tau_HR=20.0)
    scalar = np.array([ag.score_hr(x, y, tau_HR=20.0) for x, y in zip(a, b, strict=False)])
    np.testing.assert_allclose(batch, scalar)

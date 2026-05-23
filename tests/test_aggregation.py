"""Tests for hifd.aggregation."""

import numpy as np
import pytest

from hifd import aggregation as agg


def test_U1_simple_mean():
    assert agg.U1(0.1, 0.2, 0.3, 0.4, 0.5) == pytest.approx(0.3)


def test_U2_with_microexp():
    assert agg.U2(0.8, s_me=0.6) == pytest.approx(0.7)


def test_U2_without_microexp_falls_back_to_gaze():
    assert agg.U2(0.8, s_me=None) == pytest.approx(0.8)


def test_U3_mean_of_per_estimator_combined():
    bvp = [0.6, 0.8]
    hr  = [0.4, 0.2]
    # per-est: (0.6+0.4)/2=0.5, (0.8+0.2)/2=0.5 → mean=0.5
    assert agg.U3(bvp, hr) == pytest.approx(0.5)


def test_aggregate_method_scores_columns():
    per_sample = {
        "age": np.array([0.9, 0.95]),
        "gender": np.array([0.8, 0.85]),
        "P": np.array([0.7, 0.75]),
        "Q": np.array([0.5, 0.55]),
        "U1": np.array([0.85, 0.9]),
    }
    df = agg.aggregate_method_scores(per_sample, method_name="MethodA")
    assert set(df.columns) == {"method", "dimension", "sub_score", "value", "n_samples", "std"}
    assert (df["method"] == "MethodA").all()
    # Dimension assignment
    row_age = df[df["sub_score"] == "age"].iloc[0]
    assert row_age["dimension"] == "L1"
    row_p = df[df["sub_score"] == "P"].iloc[0]
    assert row_p["dimension"] == "Privacy"


def test_batch_U2_handles_no_microexp():
    gaze = np.array([0.4, 0.6])
    np.testing.assert_allclose(agg.batch_U2(gaze, s_me=None), gaze)


def test_batch_U2_with_microexp():
    gaze = np.array([0.4, 0.6])
    me = np.array([0.6, 0.4])
    np.testing.assert_allclose(agg.batch_U2(gaze, me), np.array([0.5, 0.5]))

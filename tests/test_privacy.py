"""Tests for hifd.scores.privacy."""

import numpy as np
import pytest

from hifd.scores import privacy as priv


def test_privacy_score_identical_embeddings_is_zero():
    e = np.array([1.0, 0.0, 0.0])
    assert priv.privacy_score(e, e) == pytest.approx(0.0, abs=1e-9)


def test_privacy_score_antipodal_embeddings_is_one():
    e1 = np.array([1.0, 0.0, 0.0])
    e2 = np.array([-1.0, 0.0, 0.0])
    assert priv.privacy_score(e1, e2) == pytest.approx(1.0)


def test_privacy_score_orthogonal_is_half():
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    assert priv.privacy_score(e1, e2) == pytest.approx(0.5)


def test_privacy_score_handles_unnormalized_inputs():
    # Cosine similarity is scale-invariant; magnitudes should not matter.
    assert priv.privacy_score(np.array([2.0, 0.0]), np.array([5.0, 0.0])) == pytest.approx(0.0)


def test_batch_privacy_score_matches_scalar():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(10, 8))
    b = rng.normal(size=(10, 8))
    batch = priv.batch_privacy_score(a, b)
    scalar = np.array([priv.privacy_score(a[i], b[i]) for i in range(10)])
    np.testing.assert_allclose(batch, scalar)


def test_privacy_ensemble_averages_across_backbones():
    scores = {
        "arcface": np.array([0.2, 0.4]),
        "cosface": np.array([0.4, 0.6]),
        "adaface": np.array([0.6, 0.8]),
    }
    out = priv.privacy_ensemble(scores)
    np.testing.assert_allclose(out, np.array([0.4, 0.6]))

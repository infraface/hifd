"""Tests for hifd.io."""

import json
from pathlib import Path

import numpy as np
import pytest

from hifd import io as hio

# ----- Helpers ----------------------------------------------------------------

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def envelope(task: str, source: str, predictions: dict, **extra) -> dict:
    base = {
        "task": task,
        "source": source,
        "estimator": "tester",
        "schema_version": "1.0",
        "predictions": predictions,
    }
    base.update(extra)
    return base


# ----- Envelope validation ----------------------------------------------------

def test_validate_pair_age_ok(tmp_path):
    o = envelope("age", "original", {"s1": {"value": 30.0}})
    d = envelope("age", "deid", {"s1": {"value": 31.0}})
    op = tmp_path / "orig.json"
    write_json(op, o)
    dp = tmp_path / "deid.json"
    write_json(dp, d)
    task, common = hio.validate_pair(str(op), str(dp))
    assert task == "age"
    assert common == ["s1"]


def test_validate_pair_schema_version_mismatch_raises(tmp_path):
    o = envelope("age", "original", {"s1": {"value": 30.0}})
    o["schema_version"] = "0.9"
    d = envelope("age", "deid", {"s1": {"value": 30.0}})
    op = tmp_path / "o.json"
    write_json(op, o)
    dp = tmp_path / "d.json"
    write_json(dp, d)
    with pytest.raises(ValueError, match="schema_version"):
        hio.validate_pair(str(op), str(dp))


def test_validate_pair_source_mismatch_raises(tmp_path):
    o = envelope("age", "deid", {"s1": {"value": 30.0}})  # wrong source
    d = envelope("age", "deid", {"s1": {"value": 30.0}})
    op = tmp_path / "o.json"
    write_json(op, o)
    dp = tmp_path / "d.json"
    write_json(dp, d)
    with pytest.raises(ValueError, match="source"):
        hio.validate_pair(str(op), str(dp))


def test_validate_pair_categorical_class_names_mismatch_raises(tmp_path):
    o = envelope(
        "gender", "original", {"s1": {"probs": [0.5, 0.5]}},
        class_names=["male", "female"],
    )
    d = envelope(
        "gender", "deid", {"s1": {"probs": [0.5, 0.5]}},
        class_names=["m", "f"],
    )
    op = tmp_path / "o.json"
    write_json(op, o)
    dp = tmp_path / "d.json"
    write_json(dp, d)
    with pytest.raises(ValueError, match="class_names"):
        hio.validate_pair(str(op), str(dp))


def test_validate_pair_landmark_point_count_mismatch_raises(tmp_path):
    o = envelope("landmark", "original", {
        "s1": {"points": [[0, 0], [1, 1]], "left_eye_center": [0, 0], "right_eye_center": [1, 0]},
    })
    d = envelope("landmark", "deid", {
        "s1": {"points": [[0, 0]], "left_eye_center": [0, 0], "right_eye_center": [1, 0]},
    })
    op = tmp_path / "o.json"
    write_json(op, o)
    dp = tmp_path / "d.json"
    write_json(dp, d)
    with pytest.raises(ValueError, match="point count"):
        hio.validate_pair(str(op), str(dp))


def test_validate_pair_rppg_fps_mismatch_raises(tmp_path):
    o = envelope("rppg", "original", {"s1": {"bvp": [0.0] * 60, "hr_bpm": 60.0, "fps": 30}})
    d = envelope("rppg", "deid", {"s1": {"bvp": [0.0] * 60, "hr_bpm": 60.0, "fps": 60}})
    op = tmp_path / "o.json"
    write_json(op, o)
    dp = tmp_path / "d.json"
    write_json(dp, d)
    with pytest.raises(ValueError, match="fps"):
        hio.validate_pair(str(op), str(dp))


def test_validate_pair_no_common_ids_raises(tmp_path):
    o = envelope("age", "original", {"s1": {"value": 30.0}})
    d = envelope("age", "deid", {"s2": {"value": 30.0}})
    op = tmp_path / "o.json"
    write_json(op, o)
    dp = tmp_path / "d.json"
    write_json(dp, d)
    with pytest.raises(ValueError, match="common"):
        hio.validate_pair(str(op), str(dp))


# ----- Extractors -------------------------------------------------------------

def test_extract_age():
    preds = {"s1": {"value": 32.5}}
    assert hio.extract_age(preds, "s1") == 32.5


def test_extract_probs_renormalizes(caplog):
    import logging
    caplog.set_level(logging.WARNING)
    preds = {"s1": {"probs": [0.4, 0.4]}}
    out = hio.extract_probs(preds, "s1")
    np.testing.assert_allclose(out, np.array([0.5, 0.5]))
    assert "renormalizing" in caplog.text


def test_extract_landmark():
    preds = {"s1": {
        "points": [[0.0, 0.0], [1.0, 1.0]],
        "left_eye_center": [0.0, 0.0],
        "right_eye_center": [1.0, 0.0],
    }}
    pts, le, re = hio.extract_landmark(preds, "s1")
    np.testing.assert_allclose(pts, np.array([[0.0, 0.0], [1.0, 1.0]]))
    np.testing.assert_allclose(le, np.array([0.0, 0.0]))
    np.testing.assert_allclose(re, np.array([1.0, 0.0]))


def test_extract_gaze():
    preds = {"s1": {"direction": [0.0, 0.0, 1.0]}}
    np.testing.assert_allclose(hio.extract_gaze(preds, "s1"), np.array([0.0, 0.0, 1.0]))


def test_extract_rppg():
    preds = {"s1": {"bvp": [1, 2, 3], "hr_bpm": 72.0, "fps": 30}}
    bvp, hr, fps = hio.extract_rppg(preds, "s1")
    np.testing.assert_allclose(bvp, np.array([1.0, 2.0, 3.0]))
    assert hr == 72.0
    assert fps == 30


def test_extract_embedding():
    preds = {"s1": {"embedding": [1.0, 0.0, 0.0]}}
    np.testing.assert_allclose(hio.extract_embedding(preds, "s1"), np.array([1.0, 0.0, 0.0]))


def test_extract_niqe():
    preds = {"s1": {"niqe": 3.5}}
    assert hio.extract_niqe(preds, "s1") == 3.5


# ----- load_predictions / load_pair ------------------------------------------

def test_load_predictions(tmp_path):
    o = envelope("age", "original", {"s1": {"value": 30.0}})
    p = tmp_path / "o.json"
    write_json(p, o)
    data = hio.load_predictions(str(p))
    assert data["task"] == "age"
    assert "s1" in data["predictions"]


def test_load_pair(tmp_path):
    o = envelope("age", "original", {"s1": {"value": 30.0}})
    d = envelope("age", "deid", {"s1": {"value": 31.0}, "s2": {"value": 40.0}})
    op = tmp_path / "o.json"
    write_json(op, o)
    dp = tmp_path / "d.json"
    write_json(dp, d)
    orig_p, deid_p, common = hio.load_pair(str(op), str(dp))
    assert common == ["s1"]
    assert orig_p["s1"]["value"] == 30.0
    assert deid_p["s2"]["value"] == 40.0

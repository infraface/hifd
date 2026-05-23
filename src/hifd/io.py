"""JSON-envelope IO and schema validation."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from .constants import SCHEMA_VERSION

logger = logging.getLogger(__name__)

TASK_TYPES: frozenset[str] = frozenset({
    "age", "gender", "ethnicity", "macro_exp", "micro_exp",
    "landmark", "gaze", "rppg", "face_embedding", "niqe",
})

CATEGORICAL_TASKS: frozenset[str] = frozenset({"gender", "ethnicity", "macro_exp", "micro_exp"})


# ----- Low-level --------------------------------------------------------------

def load_predictions(path: str) -> dict[str, Any]:
    """Load a single JSON envelope file (no cross-file validation)."""
    with open(path) as f:
        data: dict[str, Any] = json.load(f)
    return data


def _validate_envelope(data: dict[str, Any], expected_task: str | None = None) -> str:
    required = {"task", "source", "estimator", "schema_version", "predictions"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"missing envelope fields: {sorted(missing)}")
    if data["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: expected {SCHEMA_VERSION}, got {data['schema_version']}"
        )
    task = data["task"]
    if task not in TASK_TYPES:
        raise ValueError(f"unknown task: {task}")
    if expected_task is not None and task != expected_task:
        raise ValueError(f"expected task '{expected_task}', got '{task}'")
    return str(task)


def _validate_categorical_pair(orig: dict[str, Any], deid: dict[str, Any]) -> None:
    on = orig.get("class_names")
    dn = deid.get("class_names")
    if on is None or dn is None:
        raise ValueError("categorical task: missing 'class_names' at envelope level")
    if on != dn:
        raise ValueError(f"class_names mismatch: orig={on}, deid={dn}")
    n_classes = len(on)
    for sid, pred in list(orig["predictions"].items())[:5]:
        if len(pred["probs"]) != n_classes:
            raise ValueError(
                f"sample {sid}: probs length {len(pred['probs'])} != class_names length {n_classes}"
            )


def _validate_landmark_pair(orig: dict[str, Any], deid: dict[str, Any]) -> None:
    for sid in list(orig["predictions"].keys())[:5]:
        if sid not in deid["predictions"]:
            continue
        if len(orig["predictions"][sid]["points"]) != len(deid["predictions"][sid]["points"]):
            raise ValueError(
                f"sample {sid}: landmark point count mismatch"
            )


def _validate_rppg_pair(
    orig: dict[str, Any], deid: dict[str, Any], tol_seconds: float = 1.0
) -> None:
    for sid in list(orig["predictions"].keys())[:5]:
        if sid not in deid["predictions"]:
            continue
        op = orig["predictions"][sid]
        dp = deid["predictions"][sid]
        fps_o = op.get("fps", orig.get("fps"))
        fps_d = dp.get("fps", deid.get("fps"))
        if fps_o != fps_d:
            raise ValueError(f"sample {sid}: fps mismatch: orig={fps_o}, deid={fps_d}")
        len_o = len(op["bvp"])
        len_d = len(dp["bvp"])
        fps = fps_o if fps_o else 30
        if abs(len_o - len_d) > fps * tol_seconds:
            raise ValueError(
                f"sample {sid}: bvp length diff {abs(len_o - len_d)} exceeds "
                f"{tol_seconds}s tolerance at {fps} fps"
            )


def validate_pair(orig_path: str, deid_path: str) -> tuple[str, list[str]]:
    """Validate orig + deid JSON pair. Returns (task, common_sample_ids)."""
    orig = load_predictions(orig_path)
    deid = load_predictions(deid_path)

    task_o = _validate_envelope(orig)
    _validate_envelope(deid, expected_task=task_o)

    if orig["source"] != "original":
        raise ValueError(f"{orig_path}: expected source='original', got '{orig['source']}'")
    if deid["source"] != "deid":
        raise ValueError(f"{deid_path}: expected source='deid', got '{deid['source']}'")

    common = sorted(set(orig["predictions"].keys()) & set(deid["predictions"].keys()))
    if not common:
        raise ValueError("no common sample IDs between orig and deid")

    if task_o in CATEGORICAL_TASKS:
        _validate_categorical_pair(orig, deid)
    elif task_o == "landmark":
        _validate_landmark_pair(orig, deid)
    elif task_o == "rppg":
        _validate_rppg_pair(orig, deid)

    return task_o, common


def load_pair(
    orig_path: str, deid_path: str, validate: bool = True
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    """Load orig + deid JSON, optionally validating, return (orig_preds, deid_preds, common_ids)."""
    if validate:
        _, common = validate_pair(orig_path, deid_path)
    else:
        o = load_predictions(orig_path)
        d = load_predictions(deid_path)
        common = sorted(set(o["predictions"].keys()) & set(d["predictions"].keys()))
    orig = load_predictions(orig_path)
    deid = load_predictions(deid_path)
    orig_preds: dict[str, Any] = orig["predictions"]
    deid_preds: dict[str, Any] = deid["predictions"]
    return orig_preds, deid_preds, common


# ----- Extractors -------------------------------------------------------------

def extract_age(predictions: dict[str, Any], sample_id: str) -> float:
    return float(predictions[sample_id]["value"])


def extract_probs(predictions: dict[str, Any], sample_id: str) -> NDArray[np.float64]:
    probs = np.array(predictions[sample_id]["probs"], dtype=np.float64)
    total = float(probs.sum())
    if abs(total - 1.0) > 1e-4:
        logger.warning(f"sample {sample_id}: probs sum={total:.6f}, renormalizing")
        probs = probs / total
    return probs


def extract_landmark(
    predictions: dict[str, Any], sample_id: str
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    p = predictions[sample_id]
    return (
        np.array(p["points"], dtype=np.float64),
        np.array(p["left_eye_center"], dtype=np.float64),
        np.array(p["right_eye_center"], dtype=np.float64),
    )


def extract_gaze(predictions: dict[str, Any], sample_id: str) -> NDArray[np.float64]:
    return np.array(predictions[sample_id]["direction"], dtype=np.float64)


def extract_rppg(
    predictions: dict[str, Any], sample_id: str
) -> tuple[NDArray[np.float64], float, int]:
    p = predictions[sample_id]
    return (
        np.array(p["bvp"], dtype=np.float64),
        float(p["hr_bpm"]),
        int(p.get("fps", 30)),
    )


def extract_embedding(predictions: dict[str, Any], sample_id: str) -> NDArray[np.float64]:
    return np.array(predictions[sample_id]["embedding"], dtype=np.float64)


def extract_niqe(predictions: dict[str, Any], sample_id: str) -> float:
    return float(predictions[sample_id]["niqe"])


# ----- YAML helpers (for CLI --constants / --profiles) -----------------------

def load_constants_yaml(path: str) -> dict[str, float]:
    """Read a YAML file of constants. Validation against DEFAULT_CONSTANTS happens upstream."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected top-level mapping")
    return {str(k): float(v) for k, v in raw.items()}


def load_methods_yaml(path: str) -> dict[str, Any]:
    """Read methods.yaml: {original: {base_path: ...}, methods: [{name, base_path}, ...]}."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected top-level mapping")
    if "original" not in raw or "methods" not in raw:
        raise ValueError(f"{path}: missing 'original' or 'methods' key")
    result: dict[str, Any] = raw
    return result


def load_estimators_yaml(path: str) -> dict[str, Any]:
    """Read estimators.yaml: {tasks: {<task>: {estimators: [...]}, ...}}."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected top-level mapping")
    if "tasks" not in raw:
        raise ValueError(f"{path}: missing top-level 'tasks' key")
    result: dict[str, Any] = raw
    return result

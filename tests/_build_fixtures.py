"""One-off script to generate the mini fixture set. Run once; commit the output.

Run with:  python -m tests._build_fixtures   (from hifd/ root)
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).parent / "fixtures" / "mini"
SAMPLES = ["s1", "s2", "s3"]


def envelope(task, source, estimator, predictions, **extra):
    base = {
        "task": task, "source": source, "estimator": estimator,
        "schema_version": "1.0", "predictions": predictions,
    }
    base.update(extra)
    return base


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def build_for(root_path, filename, source_field, vary):
    """vary=0 → identical to orig; vary>0 introduces per-task drift.

    Args:
        root_path: Base directory for outputs.
        filename: Filename prefix (e.g., "orig" or "deid").
        source_field: JSON envelope source field value (e.g., "original" or "deid").
        vary: Variation factor for drift.
    """
    # age
    write(root_path / "age" / f"{filename}.json", envelope(
        "age", source_field, "mivolo",
        {sid: {"value": 30.0 + i + vary * 5} for i, sid in enumerate(SAMPLES)},
    ))
    # gender
    write(root_path / "gender" / f"{filename}.json", envelope(
        "gender", source_field, "mivolo",
        {sid: {"probs": [1.0 - vary * 0.2, vary * 0.2]} for sid in SAMPLES},
        class_names=["male", "female"],
    ))
    # ethnicity
    write(root_path / "ethnicity" / f"{filename}.json", envelope(
        "ethnicity", source_field, "fairface",
        {sid: {"probs": [1.0 - vary * 0.1, vary * 0.1, 0.0]} for sid in SAMPLES},
        class_names=["white", "black", "asian"],
    ))
    # macro_exp
    write(root_path / "macro_exp" / f"{filename}.json", envelope(
        "macro_exp", source_field, "poster",
        {sid: {"probs": [1.0 - vary * 0.1, vary * 0.1]} for sid in SAMPLES},
        class_names=["neutral", "happy"],
    ))
    # micro_exp
    write(root_path / "micro_exp" / f"{filename}.json", envelope(
        "micro_exp", source_field, "samer",
        {sid: {"probs": [1.0 - vary * 0.1, vary * 0.1]} for sid in SAMPLES},
        class_names=["calm", "tense"],
    ))
    # landmark
    write(root_path / "landmark" / f"{filename}.json", envelope(
        "landmark", source_field, "mediapipe",
        {sid: {
            "points": [[10.0 + vary, 20.0], [30.0, 40.0 + vary]],
            "left_eye_center": [0.0, 0.0],
            "right_eye_center": [10.0, 0.0],
        } for sid in SAMPLES},
    ))
    # gaze
    write(root_path / "gaze" / f"{filename}.json", envelope(
        "gaze", source_field, "l2cs",
        {sid: {"direction": [0.0, vary * 0.1, 1.0]} for sid in SAMPLES},
    ))
    # rppg
    write(root_path / "rppg" / f"{filename}.json", envelope(
        "rppg", source_field, "rppg_estimator",
        {sid: {
            "bvp": [float(i + vary * 0.01) for i in range(60)],
            "hr_bpm": 72.0 + vary * 5,
            "fps": 30,
        } for sid in SAMPLES},
        fps=30,
    ))
    # face_embedding — three backbones
    for bb in ("arcface", "cosface", "adaface"):
        write(root_path / "face_embedding" / f"{bb}_{filename}.json", envelope(
            "face_embedding", source_field, bb,
            {sid: {"embedding": [1.0 - vary * 0.1, vary * 0.1, 0.0]} for sid in SAMPLES},
        ))
    # niqe — de-id only
    if source_field == "deid":
        write(root_path / "niqe" / f"{filename}.json", envelope(
            "niqe", source_field, "pyiqa",
            {sid: {"niqe": 2.5 + vary * 0.5} for sid in SAMPLES},
        ))


def main():
    build_for(ROOT / "utilface", filename="orig", source_field="original", vary=0)
    build_for(ROOT / "methods" / "methodA", filename="deid", source_field="deid", vary=1)
    build_for(ROOT / "methods" / "methodB", filename="deid", source_field="deid", vary=2)

    # methods.yaml + estimators.yaml
    (ROOT / "methods.yaml").write_text(
        "original:\n"
        "  base_path: utilface\n"
        "methods:\n"
        "  - name: methodA\n"
        "    base_path: methods/methodA\n"
        "  - name: methodB\n"
        "    base_path: methods/methodB\n"
    )
    fixture_file = Path(__file__).parent / "_estimators_fixture.txt"
    (ROOT / "estimators.yaml").write_text(fixture_file.read_text())

    print("Fixtures written to", ROOT)


if __name__ == "__main__":
    main()

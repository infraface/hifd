"""High-level orchestrator: compute -> aggregate -> compose."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from . import __version__
from . import aggregation as agg
from . import io as hio
from .composite import compose_profiles
from .constants import DEFAULT_CONSTANTS, DEFAULT_PROFILES
from .scores import agreement as ag
from .scores import privacy as priv

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    per_sample_df: pd.DataFrame
    per_method_df: pd.DataFrame
    composite_df: pd.DataFrame
    output_dir: Path


# ----- CSV header helper ------------------------------------------------------

def _write_csv_with_header(
    df: pd.DataFrame, path: Path, constants: Mapping[str, float] | None
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header_lines = [
        f"# Generated: {datetime.now().isoformat()}",
        f"# Package: hifd=={__version__}",
        "# Seed: 42",
    ]
    if constants:
        header_lines.append(f"# Constants: {dict(constants)}")
    with open(path, "w") as f:
        for line in header_lines:
            f.write(line + "\n")
        df.to_csv(f, index=False)


def _read_csv_with_header(path: Path) -> pd.DataFrame:
    with open(path) as f:
        skip = 0
        for line in f:
            if line.startswith("#"):
                skip += 1
            else:
                break
    # pandas read_csv returns Any-typed under strict mypy
    result: pd.DataFrame = pd.read_csv(path, skiprows=skip)
    return result


# ----- Stage 1: compute per-sample agreements --------------------------------

def _compute_agreements_for_method(
    method_name: str,
    orig_base: Path,
    method_base: Path,
    estimators_cfg: dict[str, Any],
    constants: Mapping[str, float],
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    tasks = estimators_cfg["tasks"]

    # age
    for est in tasks.get("age", {}).get("estimators", []):
        op = orig_base / est["orig_json"]
        dp = method_base / est["deid_json"]
        if not (op.exists() and dp.exists()):
            continue
        op_d, dp_d, common = hio.load_pair(str(op), str(dp))
        for sid in common:
            records.append({
                "method": method_name, "sample_id": sid, "task": "age",
                "estimator": est["name"],
                "s_k": ag.score_age(
                    hio.extract_age(op_d, sid), hio.extract_age(dp_d, sid),
                    constants["Delta_age"],
                ),
            })

    # categorical
    for task in ("gender", "ethnicity", "macro_exp", "micro_exp"):
        for est in tasks.get(task, {}).get("estimators", []):
            op = orig_base / est["orig_json"]
            dp = method_base / est["deid_json"]
            if not (op.exists() and dp.exists()):
                continue
            op_d, dp_d, common = hio.load_pair(str(op), str(dp))
            for sid in common:
                records.append({
                    "method": method_name, "sample_id": sid, "task": task,
                    "estimator": est["name"],
                    "s_k": ag.score_categorical(
                        hio.extract_probs(op_d, sid), hio.extract_probs(dp_d, sid),
                    ),
                })

    # landmark
    for est in tasks.get("landmark", {}).get("estimators", []):
        op = orig_base / est["orig_json"]
        dp = method_base / est["deid_json"]
        if not (op.exists() and dp.exists()):
            continue
        op_d, dp_d, common = hio.load_pair(str(op), str(dp))
        for sid in common:
            L_o, le, re_ = hio.extract_landmark(op_d, sid)
            L_d, _, _ = hio.extract_landmark(dp_d, sid)
            records.append({
                "method": method_name, "sample_id": sid, "task": "landmark",
                "estimator": est["name"],
                "s_k": ag.score_landmark(L_o, L_d, le, re_, constants["tau_NME"]),
            })

    # gaze
    for est in tasks.get("gaze", {}).get("estimators", []):
        op = orig_base / est["orig_json"]
        dp = method_base / est["deid_json"]
        if not (op.exists() and dp.exists()):
            continue
        op_d, dp_d, common = hio.load_pair(str(op), str(dp))
        for sid in common:
            records.append({
                "method": method_name, "sample_id": sid, "task": "gaze",
                "estimator": est["name"],
                "s_k": ag.score_gaze(
                    hio.extract_gaze(op_d, sid), hio.extract_gaze(dp_d, sid),
                    constants["theta_max_deg"],
                ),
            })

    # rppg -> emits bvp and hr rows
    for est in tasks.get("rppg", {}).get("estimators", []):
        op = orig_base / est["orig_json"]
        dp = method_base / est["deid_json"]
        if not (op.exists() and dp.exists()):
            continue
        op_d, dp_d, common = hio.load_pair(str(op), str(dp))
        for sid in common:
            bvp_o, hr_o, _ = hio.extract_rppg(op_d, sid)
            bvp_d, hr_d, _ = hio.extract_rppg(dp_d, sid)
            records.append({
                "method": method_name, "sample_id": sid, "task": "bvp",
                "estimator": est["name"], "s_k": ag.score_bvp(bvp_o, bvp_d),
            })
            records.append({
                "method": method_name, "sample_id": sid, "task": "hr",
                "estimator": est["name"],
                "s_k": ag.score_hr(hr_o, hr_d, constants["tau_HR"]),
            })

    # face_embedding -> privacy
    for est in tasks.get("face_embedding", {}).get("estimators", []):
        op = orig_base / est["orig_json"]
        dp = method_base / est["deid_json"]
        if not (op.exists() and dp.exists()):
            continue
        op_d, dp_d, common = hio.load_pair(str(op), str(dp))
        for sid in common:
            records.append({
                "method": method_name, "sample_id": sid, "task": "privacy",
                "estimator": est["name"],
                "s_k": priv.privacy_score(
                    hio.extract_embedding(op_d, sid), hio.extract_embedding(dp_d, sid),
                ),
            })

    # niqe - deid only
    for est in tasks.get("niqe", {}).get("estimators", []):
        dp = method_base / est["deid_json"]
        if not dp.exists():
            continue
        deid = hio.load_predictions(str(dp))
        for sid, pred in deid["predictions"].items():
            records.append({
                "method": method_name, "sample_id": sid, "task": "niqe",
                "estimator": est["name"], "s_k": float(pred["niqe"]),
            })

    return pd.DataFrame(records)


def compute_agreements(
    methods_config: str,
    data_dir: str,
    out_dir: str,
    *,
    estimators_config: str | None = None,
    constants: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Stage 1: per-sample agreement scores for every (method, task, estimator, sample)."""
    methods_cfg = hio.load_methods_yaml(methods_config)
    estimators_cfg = (
        hio.load_estimators_yaml(estimators_config)
        if estimators_config else {"tasks": {}}
    )
    consts = dict(constants) if constants is not None else dict(DEFAULT_CONSTANTS)

    data = Path(data_dir)
    orig_base = data / methods_cfg["original"]["base_path"]

    dfs = []
    for m in methods_cfg["methods"]:
        logger.info(f"Computing agreements for: {m['name']}")
        dfs.append(_compute_agreements_for_method(
            m["name"], orig_base, data / m["base_path"], estimators_cfg, consts,
        ))

    combined = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    out_path = Path(out_dir) / "per_sample_agreements.csv"
    _write_csv_with_header(combined, out_path, consts)
    return combined


# ----- Stage 2: aggregate per-sample -> per-method ----------------------------

def aggregate(in_dir: str, out_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 2: per-method aggregated and method-level CSVs."""
    in_p = Path(in_dir)
    df = _read_csv_with_header(in_p / "per_sample_agreements.csv")
    methods = list(df["method"].unique())

    agg_rows = []
    for method in methods:
        mdf = df[df["method"] == method]
        for task in mdf["task"].unique():
            tdf = mdf[mdf["task"] == task]
            per_sample = tdf.groupby("sample_id")["s_k"].mean()
            agg_rows.append({
                "method": method, "sub_score": task,
                "value": float(per_sample.mean()),
                "std": float(per_sample.std()),
                "n_samples": len(per_sample),
            })
    agg_df = pd.DataFrame(agg_rows)

    method_rows = []
    for method in methods:
        scores = dict(zip(
            agg_df[agg_df["method"] == method]["sub_score"],
            agg_df[agg_df["method"] == method]["value"],
            strict=False,
        ))
        u1_keys = ["age", "gender", "ethnicity", "macro_exp", "landmark"]
        u1_vals: list[float] = [
            float(scores[k]) for k in u1_keys if scores.get(k) is not None
        ]
        u1: float | None = float(np.mean(u1_vals)) if u1_vals else None

        u2_gaze = scores.get("gaze")
        u2_me = scores.get("micro_exp")
        u2: float | None = agg.U2(u2_gaze, u2_me) if u2_gaze is not None else None

        bvp, hr = scores.get("bvp"), scores.get("hr")
        u3: float | None = (
            agg.U3([bvp], [hr]) if bvp is not None and hr is not None else None
        )

        method_rows.append({
            "method": method,
            "P": scores.get("privacy"),
            "Q": scores.get("niqe"),
            "U1": u1, "U2": u2, "U3": u3,
            **{k: scores.get(k) for k in (
                "age", "gender", "ethnicity", "macro_exp", "landmark",
                "gaze", "micro_exp", "bvp", "hr",
            )},
        })
    method_df = pd.DataFrame(method_rows)

    out_p = Path(out_dir)
    _write_csv_with_header(agg_df, out_p / "per_method_aggregated.csv", None)
    _write_csv_with_header(method_df, out_p / "method_level.csv", None)
    return agg_df, method_df


# ----- Stage 3: compose composite under profiles -----------------------------

def compose(
    in_dir: str,
    out_path: str,
    *,
    profiles: Mapping[str, Mapping[str, float]] | None = None,
    constants: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Stage 3: composite scores for each method under each profile."""
    in_p = Path(in_dir)
    method_df = _read_csv_with_header(in_p / "method_level.csv")
    profs = profiles if profiles is not None else DEFAULT_PROFILES
    consts = dict(constants) if constants is not None else dict(DEFAULT_CONSTANTS)
    eps = consts.get("epsilon", 1.0e-6)

    rows = []
    for _, row in method_df.iterrows():
        # row.get() returns Any under strict mypy; pd.notna() likewise.
        p_val = row.get("P") if pd.notna(row.get("P")) else None
        q_val = row.get("Q") if pd.notna(row.get("Q")) else None
        u1_val = row.get("U1") if pd.notna(row.get("U1")) else None
        u2_val = row.get("U2") if pd.notna(row.get("U2")) else None
        u3_val = row.get("U3") if pd.notna(row.get("U3")) else None
        per_profile = compose_profiles(
            p_val, q_val, u1_val, u2_val, u3_val,
            profiles=profs, epsilon=eps,
        )
        rows.append({
            "method": row["method"],
            "P": row.get("P"), "Q": row.get("Q"),
            "U1": row.get("U1"), "U2": row.get("U2"), "U3": row.get("U3"),
            **per_profile,
        })
    composite_df = pd.DataFrame(rows)

    out_p = Path(out_path)
    _write_csv_with_header(composite_df, out_p, consts)
    _export_latex(composite_df, out_p.with_suffix(".tex"))
    return composite_df


def _export_latex(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    profile_cols = [c for c in df.columns if c not in {"method", "P", "Q", "U1", "U2", "U3"}]
    cols = ["method", "P", "Q", "U1", "U2", "U3", *profile_cols]
    keep = [c for c in cols if c in df.columns]
    # to_latex returns Any under strict mypy
    latex: str = df[keep].to_latex(index=False, float_format="%.3f")
    out_path.write_text(latex)


# ----- run_pipeline (all stages) ---------------------------------------------

def run_pipeline(
    methods_config: str,
    data_dir: str,
    out_dir: str,
    *,
    estimators_config: str | None = None,
    constants: Mapping[str, float] | None = None,
    profiles: Mapping[str, Mapping[str, float]] | None = None,
) -> PipelineResult:
    """Run all three stages end-to-end. Returns DataFrames + output path."""
    out_p = Path(out_dir)
    per_sample = compute_agreements(
        methods_config, data_dir, str(out_p / "per_sample"),
        estimators_config=estimators_config, constants=constants,
    )
    _agg, per_method = aggregate(str(out_p / "per_sample"), str(out_p / "per_method"))
    composite_df = compose(
        str(out_p / "per_method"), str(out_p / "tables" / "composite.csv"),
        profiles=profiles, constants=constants,
    )
    return PipelineResult(
        per_sample_df=per_sample,
        per_method_df=per_method,
        composite_df=composite_df,
        output_dir=out_p,
    )

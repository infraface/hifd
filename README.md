# hifd

[![CI](https://github.com/infraface/hifd/actions/workflows/ci.yml/badge.svg)](https://github.com/infraface/hifd/actions/workflows/ci.yml)

**HiFD** is a holistic metric for evaluating face de-identification methods.
It combines privacy (face-recognition resistance), perceptual quality, and
three levels of utility preservation (macro cues, fine-grained cues,
physiological signals) into a single composite score.

This package provides the **scoring core**: given JSON files of pre-computed
estimator predictions on (a) original images and (b) de-identified outputs,
`hifd` produces per-sample agreement scores, per-method utility levels,
ensemble privacy, quality, and composite HiFD scores under three application
profiles (Privacy-First, Balanced, Clinical).

The package does **not** run the estimators themselves — you bring your own
predictions in the [documented JSON schema](docs/schema.md).

## Install

```bash
pip install hifd
```

Runtime requirements: Python ≥ 3.10, numpy, pandas, pyyaml.

## Quickstart (CLI)

```bash
hifd run-all \
    --methods    methods.yaml \
    --estimators estimators.yaml \
    --data-dir   predictions/ \
    --out        scores/
```

Produces:

```
scores/
├── per_sample/per_sample_agreements.csv
├── per_method/per_method_aggregated.csv
├── per_method/method_level.csv
├── tables/composite.csv
└── tables/composite.tex
```

## Quickstart (Python API)

```python
import hifd
import numpy as np

# Per-task primitives (scalar)
hifd.score_age(35.0, 36.0)                 # 0.99
hifd.score_gaze([0, 0, 1], [0, 0.1, 1])    # ≈ 0.937

# Batch variants for scalar-input primitives
hifd.batch_score_age(np.array([35.0]), np.array([36.0]))  # ndarray([0.99])
# Also: batch_score_categorical, batch_score_hr, batch_privacy_score,
# batch_quality_score, batch_U1, batch_U2, batch_U3
# (Geometric primitives — score_landmark, score_gaze, score_bvp — have no
# batch form because their inputs are per-sample variable-shape arrays.)

# Composite under a profile
hifd.composite(P=0.7, Q=0.8, U1=0.6, U2=0.5, U3=0.4,
               weights=hifd.DEFAULT_PROFILES["balanced"])

# Full pipeline
r = hifd.run_pipeline(
    methods_config="methods.yaml",
    estimators_config="estimators.yaml",
    data_dir="predictions/",
    out_dir="scores/",
)
print(r.composite_df)
```

## Input format

See [docs/schema.md](docs/schema.md) for the JSON envelope spec and one example
per task type.

## Defaults

Constants and profiles ship as immutable mappings:

```python
hifd.DEFAULT_CONSTANTS["tau_HR"]            # 20.0
hifd.DEFAULT_PROFILES["balanced"]           # {"P": 0.2, "Q": 0.2, "U1": 0.2, ...}
```

To override, build a plain dict and pass it (`run_pipeline(constants=...)`) or
provide `--constants my.yaml` on the CLI.

## Versioning and stability

`hifd` is `0.x`: the JSON schema and Python API may evolve until 1.0. Schema
version is asserted at load time; mismatched files are rejected with a clear
error.

## License

MIT.

## Citation

If you use HiFD in published work, please cite the paper [TBD].

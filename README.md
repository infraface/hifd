# HiFD metric

[![CI](https://github.com/infraface/hifd/actions/workflows/ci.yml/badge.svg)](https://github.com/infraface/hifd/actions/workflows/ci.yml)

**HiFD** is a metric for evaluating face de-identification methods.
It combines privacy (face-recognition resistance), perceptual quality, and
three levels of utility preservation (macro cues, micro cues,
physiological signals) into a single composite score.

This is the official toolkit of the NeurIPS 2026 Evaluations & Datasets
track paper:

> **How Private is Private? A Comparative Study for Face De-Identification**<br>
> Hui Wei, Hao Yu, Hui Kuurila-Zhang, Guoying Zhao<br>
> Center for Machine Vision and Signal Analysis, University of Oulu, Finland

| | |
|---|---|
| 📄 Paper | https://arxiv.org/abs/2610.10334 |
| 🌐 Project page & leaderboard | <https://cv-ac.github.io/hifd/> |
| 💻 Code | <https://github.com/infraface/hifd> |
| 🗂️ Dataset (UtilFace) | <https://huggingface.co/datasets/huiwei25/UtilFace> (gated, research-only) |
| ✉️ Contact | [hui.wei@oulu.fi](mailto:hui.wei@oulu.fi) |


## The UtilFace dataset

The [Hugging Face release](https://huggingface.co/datasets/huiwei25/UtilFace)
contains **no images**. It provides the identity keys and file list that define
UtilFace inside WebFace42M, the sampling strata, the HiFD scores of the twelve
baselines, Croissant metadata (core + Responsible AI), and **mini-UtilFace**, a
2,000-image bundle of per-image sub-scores that reproduces the published ranking
without any face data. To rebuild the images, obtain WebFace42M from its
official providers (<https://www.face-benchmark.org>) and select the files
listed in `data/images.csv`. The release is CC BY-NC 4.0 and gated for
non-commercial research on face de-identification only.

## This package

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

## License

The `hifd` code is MIT-licensed. The UtilFace release is CC BY-NC 4.0; the
underlying WebFace42M images remain subject to the WebFace42M terms of use.

## Citation

If you use HiFD or UtilFace in published work, please cite:

```bibtex
@inproceedings{wei2026private,
  title     = {How Private is Private? A Comparative Study for Face De-Identification},
  author    = {Wei, Hui and Yu, Hao and Kuurila-Zhang, Hui and Zhao, Guoying},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Evaluations and Datasets Track},
  year      = {2026}
}
```

If you use UtilFace, please also cite WebFace260M/WebFace42M (Zhu et al., CVPR 2021).

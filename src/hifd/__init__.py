"""HiFD: a holistic metric for face de-identification evaluation."""

from __future__ import annotations

__version__ = "0.1.0"

from .aggregation import (
    U1,
    U2,
    U3,
    aggregate_method_scores,
    batch_U1,
    batch_U2,
    batch_U3,
)
from .composite import compose_profiles, composite, load_profiles_yaml
from .constants import DEFAULT_CONSTANTS, DEFAULT_PROFILES, SCHEMA_VERSION
from .io import load_pair, load_predictions, validate_pair
from .pipeline import (
    PipelineResult,
    aggregate,
    compose,
    compute_agreements,
    run_pipeline,
)
from .scores.agreement import (
    batch_score_age,
    batch_score_categorical,
    batch_score_hr,
    score_age,
    score_bvp,
    score_categorical,
    score_gaze,
    score_hr,
    score_landmark,
)
from .scores.privacy import batch_privacy_score, privacy_ensemble, privacy_score
from .scores.quality import batch_quality_score, quality_score

__all__ = [
    "DEFAULT_CONSTANTS",
    "DEFAULT_PROFILES",
    "SCHEMA_VERSION",
    "U1",
    "U2",
    "U3",
    "PipelineResult",
    "__version__",
    "aggregate",
    "aggregate_method_scores",
    "batch_U1",
    "batch_U2",
    "batch_U3",
    "batch_privacy_score",
    "batch_quality_score",
    "batch_score_age",
    "batch_score_categorical",
    "batch_score_hr",
    "compose",
    "compose_profiles",
    "composite",
    "compute_agreements",
    "load_pair",
    "load_predictions",
    "load_profiles_yaml",
    "privacy_ensemble",
    "privacy_score",
    "quality_score",
    "run_pipeline",
    "score_age",
    "score_bvp",
    "score_categorical",
    "score_gaze",
    "score_hr",
    "score_landmark",
    "validate_pair",
]

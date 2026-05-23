"""Tests for hifd.pipeline."""

from __future__ import annotations

import json

import pytest

from hifd import pipeline as pl


def test_run_pipeline_produces_expected_csvs(mini_fixture_dir, tmp_path):
    out_dir = tmp_path / "out"
    result = pl.run_pipeline(
        methods_config=str(mini_fixture_dir / "methods.yaml"),
        estimators_config=str(mini_fixture_dir / "estimators.yaml"),
        data_dir=str(mini_fixture_dir),
        out_dir=str(out_dir),
    )

    assert (out_dir / "per_sample" / "per_sample_agreements.csv").exists()
    assert (out_dir / "per_method" / "per_method_aggregated.csv").exists()
    assert (out_dir / "per_method" / "method_level.csv").exists()
    assert (out_dir / "tables" / "composite.csv").exists()
    assert (out_dir / "tables" / "composite.tex").exists()

    # Loaded result mirrors disk
    assert {"methodA", "methodB"} == set(result.composite_df["method"])
    # Composite columns present
    for col in ("P", "Q", "U1", "U2", "U3", "privacy_first", "balanced", "clinical"):
        assert col in result.composite_df.columns


def test_run_pipeline_methodA_better_than_methodB(mini_fixture_dir, tmp_path):
    """methodA has less drift (vary=1) than methodB (vary=2) -> higher utility scores."""
    out_dir = tmp_path / "out"
    result = pl.run_pipeline(
        methods_config=str(mini_fixture_dir / "methods.yaml"),
        estimators_config=str(mini_fixture_dir / "estimators.yaml"),
        data_dir=str(mini_fixture_dir),
        out_dir=str(out_dir),
    )

    by_method = result.composite_df.set_index("method")
    # methodA should preserve utility better than methodB
    assert by_method.loc["methodA", "U1"] > by_method.loc["methodB", "U1"]


def test_run_pipeline_matches_expected_if_present(mini_fixture_dir, tmp_path):
    """Regression-pin: composite score for methodA matches expected.json (if present)."""
    expected_path = mini_fixture_dir / "expected.json"
    if not expected_path.exists():
        pytest.skip("expected.json not generated yet")

    out_dir = tmp_path / "out"
    result = pl.run_pipeline(
        methods_config=str(mini_fixture_dir / "methods.yaml"),
        estimators_config=str(mini_fixture_dir / "estimators.yaml"),
        data_dir=str(mini_fixture_dir),
        out_dir=str(out_dir),
    )
    expected = json.loads(expected_path.read_text())
    composite = result.composite_df.set_index("method")
    for method, vals in expected.items():
        for col, want in vals.items():
            got = float(composite.loc[method, col])
            assert got == pytest.approx(want, abs=1e-9), f"{method}.{col}: got {got}, want {want}"

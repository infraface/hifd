"""Tests for the `hifd` CLI."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from hifd import cli


def _run(args: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hifd.cli", *args],
        capture_output=True, text=True, cwd=cwd,
    )


def test_cli_version():
    r = _run(["version"])
    assert r.returncode == 0
    assert r.stdout.strip()


def test_cli_run_all_on_fixture(mini_fixture_dir, tmp_path):
    out = tmp_path / "out"
    r = _run([
        "run-all",
        "--methods", str(mini_fixture_dir / "methods.yaml"),
        "--estimators", str(mini_fixture_dir / "estimators.yaml"),
        "--data-dir", str(mini_fixture_dir),
        "--out", str(out),
    ])
    assert r.returncode == 0, f"stderr: {r.stderr}\nstdout: {r.stdout}"
    assert (out / "tables" / "composite.csv").exists()


def test_cli_validate_failure_returns_1(tmp_path, mini_fixture_dir):
    # Build a broken methods.yaml pointing to a non-existent method dir.
    bad = tmp_path / "methods.yaml"
    bad.write_text(
        "original:\n  base_path: utilface\n"
        "methods:\n  - name: nope\n    base_path: methods/does_not_exist\n"
    )
    r = _run([
        "validate",
        "--methods", str(bad),
        "--estimators", str(mini_fixture_dir / "estimators.yaml"),
        "--data-dir", str(mini_fixture_dir),
    ])
    assert r.returncode == 1
    assert "not found" in r.stderr or "not found" in r.stdout


# --- Direct unit tests for coverage ---


def test_main_no_command():
    """Test main() with no command (prints help)."""
    ret = cli.main([])
    assert ret == 2


def test_main_version_direct():
    """Test main() calling version command directly."""
    ret = cli.main(["version"])
    assert ret == 0


def test_main_with_log_level():
    """Test main() respects --log-level option."""
    ret = cli.main(["--log-level", "DEBUG", "version"])
    assert ret == 0


def test_cmd_version():
    """Test cmd_version directly."""
    import argparse
    args = argparse.Namespace()
    ret = cli.cmd_version(args)
    assert ret == 0


def test_cmd_validate_ok(mini_fixture_dir):
    """Test cmd_validate with valid config."""
    import argparse
    args = argparse.Namespace(
        methods=str(mini_fixture_dir / "methods.yaml"),
        estimators=str(mini_fixture_dir / "estimators.yaml"),
        data_dir=str(mini_fixture_dir),
    )
    ret = cli.cmd_validate(args)
    assert ret == 0


def test_cmd_validate_missing_method_dir(tmp_path, mini_fixture_dir):
    """Test cmd_validate with missing method directory."""
    import argparse
    bad_methods = tmp_path / "methods.yaml"
    bad_methods.write_text(
        "original:\n  base_path: utilface\n"
        "methods:\n  - name: nope\n    base_path: methods/does_not_exist\n"
    )
    args = argparse.Namespace(
        methods=str(bad_methods),
        estimators=str(mini_fixture_dir / "estimators.yaml"),
        data_dir=str(mini_fixture_dir),
    )
    ret = cli.cmd_validate(args)
    assert ret == 1


def test_cmd_compute_agreements(mini_fixture_dir, tmp_path):
    """Test cmd_compute_agreements."""
    import argparse
    args = argparse.Namespace(
        methods=str(mini_fixture_dir / "methods.yaml"),
        estimators=str(mini_fixture_dir / "estimators.yaml"),
        data_dir=str(mini_fixture_dir),
        out=str(tmp_path / "agreements"),
        constants=None,
    )
    ret = cli.cmd_compute_agreements(args)
    assert ret == 0


def test_cmd_aggregate(mini_fixture_dir, tmp_path):
    """Test cmd_aggregate."""
    import argparse
    # First generate agreements
    args_comp = argparse.Namespace(
        methods=str(mini_fixture_dir / "methods.yaml"),
        estimators=str(mini_fixture_dir / "estimators.yaml"),
        data_dir=str(mini_fixture_dir),
        out=str(tmp_path / "agreements"),
        constants=None,
    )
    cli.cmd_compute_agreements(args_comp)

    # Now aggregate
    args = argparse.Namespace(
        input=str(tmp_path / "agreements"),
        out=str(tmp_path / "aggregate"),
    )
    ret = cli.cmd_aggregate(args)
    assert ret == 0


def test_cmd_compose(mini_fixture_dir, tmp_path):
    """Test cmd_compose."""
    import argparse
    # First generate agreements and aggregate
    args_comp = argparse.Namespace(
        methods=str(mini_fixture_dir / "methods.yaml"),
        estimators=str(mini_fixture_dir / "estimators.yaml"),
        data_dir=str(mini_fixture_dir),
        out=str(tmp_path / "agreements"),
        constants=None,
    )
    cli.cmd_compute_agreements(args_comp)

    args_agg = argparse.Namespace(
        input=str(tmp_path / "agreements"),
        out=str(tmp_path / "aggregate"),
    )
    cli.cmd_aggregate(args_agg)

    # Now compose
    args = argparse.Namespace(
        input=str(tmp_path / "aggregate"),
        out=str(tmp_path / "composed.csv"),
        profiles=None,
        constants=None,
    )
    ret = cli.cmd_compose(args)
    assert ret == 0


def test_cmd_run_all(mini_fixture_dir, tmp_path):
    """Test cmd_run_all."""
    import argparse
    args = argparse.Namespace(
        methods=str(mini_fixture_dir / "methods.yaml"),
        estimators=str(mini_fixture_dir / "estimators.yaml"),
        data_dir=str(mini_fixture_dir),
        out=str(tmp_path / "full_run"),
        profiles=None,
        constants=None,
    )
    ret = cli.cmd_run_all(args)
    assert ret == 0

"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def mini_fixture_dir() -> Path:
    """Path to the mini multi-task fixture set (built by tests/_build_fixtures.py)."""
    return Path(__file__).parent / "fixtures" / "mini"

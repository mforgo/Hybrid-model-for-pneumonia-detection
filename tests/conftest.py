"""Shared pytest fixtures for the pipeline test suite."""

import pytest


@pytest.fixture
def cfg(tmp_path):
    """Config instance with tmp_path-based artifacts/results/figures dirs.

    Import-guarded: pipeline.config is implemented in a later task (T2).
    Until then, tests requesting this fixture are skipped at setup time
    rather than breaking collection.
    """
    try:
        from pipeline.config import Config
    except ImportError:
        pytest.skip("pipeline.config not implemented yet (T2)")

    return Config(
        artifacts_dir=str(tmp_path / "artifacts"),
        results_dir=str(tmp_path / "results"),
        figures_dir=str(tmp_path / "figures"),
    )
"""Tests for ``pipeline.features``: deterministic DANN feature extraction.

RED phase: ``pipeline.features`` is an empty stub and torch is not installed
in the development environment, so this test fails at runtime until Task 6
implements the module. Collection must still succeed, hence all heavy
imports are lazy (function-level).
"""

import numpy as np


def test_extract_dann_deterministic():
    """Same seed -> identical DANN-extracted features (reproducibility)."""
    from pipeline.features import extract_dann_features

    rng = np.random.default_rng(6)
    X = rng.normal(size=(16, 768))
    y = rng.integers(0, 2, size=16)
    domain = rng.integers(0, 2, size=16)

    feats1, _ = extract_dann_features(X, y, domain, seed=6, epochs=1)
    feats2, _ = extract_dann_features(X, y, domain, seed=6, epochs=1)

    assert feats1.shape == feats2.shape
    np.testing.assert_array_equal(feats1, feats2)
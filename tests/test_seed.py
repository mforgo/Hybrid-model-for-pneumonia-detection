"""Tests for ``pipeline.seed``: global reproducibility seeding.

RED phase: ``seed_everything`` imports ``torch`` internally, which is not
installed in the development environment, so this test fails at runtime
until the dependency is available. Collection must still succeed, hence the
pipeline import is lazy (function-level).
"""

import random

import numpy as np


def test_seed_deterministic():
    """Same seed -> identical random sequences from ``random`` and ``numpy``."""
    from pipeline.seed import seed_everything

    seed_everything(6)
    py_seq = [random.random() for _ in range(5)]
    np_seq = np.random.rand(5)

    seed_everything(6)
    assert [random.random() for _ in range(5)] == py_seq
    np.testing.assert_array_equal(np.random.rand(5), np_seq)
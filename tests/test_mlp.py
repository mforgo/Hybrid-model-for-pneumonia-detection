"""Tests for ``pipeline.mlp``: forward pass shape and pos_weight computation.

RED phase: ``pipeline.mlp`` is an empty stub and torch is not installed in
the development environment, so these tests fail at runtime until Task 10
implements the module. Collection must still succeed, hence all heavy
imports are lazy (function-level).
"""

import numpy as np
import pytest


def test_mlp_forward():
    """MLP must output raw logits of shape (B, 1) for (B, 64) input."""
    import torch

    from pipeline.mlp import MLP

    model = MLP(input_dim=64, hidden=32, dropout=0.3)
    out = model(torch.randn(8, 64))
    assert out.shape == (8, 1)
    assert torch.isfinite(out).all()


def test_mlp_pos_weight():
    """compute_pos_weight must return n_neg / n_pos for imbalanced labels."""
    from pipeline.mlp import compute_pos_weight

    y = np.array([0, 0, 0, 1])  # 3 negatives : 1 positive
    w = compute_pos_weight(y)
    assert w > 0
    assert w == pytest.approx(3.0)
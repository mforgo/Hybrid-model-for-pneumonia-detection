"""Tests for ``pipeline.vqc``: Z0 extraction helper and CORAL/MixUp gradient flow.

RED phase: ``pipeline.vqc`` / ``pipeline.qpu`` are empty stubs, so these
tests fail at runtime until Tasks 9/12 implement the modules. Collection
must still succeed, hence all pipeline imports are lazy (function-level).
"""

import numpy as np
import pytest


def test_z0_extraction_helper():
    """extract_z0_from_counts must read qubit 0 from the LAST char of the
    big-endian Qiskit bitstring (bug fix #3)."""
    from pipeline.qpu import extract_z0_from_counts

    # 500x qubit0='0' (+1) + 500x qubit0='1' (-1) -> <Z0> = 0.0
    assert extract_z0_from_counts({"111110": 500, "111111": 500}, n_qubits=6) == pytest.approx(0.0)
    # all qubit0='0' -> <Z0> = +1.0
    assert extract_z0_from_counts({"000000": 1024}, n_qubits=6) == pytest.approx(1.0)
    # all qubit0='1' -> <Z0> = -1.0
    assert extract_z0_from_counts({"111111": 1024}, n_qubits=6) == pytest.approx(-1.0)


def test_coral_mixup_gradient_flow():
    """CORAL + MixUp terms must contribute non-zero gradients w.r.t. the
    trainable parameters (bug fixes #1/#2: previously dead code)."""
    from pipeline.vqc import combined_loss

    import torch

    rng = np.random.default_rng(6)
    Xb = rng.normal(size=(8, 64))
    Xb = Xb / np.linalg.norm(Xb, axis=1, keepdims=True)
    yb = np.array([1.0, -1.0] * 4)
    X_test_pool = rng.normal(size=(8, 64))
    X_test_pool = X_test_pool / np.linalg.norm(X_test_pool, axis=1, keepdims=True)
    params = rng.normal(size=(3, 6, 3)) * 0.1

    # Identical MixUp draws in paired runs (the Beta sampler uses the global
    # torch RNG), so any difference isolates the term under test.
    torch.manual_seed(0)
    loss, grads = combined_loss(
        params, Xb, yb, X_test_pool, circuit=None,
        lambda_coral=0.1, mixup_alpha=0.2,
    )
    assert np.isfinite(loss)
    assert np.asarray(grads).shape == params.shape
    # The ansatz has structurally zero gradients on RZ gates that commute
    # with the ⟨Z₀⟩ measurement, so only some non-zero gradients are required.
    assert np.any(np.abs(grads) > 1e-9)

    # CORAL must be live: switching its weight must change the gradient.
    torch.manual_seed(0)
    _, grads_no_coral = combined_loss(
        params, Xb, yb, X_test_pool, circuit=None,
        lambda_coral=0.0, mixup_alpha=0.2,
    )
    assert np.max(np.abs(np.asarray(grads) - np.asarray(grads_no_coral))) > 1e-9

    # MixUp must be live: a different alpha (same torch seed) must change
    # the gradient through the mixed batch.
    torch.manual_seed(0)
    loss_strong, grads_strong = combined_loss(
        params, Xb, yb, X_test_pool, circuit=None,
        lambda_coral=0.1, mixup_alpha=0.5,
    )
    assert loss_strong != loss
    assert np.max(np.abs(np.asarray(grads) - np.asarray(grads_strong))) > 1e-9
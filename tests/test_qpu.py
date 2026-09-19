"""Tests for ``pipeline.qpu``: Z0 extraction from big-endian Qiskit counts.

RED phase: ``pipeline.qpu`` is an empty stub, so this test fails at runtime
until Task 12 implements the module. Collection must still succeed, hence
the pipeline import is lazy (function-level).
"""

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


def test_job_chunk_size_explicit_cap():
    """An explicit qpu_max_circuits_per_job must cap the chunk, never exceed
    the circuit count, and never produce a zero-sized job."""
    from pipeline.qpu import _job_chunk_size

    assert _job_chunk_size(100, 1024, max_circuits_per_job=32) == 32
    assert _job_chunk_size(10, 1024, max_circuits_per_job=32) == 10  # capped at n_circuits
    assert _job_chunk_size(1, 1024, max_circuits_per_job=32) == 1
    assert _job_chunk_size(0, 1024, max_circuits_per_job=32) == 1  # never 0


def test_job_chunk_size_auto_full_test_fits_one_job():
    """Auto cap (0) derives from the ~10M executions/job service limit scaled
    by the safety margin; the full 624-circuit test set at 1024 shots must fit
    in ONE SamplerV2 job (624 x 1024 = 638,976 executions)."""
    from pipeline.qpu import _MAX_EXECUTIONS_PER_JOB, _job_chunk_size

    assert _job_chunk_size(624, 1024) == 624  # full test set, single job

    auto = _job_chunk_size(20_000, 1024)
    hard_cap = _MAX_EXECUTIONS_PER_JOB // 1024
    assert auto == int(hard_cap * 0.8)  # safety margin applied to the hard cap
    assert auto < hard_cap
    assert _job_chunk_size(1, 1024) == 1


def _label_sorted_y(n_neg: int = 234, n_pos: int = 390) -> "np.ndarray":
    """Replicate the real test split: label-sorted (negatives first), which
    made the naive ``X_test[:n]`` QPU slice single-class."""
    import numpy as np

    return np.concatenate([np.zeros(n_neg, dtype=int), np.ones(n_pos, dtype=int)])


def test_select_qpu_subset_balanced():
    """select_qpu_subset must draw a class-balanced subset and return (X_sub,
    y_sub, idx) with idx = original row indices (bug fix: label-sorted split
    made naive slices single-class, leaving AUC/sensitivity undefined)."""
    import numpy as np
    from pipeline.qpu import select_qpu_subset

    rng = np.random.default_rng(0)
    X = rng.normal(size=(624, 64))
    y = _label_sorted_y()

    X_sub, y_sub, idx = select_qpu_subset(X, y, n_samples=50, seed=6)

    assert len(idx) == 50
    assert len(set(idx.tolist())) == 50
    assert 0 <= int(idx.min()) and int(idx.max()) < 624
    assert int(np.sum(y_sub == 1)) == 25
    assert int(np.sum(y_sub == 0)) == 25
    assert np.array_equal(X[idx], X_sub)
    assert np.array_equal(y[idx], y_sub)
    # Submission order must be class-mixed (interleaved), not class-blocked.
    first_half_labels = y_sub[:25]
    assert int(np.sum(first_half_labels == 1)) >= 1
    assert int(np.sum(first_half_labels == 0)) >= 1


def test_select_qpu_subset_reproducible():
    """Same seed -> identical selection; different seed -> different draw."""
    import numpy as np
    from pipeline.qpu import select_qpu_subset

    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 64))
    y = _label_sorted_y(n_neg=40, n_pos=60)

    _, _, idx_a = select_qpu_subset(X, y, n_samples=20, seed=6)
    _, _, idx_b = select_qpu_subset(X, y, n_samples=20, seed=6)
    _, _, idx_c = select_qpu_subset(X, y, n_samples=20, seed=7)

    assert np.array_equal(idx_a, idx_b)
    assert not np.array_equal(idx_a, idx_c)


def test_select_qpu_subset_edge_cases():
    """Subset size is capped at len(y); under-populated classes are topped up."""
    import numpy as np
    from pipeline.qpu import select_qpu_subset

    X = np.zeros((10, 4))
    y = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    _, y_sub, idx = select_qpu_subset(X, y, n_samples=20, seed=6)
    assert len(idx) == 10
    assert len(set(idx.tolist())) == 10
    assert int(np.sum(y_sub == 1)) == 8
    assert int(np.sum(y_sub == 0)) == 2

    y_all_pos = np.ones(10, dtype=int)
    _, y_sub2, idx2 = select_qpu_subset(X, y_all_pos, n_samples=8, seed=6)
    assert len(idx2) == 8
    assert int(np.sum(y_sub2 == 1)) == 8
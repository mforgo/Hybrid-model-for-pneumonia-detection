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
"""Tests for ``pipeline.ansatz``: variational circuit parameter counting.

RED phase: ``pipeline.ansatz`` is currently an empty stub, so these tests
fail at runtime until Task 8 implements the module. Collection must still
succeed, hence the pipeline imports are lazy (function-level).
"""


def test_circuit_param_count_62():
    """Default flags (learnable scale + measurement basis) -> 62 params."""
    from pipeline.ansatz import count_params

    assert count_params(n_qubits=6, n_layers=3, use_scale=True, use_meas_basis=True) == 62


def test_circuit_param_count_54():
    """Both ablation flags OFF -> 54 params (3 x 6 x 3 Euler angles)."""
    from pipeline.ansatz import count_params

    assert count_params(n_qubits=6, n_layers=3, use_scale=False, use_meas_basis=False) == 54
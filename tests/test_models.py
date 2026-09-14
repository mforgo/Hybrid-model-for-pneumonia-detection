"""Tests for ``pipeline.models``: the ModelSpec registry contract.

Every trainable model exposes a unified train contract
``train(cfg, X_train, y_train, X_val, y_val, X_test=None, y_test=None)``
returning a dict with at least ``val_probs``, ``test_probs``, ``history``,
``params`` and ``epochs_trained``. Parameter counts are locked design
values (the VQC's 62 = 54 rotations + 6 scale + 2 measurement basis).
Imports stay lazy (function-level) to match the module-level
stdlib+numpy-only import policy.
"""

# (kind, locked param count at default config). -1 = non-parametric.
_LOCKED = {
    "vqc": ("vqc", 62),
    "mlp": ("torch", 2113),
    "logreg": ("sklearn", 65),
    "rbf_svm": ("sklearn", -1),
    "random_forest": ("sklearn", -1),
    "knn": ("sklearn", -1),
    "mlp_deep": ("torch", 6273),
    "vqc_angle": ("vqc", 80),
    "vqc_he": ("vqc", 38),
    "quantum_kernel": ("quantum_kernel", -1),
}


def test_registry_covers_all_models():
    from pipeline import models

    assert set(models.available_models()) == set(_LOCKED)


def test_param_counts_match_locked_design():
    from pipeline import models
    from pipeline.config import load_config

    cfg = load_config("configs/default.yaml")
    for name, (kind, expected) in _LOCKED.items():
        spec = models.get_model(name)
        assert spec.kind == kind, name
        assert spec.param_count(cfg) == expected, name


def test_logreg_train_returns_unified_dict_contract():
    import numpy as np
    from pipeline import models
    from pipeline.config import load_config

    cfg = load_config("configs/default.yaml")
    rng = np.random.default_rng(6)
    X_tr, y_tr = rng.normal(size=(60, 64)), (rng.random(60) > 0.5).astype(int)
    X_va, y_va = rng.normal(size=(30, 64)), (rng.random(30) > 0.5).astype(int)
    X_te, y_te = rng.normal(size=(20, 64)), (rng.random(20) > 0.5).astype(int)

    out = models.MODELS["logreg"].train(cfg, X_tr, y_tr, X_va, y_va, X_te, y_te)

    assert set(out) >= {"val_probs", "test_probs", "history", "params", "epochs_trained"}
    assert out["val_probs"].shape == (30,)
    assert out["test_probs"].shape == (20,)
    assert np.all((out["val_probs"] >= 0) & (out["val_probs"] <= 1))
    assert np.all((out["test_probs"] >= 0) & (out["test_probs"] <= 1))
    # history: list for epoch-based trainers (VQC/MLP), dict for sklearn.
    assert isinstance(out["history"], (list, dict))
    assert isinstance(out["epochs_trained"], int) and out["epochs_trained"] >= 1


def test_vqc_param_count_breaks_down_62():
    from pipeline.ansatz import count_params

    assert count_params(n_qubits=6, n_layers=3, use_scale=True, use_meas_basis=True) == 62
    assert count_params(n_qubits=6, n_layers=3, use_scale=False, use_meas_basis=True) == 56
    assert count_params(n_qubits=6, n_layers=3, use_scale=False, use_meas_basis=False) == 54
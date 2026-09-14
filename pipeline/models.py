"""Model registry for the hybrid quantum-classical pneumonia pipeline.

Defines the ``ModelSpec`` dataclass and the ``MODELS`` registry mapping
model names to their training functions, parameter-count functions, and
dependency requirements.  The registry is the single source of truth for
the benchmark stage (later wave): every model in ``cfg.models`` is trained
on identical features and compared via a CSV.

Every ``ModelSpec.train`` callable has the uniform signature::

    (cfg, X_train, y_train, X_val, y_val, X_test=None, y_test=None) -> dict

and returns a dict with keys ``val_probs``, ``test_probs``, ``history``,
``params``, ``epochs_trained`` and optionally ``n_params``.  ``val_probs`` /
``test_probs`` are flat float arrays in ``[0, 1]``.

Import policy
-------------
Only the standard library and NumPy are imported at module level.  All heavy
libraries (sklearn, torch, pennylane, qiskit) are imported lazily inside the
train functions, so ``import pipeline.models`` succeeds in a stdlib+numpy-only
environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

__all__ = [
    "ModelSpec",
    "MODELS",
    "get_model",
    "available_models",
]


# ---------------------------------------------------------------------------
# Registry entry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Registry entry describing one trainable model.

    Attributes:
        name: Unique model name (registry key).
        kind: One of ``"vqc"``, ``"torch"``, ``"sklearn"``,
            ``"quantum_kernel"``.
        train: Callable ``(cfg, X_train, y_train, X_val, y_val, X_test=None,
            y_test=None) -> dict``.  The returned dict MUST contain
            ``val_probs``, ``test_probs``, ``history``, ``params``,
            ``epochs_trained`` and optionally ``n_params``.
        param_count: Pure-arithmetic parameter counter ``(cfg) -> int``;
            ``-1`` for non-parametric models (data-dependent counts are
            reported post-fit via the ``n_params`` result key).
        requires: Tuple of extra package names needed at train time.
    """

    name: str
    kind: str
    train: Callable
    param_count: Callable[["Config"], int]
    requires: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Lazy heavy imports
# ---------------------------------------------------------------------------


def _import_torch():
    """Import torch lazily, raising a clear RuntimeError when missing."""
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError(
            "pipeline.models requires PyTorch at call time, but torch is not "
            "installed. Install the project dependencies (see requirements.txt)."
        ) from exc


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _sklearn_predict_proba(model, X_val, X_test):
    """Predict positive-class probabilities with a fitted sklearn classifier.

    Args:
        model: A fitted sklearn classifier exposing ``predict_proba``.
        X_val: Validation features, shape ``(N, 64)``.
        X_test: Optional test features, shape ``(N, 64)``.

    Returns:
        ``(val_probs, test_probs)`` — flat float64 arrays in ``[0, 1]``;
        ``test_probs`` is ``None`` when ``X_test`` is empty/``None``.
    """
    val_probs = np.asarray(
        model.predict_proba(np.asarray(X_val, dtype=np.float64))[:, 1],
        dtype=np.float64,
    )
    test_probs = None
    if X_test is not None and len(X_test) > 0:
        test_probs = np.asarray(
            model.predict_proba(np.asarray(X_test, dtype=np.float64))[:, 1],
            dtype=np.float64,
        )
    return val_probs, test_probs


# ---------------------------------------------------------------------------
# vqc / mlp delegation (existing train functions, verbatim)
# ---------------------------------------------------------------------------


def _train_vqc(cfg, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """Train the amplitude-encoding VQC (delegates to ``pipeline.vqc.train_vqc``)."""
    from pipeline.seed import seed_everything
    from pipeline.vqc import train_vqc

    seed_everything(int(getattr(cfg, "seed", 6)))
    return train_vqc(cfg, X_train, y_train, X_val, y_val, X_test, y_test)


def _count_vqc(cfg):
    """62 trainable parameters by default (3 layers x 6 qubits x 3 + scale + meas)."""
    from pipeline.ansatz import count_params

    return count_params(
        int(getattr(cfg, "n_qubits", 6)),
        int(getattr(cfg, "n_layers", 3)),
        bool(getattr(cfg, "use_learnable_scale", True)),
        bool(getattr(cfg, "use_measurement_basis", True)),
    )


def _train_mlp(cfg, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """Train the classical MLP baseline (delegates to ``pipeline.mlp.train_mlp``)."""
    from pipeline.seed import seed_everything
    from pipeline.mlp import train_mlp

    seed_everything(int(getattr(cfg, "seed", 6)))
    return train_mlp(cfg, X_train, y_train, X_val, y_val, X_test, y_test)


def _count_mlp(cfg):
    """``Linear(64 -> hidden) -> ReLU -> Dropout -> Linear(hidden -> 1)``."""
    hidden = int(getattr(cfg, "mlp_hidden", 32))
    return 64 * hidden + hidden + hidden + 1


# ---------------------------------------------------------------------------
# sklearn baselines
# ---------------------------------------------------------------------------


def _train_logreg(cfg, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """Train a logistic-regression baseline (sklearn)."""
    from pipeline.seed import seed_everything
    from sklearn.linear_model import LogisticRegression

    seed_everything(int(getattr(cfg, "seed", 6)))

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)

    C = float(getattr(cfg, "logreg_C", 1.0))
    seed = int(getattr(cfg, "seed", 6))
    model = LogisticRegression(C=C, max_iter=2000, random_state=seed)
    model.fit(X_train, y_train)

    val_probs, test_probs = _sklearn_predict_proba(model, X_val, X_test)

    return {
        "val_probs": val_probs,
        "test_probs": test_probs,
        "history": {},
        "params": {"C": C, "max_iter": 2000, "random_state": seed},
        "epochs_trained": int(model.n_iter_[0]),
        "n_params": int(X_train.shape[1]) + 1,
    }


def _count_logreg(cfg):
    """64 features + intercept."""
    return 65


def _train_rbf_svm(cfg, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """Train an RBF-kernel SVM baseline (sklearn)."""
    from pipeline.seed import seed_everything
    from sklearn.svm import SVC

    seed_everything(int(getattr(cfg, "seed", 6)))

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)

    C = float(getattr(cfg, "svm_C", 1.0))
    gamma = str(getattr(cfg, "svm_gamma", "scale"))
    seed = int(getattr(cfg, "seed", 6))
    model = SVC(C=C, gamma=gamma, probability=True, random_state=seed)
    model.fit(X_train, y_train)

    val_probs, test_probs = _sklearn_predict_proba(model, X_val, X_test)

    return {
        "val_probs": val_probs,
        "test_probs": test_probs,
        "history": {},
        "params": {"C": C, "gamma": gamma, "probability": True, "random_state": seed},
        "epochs_trained": 1,
        "n_params": int(len(model.support_)) + 1,
    }


def _count_rbf_svm(cfg):
    """Data-dependent (``n_support_vectors + 1``); reported post-fit."""
    return -1


def _train_random_forest(cfg, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """Train a random-forest baseline (sklearn)."""
    from pipeline.seed import seed_everything
    from sklearn.ensemble import RandomForestClassifier

    seed_everything(int(getattr(cfg, "seed", 6)))

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)

    n_estimators = int(getattr(cfg, "rf_n_estimators", 200))
    max_depth = getattr(cfg, "rf_max_depth", None)
    seed = int(getattr(cfg, "seed", 6))
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed, n_jobs=-1
    )
    model.fit(X_train, y_train)

    val_probs, test_probs = _sklearn_predict_proba(model, X_val, X_test)

    return {
        "val_probs": val_probs,
        "test_probs": test_probs,
        "history": {},
        "params": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "random_state": seed,
        },
        "epochs_trained": 1,
    }


def _count_random_forest(cfg):
    """Non-parametric (ensemble of decision trees)."""
    return -1


def _train_knn(cfg, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """Train a k-nearest-neighbours baseline (sklearn)."""
    from pipeline.seed import seed_everything
    from sklearn.neighbors import KNeighborsClassifier

    seed_everything(int(getattr(cfg, "seed", 6)))

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)

    k = int(getattr(cfg, "knn_k", 5))
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    val_probs, test_probs = _sklearn_predict_proba(model, X_val, X_test)

    return {
        "val_probs": val_probs,
        "test_probs": test_probs,
        "history": {},
        "params": {"n_neighbors": k},
        "epochs_trained": 1,
    }


def _count_knn(cfg):
    """Non-parametric (instance-based)."""
    return -1


# ---------------------------------------------------------------------------
# Deep MLP baseline (torch, lazy)
# ---------------------------------------------------------------------------


def _train_mlp_deep(cfg, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """Train a deeper MLP with hidden sizes from ``cfg.mlp_deep_hidden``.

    Reuses the ``pipeline.mlp.train_mlp`` loop pattern (BCEWithLogitsLoss
    with pos_weight, Adam, linear warm-up + cosine annealing, early stopping
    on validation loss) generalised to an arbitrary list of hidden sizes.
    """
    from pipeline.seed import get_torch_generator, seed_everything
    from pipeline.mlp import compute_pos_weight

    seed_everything(int(getattr(cfg, "seed", 6)))

    torch = _import_torch()
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    hidden = list(getattr(cfg, "mlp_deep_hidden", [64, 32]))
    dropout = float(getattr(cfg, "mlp_dropout", 0.3))
    batch_size = int(getattr(cfg, "batch_size", 16))
    lr = float(getattr(cfg, "learning_rate", 1e-3))
    lr_min = float(getattr(cfg, "lr_min", 1e-5))
    warmup_epochs = int(getattr(cfg, "warmup_epochs", 3))
    epochs = min(int(getattr(cfg, "epochs", 50)), 30)
    patience = int(getattr(cfg, "early_stopping_patience", 10))
    seed = int(getattr(cfg, "seed", 6))

    device = str(getattr(cfg, "device", "auto"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    layers: list[nn.Module] = []
    in_dim = X_train.shape[1]
    for h in hidden:
        layers.append(nn.Linear(in_dim, h))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    model = nn.Sequential(*layers)
    model.to(device)

    pos_weight = torch.tensor(
        compute_pos_weight(y_train), dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=lr_min
    )

    train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).to(device)

    train_ds = TensorDataset(train_t, y_train_t)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=get_torch_generator(seed),
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    epochs_trained = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "lr": []}

    model.train()
    for epoch in range(epochs):
        if epoch < warmup_epochs:
            warmup_factor = (epoch + 1) / max(warmup_epochs, 1)
            for group in optimizer.param_groups:
                group["lr"] = lr * warmup_factor
        else:
            scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb).squeeze(-1)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1

        model.eval()
        with torch.no_grad():
            val_logits = model(val_t).squeeze(-1)
            val_loss = float(criterion(val_logits, y_val_t).item())
        model.train()

        history["train_loss"].append(epoch_loss / max(n_batches, 1))
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        epochs_trained = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"  MLP-deep early stopping at epoch {epoch + 1} "
                    f"(no val improvement for {patience} epochs)"
                )
                break

    if best_state is None:
        raise ValueError(
            "mlp_deep: no epochs completed; cannot produce a best model."
        )

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_probs = torch.sigmoid(model(val_t)).squeeze(-1).cpu().numpy()
        test_probs = None
        if X_test is not None and len(X_test) > 0:
            test_t = torch.tensor(
                np.asarray(X_test, dtype=np.float32), dtype=torch.float32
            ).to(device)
            test_probs = torch.sigmoid(model(test_t)).squeeze(-1).cpu().numpy()

    return {
        "best_state": best_state,
        "history": history,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "epochs_trained": epochs_trained,
        "params": {"hidden": hidden, "dropout": dropout, "lr": lr, "epochs": epochs},
    }


def _count_mlp_deep(cfg):
    """Sum of ``Linear`` weight+bias counts over ``cfg.mlp_deep_hidden``."""
    hidden = list(getattr(cfg, "mlp_deep_hidden", [64, 32]))
    if not hidden:
        return 64 + 1
    n = 64 * hidden[0] + hidden[0]
    for a, b in zip(hidden[:-1], hidden[1:]):
        n += a * b + b
    n += hidden[-1] + 1
    return n


# ---------------------------------------------------------------------------
# VQC with alternative encodings (angle / hardware-efficient)
# ---------------------------------------------------------------------------


def _train_vqc_encoder(
    cfg,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test=None,
    y_test=None,
    encoding: str = "angle",
):
    """Train a VQC with angle or hardware-efficient encoding.

    Mirrors ``pipeline.vqc.train_vqc`` (Adam, cosine LR + warm-up, early
    stopping on validation loss) but builds the circuit with
    ``build_vqc_circuit_angle`` / ``build_vqc_circuit_he`` instead of the
    amplitude-encoding circuit.  Uses MSE loss with proper PennyLane
    gradient computation.  The returned dict has the same keys as
    ``train_vqc``.
    """
    from pipeline.seed import seed_everything
    from pipeline.vqc import (
        _l2_normalize,
        get_pennylane_device,
        lr_cosine_warmup,
    )
    from pipeline.ansatz import (
        build_vqc_circuit_angle,
        build_vqc_circuit_he,
        count_params_angle,
        count_params_he,
    )
    from sklearn.metrics import balanced_accuracy_score

    seed_everything(int(getattr(cfg, "seed", 6)))

    import pennylane as qml
    from pennylane import numpy as pnp

    n_qubits = int(getattr(cfg, "n_qubits", 6))
    n_layers = int(getattr(cfg, "n_layers", 3))
    use_scale = bool(getattr(cfg, "use_learnable_scale", True))
    use_meas = bool(getattr(cfg, "use_measurement_basis", True))
    batch_size = int(getattr(cfg, "batch_size", 16))
    lr0 = float(getattr(cfg, "learning_rate", 1e-3))
    lr_min = float(getattr(cfg, "lr_min", 1e-5))
    warmup = int(getattr(cfg, "warmup_epochs", 3))
    epochs = int(getattr(cfg, "epochs", 50))
    patience = int(getattr(cfg, "early_stopping_patience", 10))
    results_dir = Path(str(getattr(cfg, "results_dir", "results")))

    X_train = _l2_normalize(X_train)
    X_val = _l2_normalize(X_val)
    X_test_arr = None if X_test is None else _l2_normalize(X_test)

    y_train = np.asarray(y_train, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    y_train_pm1 = np.where(y_train > 0.5, 1.0, -1.0)
    y_val_pm1 = np.where(y_val > 0.5, 1.0, -1.0)
    y_val_01 = (y_val_pm1 > 0).astype(int)

    n_train = X_train.shape[0]

    dev, diff_method = get_pennylane_device(cfg)
    dev_name = (
        getattr(dev, "short_name", None) or getattr(dev, "name", "") or str(dev)
    )
    print(f"VQC ({encoding}) training on {dev_name} (diff_method={diff_method})")

    if encoding == "angle":
        circuit = build_vqc_circuit_angle(
            n_qubits,
            n_layers,
            use_scale=use_scale,
            use_meas_basis=use_meas,
            device=dev,
            diff_method=diff_method,
        )
        n_params = count_params_angle(n_qubits, n_layers, use_scale, use_meas)
    elif encoding == "he":
        circuit = build_vqc_circuit_he(
            n_qubits,
            n_layers,
            use_meas_basis=use_meas,
            device=dev,
            diff_method=diff_method,
        )
        n_params = count_params_he(n_qubits, n_layers, use_meas)
    else:
        raise ValueError(
            f"Unknown encoding {encoding!r}; expected 'angle' or 'he'."
        )

    if encoding == "angle":
        rot_init = np.random.uniform(0.0, 2.0 * np.pi, size=(n_layers * n_qubits * 3,))
        angle_init = np.ones(n_layers * n_qubits)
        parts = [rot_init, angle_init]
        if use_scale:
            parts.append(np.ones(n_qubits))
        if use_meas:
            parts.append(np.zeros(2))
    else:
        rot_init = np.random.uniform(0.0, 2.0 * np.pi, size=(n_layers * n_qubits * 2,))
        parts = [rot_init]
        if use_meas:
            parts.append(np.zeros(2))
    params = np.concatenate(parts)
    if params.shape != (n_params,):
        raise ValueError(
            f"parameter layout mismatch: built {params.shape}, expected ({n_params},)"
        )

    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    t = 0

    def _batch_mse(params_np, Xb_arr, yb_arr):
        """MSE loss + PennyLane gradient for a mini-batch.

        Uses ``pennylane.numpy`` with ``requires_grad=True`` so that
        ``qml.grad`` tracks the parameter gradient through the circuit
        execution.
        """
        n = len(yb_arr)

        def _mse(p):
            p_flat = pnp.asarray(p, dtype=np.float64)
            total = pnp.float64(0.0)
            for i in range(n):
                pred = circuit(p_flat, Xb_arr[i])
                total = total + (pred - yb_arr[i]) ** 2
            return total / n

        pt = pnp.asarray(params_np, dtype=np.float64, requires_grad=True)
        loss_val = float(_mse(pt))
        grad_raw = qml.grad(_mse)(pt)
        return loss_val, np.asarray(grad_raw, dtype=np.float64)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_bal_acc": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_val_bal_acc = -1.0
    best_params = params.copy()
    best_epoch = -1
    patience_counter = 0
    epochs_trained = 0

    for epoch in range(epochs):
        epochs_trained += 1
        lr = lr_cosine_warmup(epoch, lr0, lr_min, warmup, epochs)

        perm = np.random.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            Xb = X_train[idx]
            yb = y_train_pm1[idx]

            loss, grads = _batch_mse(params, Xb, yb)

            t += 1
            m = beta1 * m + (1.0 - beta1) * grads
            v = beta2 * v + (1.0 - beta2) * grads**2
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)
            params = params - lr * m_hat / (np.sqrt(v_hat) + eps)

            epoch_loss += float(loss)
            n_batches += 1

        z_val = np.asarray([circuit(params, x) for x in X_val])
        val_loss = float(np.mean((z_val - y_val_pm1) ** 2))
        val_probs_epoch = (1.0 + z_val) / 2.0
        val_bal_acc = float(
            balanced_accuracy_score(y_val_01, (val_probs_epoch > 0.5).astype(int))
        )

        history["train_loss"].append(epoch_loss / max(n_batches, 1))
        history["val_loss"].append(val_loss)
        history["val_bal_acc"].append(val_bal_acc)
        history["lr"].append(lr)

        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_params = params.copy()
            best_epoch = epoch + 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"  VQC ({encoding}) early stopping at epoch {epoch + 1} "
                    f"(no val-loss improvement for {patience} epochs)"
                )
                break

        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"  VQC ({encoding}) Epoch {epoch + 1}/{epochs} | "
                f"Train: {history['train_loss'][-1]:.4f} | "
                f"Val: {history['val_loss'][-1]:.4f} | "
                f"BalAcc: {history['val_bal_acc'][-1]:.4f} | "
                f"LR: {lr:.2e}"
            )

    val_probs = (1.0 + np.asarray([circuit(best_params, x) for x in X_val])) / 2.0
    test_probs = None
    if X_test_arr is not None:
        test_probs = (
            1.0 + np.asarray([circuit(best_params, x) for x in X_test_arr])
        ) / 2.0

    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / f"vqc_{encoding}_best_params.npy", best_params)
    with open(results_dir / f"vqc_{encoding}_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    np.save(results_dir / f"vqc_{encoding}_val_probs.npy", val_probs)
    if test_probs is not None:
        np.save(results_dir / f"vqc_{encoding}_test_probs.npy", test_probs)

    print(
        f"  VQC ({encoding}) training finished: {epochs_trained} epochs, "
        f"best val balanced accuracy {best_val_bal_acc:.4f} at epoch {best_epoch}"
    )

    return {
        "best_params": best_params,
        "history": history,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "epochs_trained": epochs_trained,
        "params": {"encoding": encoding, "n_qubits": n_qubits, "n_layers": n_layers},
    }


def _train_vqc_angle(cfg, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """Train the angle-encoding VQC (per-layer RY re-upload)."""
    return _train_vqc_encoder(
        cfg, X_train, y_train, X_val, y_val, X_test, y_test, encoding="angle"
    )


def _count_vqc_angle(cfg):
    """``n_layers*(n_qubits*3 + n_qubits) + scale + meas`` (80 by default)."""
    from pipeline.ansatz import count_params_angle

    return count_params_angle(
        int(getattr(cfg, "n_qubits", 6)),
        int(getattr(cfg, "n_layers", 3)),
        bool(getattr(cfg, "use_learnable_scale", True)),
        bool(getattr(cfg, "use_measurement_basis", True)),
    )


def _train_vqc_he(cfg, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """Train the hardware-efficient-ansatz VQC."""
    return _train_vqc_encoder(
        cfg, X_train, y_train, X_val, y_val, X_test, y_test, encoding="he"
    )


def _count_vqc_he(cfg):
    """``n_layers*n_qubits*2 + meas`` (38 by default)."""
    from pipeline.ansatz import count_params_he

    return count_params_he(
        int(getattr(cfg, "n_qubits", 6)),
        int(getattr(cfg, "n_layers", 3)),
        bool(getattr(cfg, "use_measurement_basis", True)),
    )


# ---------------------------------------------------------------------------
# Quantum kernel (lazy qiskit)
# ---------------------------------------------------------------------------


def _train_quantum_kernel(cfg, X_train, y_train, X_val, y_val, X_test=None, y_test=None):
    """Train a quantum-kernel SVM (qiskit-machine-learning, lazy import).

    Sub-samples the training set to ``cfg.qk_n_train`` rows, computes the
    ``FidelityQuantumKernel`` Gram matrices, and fits a precomputed-kernel
    SVC.  Raises a clear ``RuntimeError`` when qiskit-machine-learning is
    not installed.
    """
    from pipeline.seed import seed_everything
    from sklearn.svm import SVC

    seed_everything(int(getattr(cfg, "seed", 6)))

    try:
        from qiskit_machine_learning.kernels import FidelityQuantumKernel
    except ImportError as exc:
        raise RuntimeError(
            "quantum_kernel requires qiskit-machine-learning and qiskit "
            "(requires=('qiskit-machine-learning', 'qiskit')). Install with "
            "`pip install qiskit qiskit-machine-learning`."
        ) from exc

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)

    n_train = int(getattr(cfg, "qk_n_train", 400))
    shots = int(getattr(cfg, "qk_shots", 1024))
    seed = int(getattr(cfg, "seed", 6))

    n = min(n_train, X_train.shape[0])
    idx = np.random.choice(X_train.shape[0], size=n, replace=False)
    X_sub = X_train[idx]
    y_sub = y_train[idx]

    kernel = FidelityQuantumKernel()
    K_train = np.asarray(kernel.evaluate(x_vec=X_sub, y_vec=X_sub), dtype=np.float64)
    K_val = np.asarray(kernel.evaluate(x_vec=X_val, y_vec=X_sub), dtype=np.float64)

    model = SVC(kernel="precomputed", probability=True, random_state=seed)
    model.fit(K_train, y_sub)

    val_probs = np.asarray(model.predict_proba(K_val)[:, 1], dtype=np.float64)
    test_probs = None
    if X_test is not None and len(X_test) > 0:
        K_test = np.asarray(
            kernel.evaluate(
                x_vec=np.asarray(X_test, dtype=np.float64), y_vec=X_sub
            ),
            dtype=np.float64,
        )
        test_probs = np.asarray(model.predict_proba(K_test)[:, 1], dtype=np.float64)

    return {
        "val_probs": val_probs,
        "test_probs": test_probs,
        "history": {},
        "params": {
            "qk_n_train": n,
            "qk_shots": shots,
            "kernel": "FidelityQuantumKernel",
        },
        "epochs_trained": 1,
    }


def _count_quantum_kernel(cfg):
    """Data-dependent (Gram-matrix size); reported post-fit."""
    return -1


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODELS: dict[str, ModelSpec] = {
    "vqc": ModelSpec(
        name="vqc",
        kind="vqc",
        train=_train_vqc,
        param_count=_count_vqc,
        requires=("pennylane", "torch"),
    ),
    "mlp": ModelSpec(
        name="mlp",
        kind="torch",
        train=_train_mlp,
        param_count=_count_mlp,
        requires=("torch",),
    ),
    "logreg": ModelSpec(
        name="logreg",
        kind="sklearn",
        train=_train_logreg,
        param_count=_count_logreg,
        requires=("scikit-learn",),
    ),
    "rbf_svm": ModelSpec(
        name="rbf_svm",
        kind="sklearn",
        train=_train_rbf_svm,
        param_count=_count_rbf_svm,
        requires=("scikit-learn",),
    ),
    "random_forest": ModelSpec(
        name="random_forest",
        kind="sklearn",
        train=_train_random_forest,
        param_count=_count_random_forest,
        requires=("scikit-learn",),
    ),
    "knn": ModelSpec(
        name="knn",
        kind="sklearn",
        train=_train_knn,
        param_count=_count_knn,
        requires=("scikit-learn",),
    ),
    "mlp_deep": ModelSpec(
        name="mlp_deep",
        kind="torch",
        train=_train_mlp_deep,
        param_count=_count_mlp_deep,
        requires=("torch",),
    ),
    "vqc_angle": ModelSpec(
        name="vqc_angle",
        kind="vqc",
        train=_train_vqc_angle,
        param_count=_count_vqc_angle,
        requires=("pennylane", "torch"),
    ),
    "vqc_he": ModelSpec(
        name="vqc_he",
        kind="vqc",
        train=_train_vqc_he,
        param_count=_count_vqc_he,
        requires=("pennylane", "torch"),
    ),
    "quantum_kernel": ModelSpec(
        name="quantum_kernel",
        kind="quantum_kernel",
        train=_train_quantum_kernel,
        param_count=_count_quantum_kernel,
        requires=("qiskit-machine-learning", "qiskit"),
    ),
}


def get_model(name: str) -> ModelSpec:
    """Return the ``ModelSpec`` for *name*, raising ``KeyError`` otherwise.

    Args:
        name: Model name (registry key).

    Returns:
        The matching :class:`ModelSpec`.

    Raises:
        KeyError: If *name* is not in the registry, listing the available
            models in the error message.
    """
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(
            f"Unknown model {name!r}. Available: {', '.join(available_models())}"
        ) from None


def available_models() -> list[str]:
    """Return the registry keys in benchmark CSV row order."""
    return list(MODELS.keys())
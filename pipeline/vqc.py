"""Variational Quantum Classifier — training loop, combined loss, prediction.

Task 9 of the notebook → package rewrite. Implements the VQC training
pipeline on top of the ansatz from ``pipeline.ansatz``:

* ``get_pennylane_device`` — device resolution with the locked fallback
  chain ``lightning.gpu`` → ``lightning.qubit`` → ``default.qubit``.
  ``adjoint`` is only ever used on lightning devices; ``default.qubit``
  always uses ``parameter-shift`` (never adjoint there).
* ``combined_loss`` — the FIXED single differentiable loss combining base
  MSE, MixUp MSE and CORAL (bug fixes #1/#2 of the notebook review). The
  gradient is computed with ``qml.grad`` on the combined expression, so
  every term contributes to every parameter update. Acceptance contract:
  ``tests/test_vqc.py::test_coral_mixup_gradient_flow``.
* ``train_vqc`` — full Adam training loop with cosine LR + warm-up and
  early stopping, saving the best parameters and validation/test
  probabilities to ``cfg.results_dir``.
* ``vqc_predict`` / ``batch_loss`` / ``lr_cosine_warmup`` — inference and
  scheduling helpers.

Import policy
-------------
Only the standard library and NumPy are imported at module level.
``pennylane`` and ``torch`` are imported lazily inside the functions that
need them (with a clear ``RuntimeError`` when missing), so
``import pipeline.vqc`` succeeds even on a machine without PennyLane /
PyTorch (e.g. the local development box).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

SEED = 6  # project-wide random seed (AGENTS.md global rule)

__all__ = [
    "SEED",
    "get_pennylane_device",
    "batch_loss",
    "combined_loss",
    "train_vqc",
    "vqc_predict",
    "lr_cosine_warmup",
]


# ---------------------------------------------------------------------------
# Lazy heavy imports (stdlib + numpy only at module level)
# ---------------------------------------------------------------------------


def _import_pennylane():
    """Import PennyLane lazily, raising a clear RuntimeError when missing."""
    try:
        import pennylane as qml

        return qml
    except ImportError as exc:
        raise RuntimeError(
            "pipeline.vqc requires PennyLane (pip install pennylane==0.45.1). "
            "It is imported lazily so the module imports with stdlib + numpy only."
        ) from exc


def _import_torch():
    """Import PyTorch lazily, raising a clear RuntimeError when missing."""
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError(
            "pipeline.vqc requires PyTorch for the MixUp augmentation "
            "(pip install torch). It is imported lazily so the module imports "
            "with stdlib + numpy only."
        ) from exc


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization (safety net for amplitude encoding)."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def get_pennylane_device(cfg: Any) -> tuple[Any, str]:
    """Resolve the PennyLane device and differentiation method for training.

    Fallback chain (LOCKED by the rewrite plan): ``lightning.gpu`` →
    ``lightning.qubit`` → ``default.qubit``. A warning is logged on every
    fallback. ``adjoint`` is only used on lightning devices; ``default.qubit``
    always uses ``parameter-shift`` (adjoint is never used there, even if
    ``cfg.diff_method`` requests it).

    Replicates the rule of ``pipeline.ansatz._resolve_device`` with explicit
    per-fallback warnings (the ansatz helper swallows the fallback details).

    Args:
        cfg: Configuration object. Recognised fields: ``n_qubits``,
            ``diff_method``.

    Returns:
        ``(device, diff_method)`` tuple ready for ``build_vqc_circuit``.
    """
    qml = _import_pennylane()

    n_qubits = int(getattr(cfg, "n_qubits", 6))
    requested = str(getattr(cfg, "diff_method", "adjoint"))

    for name in ("lightning.gpu", "lightning.qubit"):
        try:
            dev = qml.device(name, wires=n_qubits)
        except Exception as exc:  # noqa: BLE001 - device may be unavailable
            warnings.warn(
                f"PennyLane device {name!r} unavailable ({exc}); falling back "
                "to the next device in the chain.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        diff = requested if requested in ("adjoint", "parameter-shift") else "adjoint"
        return dev, diff

    dev = qml.device("default.qubit", wires=n_qubits)
    if requested != "parameter-shift":
        warnings.warn(
            "default.qubit does not support adjoint differentiation; using "
            "parameter-shift.",
            RuntimeWarning,
            stacklevel=2,
        )
    return dev, "parameter-shift"


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------


def batch_loss(params, Xb, yb, circuit) -> float:
    """Mean squared error between circuit outputs and ±1 targets.

    Args:
        params: Flat trainable parameters (layout per ``split_params``).
        Xb: Batch of L2-normalised features, shape ``(B, 64)``.
        yb: Batch of targets in {+1, −1}, shape ``(B,)``.
        circuit: A ``qml.QNode`` returning ``⟨Z₀⟩`` per sample.

    Returns:
        Scalar MSE ``mean((⟨Z₀⟩ - yb)^2)``.
    """
    preds = np.asarray(circuit(params, Xb))
    yb = np.asarray(yb, dtype=np.float64)
    return float(np.mean((preds - yb) ** 2))


def combined_loss(
    params,
    Xb,
    yb,
    X_test_pool,
    circuit=None,
    lambda_coral: float = 0.1,
    mixup_alpha: float = 0.2,
) -> tuple[Any, Any]:
    """Single differentiable loss combining base MSE, MixUp MSE and CORAL.

    Bug fixes #1/#2 of the notebook review: the notebook accumulated CORAL
    outputs per batch and added the loss AFTER the parameter update (zero
    gradient), and the MixUp loss was added without recomputing the gradient.
    Here the loss is a SINGLE differentiable expression

    ``L = MSE(circuit(params, Xb), yb)
         + MSE(circuit(params, X_mixed), y_mixed)
         + lambda_coral * CORAL(circuit(params, Xb), circuit(params, X_tgt))``

    and ``grads`` is computed with ``qml.grad`` on that combined expression,
    so every term contributes to every parameter update. No ``float()`` /
    ``np.array()`` / ``.detach()`` / ``.item()`` conversions appear inside
    the differentiable path (the path traced by ``qml.grad``).

    The MixUp features are produced by ``pipeline.vae.inter_domain_mixup``
    (torch-based; the mixed features are data only — no parameters involved,
    so converting them to NumPy for the circuit does not break autograd).
    The CORAL term mirrors ``pipeline.vae.coral_loss`` but is computed with
    plain scalar accumulation over per-sample circuit outputs, so it stays
    differentiable w.r.t. the circuit parameters through ``qml.grad`` (the
    torch version cannot be applied to autograd-tracked circuit outputs).

    Args:
        params: Trainable parameters. When ``circuit`` is ``None`` this is
            the 54-param rot block, shape ``(3, 6, 3)``; otherwise the flat
            ``split_params`` layout (62 params by default).
        Xb: Source batch of L2-normalised features, shape ``(B, 64)``.
        yb: Source batch of targets in {+1, −1}, shape ``(B,)``.
        X_test_pool: Target-domain (test) feature pool, shape ``(N, 64)``.
            A batch of size ``B`` is sampled from it when ``N != B``.
        circuit: A ``qml.QNode`` returning ``⟨Z₀⟩`` per sample. When ``None``
            the 54-param default ``build_vqc_circuit(6, 3, use_scale=False,
            use_meas_basis=False)`` is built internally.
        lambda_coral: Weight of the CORAL term (default 0.1).
        mixup_alpha: Beta distribution parameter for MixUp (default 0.2).

    Returns:
        ``(loss, grads)`` — the combined loss scalar and its gradient w.r.t.
        ``params`` (same shape as ``params``).
    """
    qml = _import_pennylane()
    from pennylane import numpy as pnp

    torch = _import_torch()

    from pipeline.ansatz import build_vqc_circuit
    from pipeline.vae import inter_domain_mixup

    if circuit is None:
        circuit = build_vqc_circuit(6, 3, use_scale=False, use_meas_basis=False)

    Xb = _l2_normalize(Xb)
    yb = np.asarray(yb, dtype=np.float64)
    X_test_pool = _l2_normalize(X_test_pool)
    params = np.asarray(params, dtype=np.float64)

    n_batch = Xb.shape[0]
    if n_batch == 0:
        raise ValueError("Xb must contain at least one sample.")
    if X_test_pool.shape[0] == 0:
        raise ValueError(
            "X_test_pool must contain at least one sample for CORAL/MixUp."
        )
    if X_test_pool.shape[0] != n_batch:
        replace = X_test_pool.shape[0] < n_batch
        idx = np.random.choice(X_test_pool.shape[0], size=n_batch, replace=replace)
        X_test_pool = X_test_pool[idx]

    # MixUp on the data (torch-based; data only, no parameters involved).
    Xb_t = torch.as_tensor(Xb, dtype=torch.float32)
    Xt_t = torch.as_tensor(X_test_pool, dtype=torch.float32)
    yb_t = torch.as_tensor(yb, dtype=torch.float32)
    X_mixed_t, y_mixed_t = inter_domain_mixup(Xb_t, Xt_t, yb_t, alpha=mixup_alpha)
    X_mixed = _l2_normalize(X_mixed_t.numpy())
    y_mixed = y_mixed_t.numpy().astype(np.float64)

    def _coral(preds_src, preds_tgt):
        """CORAL on 1-D batch outputs (mirrors ``vae.coral_loss``).

        ``L = 0.25 * (Var(preds_src) - Var(preds_tgt))^2`` — second-order
        statistics of the circuit outputs in the source vs target domain.
        ``preds_src`` / ``preds_tgt`` are Python lists of autograd-tracked
        scalars (one per sample); the mean and covariance are accumulated
        with plain scalar ops so the term stays differentiable w.r.t. the
        circuit parameters through ``qml.grad``.
        """
        n = max(n_batch - 1, 1)

        mean_src = 0.0
        for p in preds_src:
            mean_src = mean_src + p
        mean_src = mean_src / len(preds_src)

        cov_src = 0.0
        for p in preds_src:
            cov_src = cov_src + (p - mean_src) ** 2
        cov_src = cov_src / n

        mean_tgt = 0.0
        for p in preds_tgt:
            mean_tgt = mean_tgt + p
        mean_tgt = mean_tgt / len(preds_tgt)

        cov_tgt = 0.0
        for p in preds_tgt:
            cov_tgt = cov_tgt + (p - mean_tgt) ** 2
        cov_tgt = cov_tgt / n

        return 0.25 * (cov_src - cov_tgt) ** 2

    def _loss(p):
        p_flat = pnp.reshape(p, -1)

        # Per-sample circuit calls: adjoint differentiation does NOT support
        # parameter broadcasting (PennyLane issue #4180), so we loop over the
        # batch exactly like the notebook's ``batch_loss``.
        preds_base = [circuit(p_flat, x) for x in Xb]
        preds_mixed = [circuit(p_flat, x) for x in X_mixed]
        preds_tgt = [circuit(p_flat, x) for x in X_test_pool]

        base_mse = 0.0
        for pred, y in zip(preds_base, yb):
            base_mse = base_mse + (pred - y) ** 2
        base_mse = base_mse / n_batch

        mixup_mse = 0.0
        for pred, y in zip(preds_mixed, y_mixed):
            mixup_mse = mixup_mse + (pred - y) ** 2
        mixup_mse = mixup_mse / n_batch

        coral = _coral(preds_base, preds_tgt)
        return base_mse + mixup_mse + lambda_coral * coral

    # PennyLane 0.45 autograd: qml.grad returns () for plain numpy args;
    # wrap as a trainable tensor or training crashes on an empty gradient.
    params_t = qml.numpy.array(params, requires_grad=True)
    loss = _loss(params_t)
    grads = qml.grad(_loss)(params_t)
    return loss, grads


# ---------------------------------------------------------------------------
# Inference and learning-rate schedule
# ---------------------------------------------------------------------------


def vqc_predict(X, params, circuit) -> np.ndarray:
    """Predict pneumonia probabilities ``(1 + ⟨Z₀⟩) / 2`` for every row of X.

    Args:
        X: Feature matrix, shape ``(N, 64)`` (or a single row, shape
            ``(64,)``). Rows are L2-normalised internally if not already.
        params: Flat trainable parameters (layout per ``split_params``).
        circuit: A ``qml.QNode`` returning ``⟨Z₀⟩`` per sample.

    Returns:
        Probabilities in ``[0, 1]``, shape ``(N,)`` (or a scalar when ``X``
        is a single row).
    """
    X = np.asarray(X, dtype=np.float64)
    single = X.ndim == 1
    if single:
        X = X.reshape(1, -1)
    X_norm = _l2_normalize(X)
    # PennyLane 0.45 cannot batch (B, 64) inputs with per-sample gate
    # arguments (broadcast-expand raises); evaluate per sample instead.
    z = np.array([float(circuit(params, row)) for row in X_norm])
    probs = (1.0 + z) / 2.0
    if single:
        probs = probs[0]
    return probs


def lr_cosine_warmup(
    epoch: int, lr0: float, lr_min: float, warmup: int, total: int
) -> float:
    """Linear warm-up followed by cosine annealing to ``lr_min``.

    For ``epoch < warmup`` the learning rate grows linearly from
    ``lr0 / warmup`` to ``lr0``. From ``epoch == warmup`` onward it anneals
    with a cosine schedule from ``lr0`` down to ``lr_min`` at the final epoch.

    Args:
        epoch: Zero-based epoch index.
        lr0: Peak learning rate (reached at the end of the warm-up).
        lr_min: Final learning rate after cosine annealing.
        warmup: Number of warm-up epochs.
        total: Total number of epochs.

    Returns:
        The learning rate for ``epoch``.
    """
    epoch = int(epoch)
    warmup = int(warmup)
    total = int(total)

    if warmup > 0 and epoch < warmup:
        return float(lr0 * (epoch + 1) / warmup)

    denom = max(total - warmup, 1)
    progress = min(max((epoch - warmup) / denom, 0.0), 1.0)
    return float(lr_min + 0.5 * (lr0 - lr_min) * (1.0 + np.cos(np.pi * progress)))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_vqc(
    cfg: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train the VQC with Adam, cosine LR + warm-up and early stopping.

    Builds the circuit per ``cfg`` (``n_qubits=6``, ``n_layers=3``,
    ``use_learnable_scale``, ``use_measurement_basis`` → 62 trainable
    parameters by default), initialises the parameters in the flat
    ``split_params`` layout ``[rot, scale?, meas?]``, and runs a full Adam
    training loop with the combined CORAL + MixUp loss from
    :func:`combined_loss`. The learning rate follows
    :func:`lr_cosine_warmup`; training stops early when the validation loss
    does not improve for ``cfg.early_stopping_patience`` epochs. The best
    parameters are selected by validation balanced accuracy.

    Artifacts saved under ``cfg.results_dir`` (relative to the repo root):

    * ``vqc_best_params.npy`` — flat best parameter vector.
    * ``vqc_history.json`` — per-epoch ``train_loss`` / ``val_loss`` /
      ``val_bal_acc`` / ``lr``.
    * ``vqc_val_probs.npy`` — validation probabilities from the best params.
    * ``vqc_test_probs.npy`` — test probabilities (only when ``X_test`` is
      given).

    Args:
        cfg: Configuration object. Recognised fields: ``seed``, ``n_qubits``,
            ``n_layers``, ``use_learnable_scale``, ``use_measurement_basis``,
            ``diff_method``, ``batch_size``, ``learning_rate``, ``lr_min``,
            ``warmup_epochs``, ``epochs``, ``early_stopping_patience``,
            ``lambda_coral``, ``mixup_alpha``, ``results_dir``.
        X_train: Source-domain training features, shape ``(N, 64)``
            (L2-normalised; normalised again internally as a safety net).
        y_train: Training labels (0/1 or ±1), shape ``(N,)``.
        X_val: Validation features, shape ``(N, 64)``.
        y_val: Validation labels (0/1 or ±1), shape ``(N,)``.
        X_test: Optional target-domain (test) features, shape ``(N, 64)``.
            Used as the CORAL/MixUp target pool and for the saved test
            probabilities.
        y_test: Accepted for API symmetry with the evaluation stage (T11);
            not used during training.

    Returns:
        Dict with keys ``best_params`` (flat ndarray), ``history`` (dict),
        ``val_probs`` (ndarray), ``test_probs`` (ndarray or ``None``) and
        ``epochs_trained`` (int).
    """
    from pipeline.seed import seed_everything

    seed_everything(int(getattr(cfg, "seed", 6)))

    from pipeline.ansatz import build_vqc_circuit, count_params
    from sklearn.metrics import balanced_accuracy_score

    # --- hyperparameters from cfg (getattr fallbacks for robustness) ---
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
    lambda_coral = float(getattr(cfg, "lambda_coral", 0.1))
    mixup_alpha = float(getattr(cfg, "mixup_alpha", 0.2))
    results_dir = Path(str(getattr(cfg, "results_dir", "results")))

    # --- data prep (L2-normalise; labels → ±1 for the MSE loss) ---
    X_train = _l2_normalize(X_train)
    X_val = _l2_normalize(X_val)
    X_test_arr = None if X_test is None else _l2_normalize(X_test)

    y_train = np.asarray(y_train, dtype=np.float64)
    y_val = np.asarray(y_val, dtype=np.float64)
    y_train_pm1 = np.where(y_train > 0.5, 1.0, -1.0)
    y_val_pm1 = np.where(y_val > 0.5, 1.0, -1.0)
    y_val_01 = (y_val_pm1 > 0).astype(int)

    n_train = X_train.shape[0]

    # --- device + circuit ---
    dev, diff_method = get_pennylane_device(cfg)
    dev_name = (
        getattr(dev, "short_name", None)
        or getattr(dev, "name", "")
        or str(dev)
    )
    print(f"VQC training on {dev_name} (diff_method={diff_method})")
    circuit = build_vqc_circuit(
        n_qubits,
        n_layers,
        use_scale=use_scale,
        use_meas_basis=use_meas,
        device=dev,
        diff_method=diff_method,
    )

    n_params = count_params(n_qubits, n_layers, use_scale, use_meas)

    # --- parameter init (flat split_params layout: [rot, scale?, meas?]) ---
    rot_init = np.random.uniform(0.0, 2.0 * np.pi, size=(n_layers * n_qubits * 3,))
    parts = [rot_init]
    if use_scale:
        parts.append(np.ones(n_qubits))
    if use_meas:
        parts.append(np.zeros(2))
    params = np.concatenate(parts)
    if params.shape != (n_params,):
        raise ValueError(
            f"parameter layout mismatch: built {params.shape}, expected ({n_params},)"
        )

    # --- Adam state ---
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    beta1, beta2 = 0.9, 0.999
    eps = 1e-8
    t = 0

    # --- target-domain pool for CORAL/MixUp (test, else validation) ---
    X_test_pool = X_test_arr if X_test_arr is not None else X_val

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

            loss, grads = combined_loss(
                params,
                Xb,
                yb,
                X_test_pool,
                circuit=circuit,
                lambda_coral=lambda_coral,
                mixup_alpha=mixup_alpha,
            )

            # Adam update (manual — params are NumPy arrays, not torch tensors).
            t += 1
            m = beta1 * m + (1.0 - beta1) * grads
            v = beta2 * v + (1.0 - beta2) * grads**2
            m_hat = m / (1.0 - beta1**t)
            v_hat = v / (1.0 - beta2**t)
            params = params - lr * m_hat / (np.sqrt(v_hat) + eps)

            epoch_loss += float(loss)
            n_batches += 1

        # --- validation (forward pass only) ---
        z_val = np.array([float(circuit(params, row)) for row in X_val])
        val_loss = float(np.mean((z_val - y_val_pm1) ** 2))
        val_probs_epoch = (1.0 + z_val) / 2.0
        val_bal_acc = float(
            balanced_accuracy_score(y_val_01, (val_probs_epoch > 0.5).astype(int))
        )

        history["train_loss"].append(epoch_loss / max(n_batches, 1))
        history["val_loss"].append(val_loss)
        history["val_bal_acc"].append(val_bal_acc)
        history["lr"].append(lr)

        # Best-model selection on validation balanced accuracy.
        if val_bal_acc > best_val_bal_acc:
            best_val_bal_acc = val_bal_acc
            best_params = params.copy()
            best_epoch = epoch + 1

        # Early stopping on validation loss.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"  VQC early stopping at epoch {epoch + 1} "
                    f"(no val-loss improvement for {patience} epochs)"
                )
                break

        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"  VQC Epoch {epoch + 1}/{epochs} | "
                f"Train: {history['train_loss'][-1]:.4f} | "
                f"Val: {history['val_loss'][-1]:.4f} | "
                f"BalAcc: {history['val_bal_acc'][-1]:.4f} | "
                f"LR: {lr:.2e}"
            )

    # --- final predictions with the best parameters ---
    val_probs = vqc_predict(X_val, best_params, circuit)
    test_probs = None
    if X_test_arr is not None:
        test_probs = vqc_predict(X_test_arr, best_params, circuit)

    # --- save artifacts ---
    results_dir.mkdir(parents=True, exist_ok=True)
    np.save(results_dir / "vqc_best_params.npy", best_params)
    with open(results_dir / "vqc_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    np.save(results_dir / "vqc_val_probs.npy", val_probs)
    if test_probs is not None:
        np.save(results_dir / "vqc_test_probs.npy", test_probs)

    print(
        f"  VQC training finished: {epochs_trained} epochs, "
        f"best val balanced accuracy {best_val_bal_acc:.4f} at epoch {best_epoch}"
    )

    return {
        "best_params": best_params,
        "history": history,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "epochs_trained": epochs_trained,
    }
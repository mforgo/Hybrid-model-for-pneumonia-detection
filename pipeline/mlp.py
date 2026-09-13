"""Classical MLP baseline with weighted BCE loss (Task 10).

Trains a small fully connected classifier on the 64-dim L2-normalized VAE
features as the classical reference for the hybrid quantum-classical
comparison (AGENTS.md, "Classical MLP Baseline"):

* ``Linear(64 -> 32) -> ReLU -> Dropout(0.3) -> Linear(32 -> 1)`` — outputs
  **raw logits** (no sigmoid inside the model); the loss is
  ``BCEWithLogitsLoss`` with ``pos_weight = n_neg / n_pos`` computed from the
  training labels. Class imbalance is handled by the weight, **not** by a
  WeightedRandomSampler (locked design decision, rewrite plan T10).
* Cosine LR schedule with linear warm-up: ``cfg.warmup_epochs`` epochs of
  linear ramp, then ``CosineAnnealingLR`` down to ``cfg.lr_min``.
* Early stopping on validation loss with ``cfg.early_stopping_patience``
  (default 10 per AGENTS.md; the notebook used 3).

Import policy
-------------
Only the standard library and NumPy are imported at module level. ``torch``
is imported lazily inside the functions that need it, so ``import
pipeline.mlp`` and ``from pipeline.mlp import MLP, compute_pos_weight``
succeed even on a machine without PyTorch (e.g. the local development box).
``compute_pos_weight`` is pure NumPy so that
``tests/test_mlp.py::test_mlp_pos_weight`` passes without torch. The
acceptance tests run on the GPU server where torch is installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "MLP",
    "train_mlp",
    "predict_mlp",
    "compute_pos_weight",
]


# ---------------------------------------------------------------------------
# Module-level helpers (no torch at import time)
# ---------------------------------------------------------------------------


def _nn_module_base() -> type:
    """Return ``torch.nn.Module`` if torch is available, else ``object``.

    Keeps the module importable without PyTorch (dev machine). The MLP class
    is only instantiated at runtime on the GPU server where torch exists, at
    which point the base class is a genuine ``nn.Module``.
    """
    try:
        import torch.nn as nn

        return nn.Module
    except Exception:  # noqa: BLE001 - torch optional at import time
        return object


def _require_torch() -> Any:
    """Import torch or raise a clear ``RuntimeError``.

    Heavy imports are lazy (call-time only); when torch is missing the error
    message explains what to install instead of surfacing a bare
    ``ImportError``.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dev machine only
        raise RuntimeError(
            "pipeline.mlp requires PyTorch at call time, but torch is not "
            "installed. Install the project dependencies (see requirements.txt)."
        ) from exc
    return torch


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MLP(_nn_module_base()):  # type: ignore[misc, valid-type]
    """Small fully connected baseline classifier (raw logits).

    Architecture (locked by the rewrite plan / AGENTS.md):

    ``Linear(input_dim -> hidden) -> ReLU -> Dropout(dropout) ->
    Linear(hidden -> 1)``

    The output is a **raw logit** of shape ``(B, 1)`` — there is no sigmoid
    inside the model. Train with ``BCEWithLogitsLoss`` (see
    :func:`train_mlp`); apply the sigmoid at prediction time via
    :func:`predict_mlp`.

    Args:
        input_dim: Number of input features (default 64 — the VAE latent
            dimension).
        hidden: Hidden layer width (default 32).
        dropout: Dropout probability after the ReLU (default 0.3).
    """

    def __init__(
        self,
        input_dim: int = 64,
        hidden: int = 32,
        dropout: float = 0.3,
    ) -> None:
        import torch.nn as nn

        super().__init__()
        self.input_dim = input_dim
        self.hidden = hidden
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Any) -> Any:
        """Forward pass: raw logits of shape ``(B, 1)``.

        Args:
            x: Input features, shape ``(B, input_dim)``.

        Returns:
            Raw logits, shape ``(B, 1)`` (no sigmoid).
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# Loss helper
# ---------------------------------------------------------------------------


def compute_pos_weight(y_train: Any) -> float:
    """Compute the ``BCEWithLogitsLoss`` pos_weight from training labels.

    ``pos_weight = n_negative / n_positive`` — the negative / positive class
    ratio, guarding against an all-negative label vector (denominator
    clamped to 1). Pure NumPy so it works without PyTorch installed.

    Args:
        y_train: Binary labels (array-like, 0/1).

    Returns:
        The positive-class weight as a plain ``float``.
    """
    y = np.asarray(y_train, dtype=np.float64)
    n_pos = float(y.sum())
    n_neg = float(y.size - n_pos)
    return n_neg / max(n_pos, 1.0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_mlp(
    cfg: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    device: Any = None,
) -> dict[str, Any]:
    """Train the classical MLP baseline and return best state + probabilities.

    Full training loop (locked design, rewrite plan T10):

    * Loss: ``BCEWithLogitsLoss`` with
      ``pos_weight = compute_pos_weight(y_train)``.
    * Optimizer: Adam, ``lr = cfg.learning_rate`` (1e-3).
    * LR schedule: linear warm-up for ``cfg.warmup_epochs`` epochs, then
      ``CosineAnnealingLR`` down to ``cfg.lr_min``.
    * Early stopping: validation loss, patience
      ``cfg.early_stopping_patience`` (default 10).
    * Reproducibility: ``seed_everything(cfg.seed)``; DataLoader shuffling
      uses a seeded ``torch.Generator``.

    Artifacts saved under ``cfg.results_dir`` (relative to repo root — never
    ``/content/*``):

    * ``mlp_best.pt`` — best state dict (CPU, device-independent).
    * ``mlp_history.json`` — per-epoch ``train_loss`` / ``val_loss`` / ``lr``.
    * ``mlp_val_probs.npy`` — sigmoid probabilities on the validation set.
    * ``mlp_test_probs.npy`` — sigmoid probabilities on the test set (only
      when ``X_test`` is provided).

    Args:
        cfg: Configuration object. Recognised fields: ``mlp_hidden``,
            ``mlp_dropout``, ``batch_size``, ``learning_rate``, ``lr_min``,
            ``warmup_epochs``, ``epochs``, ``early_stopping_patience``,
            ``seed``, ``device``, ``results_dir``.
        X_train: Training features, shape ``(N, 64)``.
        y_train: Training labels, shape ``(N,)`` (0/1).
        X_val: Validation features, shape ``(N, 64)``.
        y_val: Validation labels, shape ``(N,)`` (0/1).
        X_test: Optional test features, shape ``(N, 64)``.
        y_test: Optional test labels — accepted for API symmetry with the
            evaluation stage (T11); **not** used for any training decision
            (threshold selection happens later on the validation set only).
        device: Optional device override (string or ``torch.device``). When
            ``None``, ``cfg.device`` is used (``"auto"`` → CUDA if available
            else CPU).

    Returns:
        Dict with keys ``best_state`` (CPU state dict), ``history``
        (per-epoch ``train_loss``/``val_loss``/``lr`` lists), ``val_probs``
        (``(N,)`` numpy), ``test_probs`` (``(N,)`` numpy or ``None``) and
        ``epochs_trained`` (int).
    """
    torch = _require_torch()
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    from pipeline.seed import get_torch_generator, seed_everything

    seed_everything(int(getattr(cfg, "seed", 6)))

    # --- hyperparameters from cfg (getattr fallbacks for robustness) ---
    hidden = int(getattr(cfg, "mlp_hidden", 32))
    dropout = float(getattr(cfg, "mlp_dropout", 0.3))
    batch_size = int(getattr(cfg, "batch_size", 16))
    lr = float(getattr(cfg, "learning_rate", 1e-3))
    lr_min = float(getattr(cfg, "lr_min", 1e-5))
    warmup_epochs = int(getattr(cfg, "warmup_epochs", 3))
    epochs = int(getattr(cfg, "epochs", 50))
    patience = int(getattr(cfg, "early_stopping_patience", 10))
    seed = int(getattr(cfg, "seed", 6))
    results_dir = Path(getattr(cfg, "results_dir", "results"))

    # --- device resolution: explicit arg > cfg.device ("auto" -> cuda/cpu) ---
    if device is None:
        device = str(getattr(cfg, "device", "auto"))
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_val = np.asarray(y_val, dtype=np.float32)

    results_dir.mkdir(parents=True, exist_ok=True)

    # --- tensors / loader (seeded shuffle for reproducibility) ---
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

    # --- model / loss / optimizer / scheduler ---
    model = MLP(input_dim=X_train.shape[1], hidden=hidden, dropout=dropout)
    model.to(device)

    pos_weight = torch.tensor(
        compute_pos_weight(y_train), dtype=torch.float32, device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=lr_min
    )

    print(
        f"Training MLP ({X_train.shape[1]} -> {hidden} -> 1, dropout={dropout}) "
        f"for up to {epochs} epochs on {device}"
    )
    print(
        f"  pos_weight={float(pos_weight):.4f} | lr={lr} -> {lr_min} "
        f"(warmup {warmup_epochs}) | patience={patience}"
    )

    best_val_loss = float("inf")
    best_state: dict[str, Any] | None = None
    patience_counter = 0
    epochs_trained = 0
    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "lr": [],
    }

    model.train()
    for epoch in range(epochs):
        # LR schedule: linear warm-up, then cosine annealing.
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

        # Validation loss (early-stopping signal).
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
                    f"Early stopping at epoch {epoch + 1} "
                    f"(no val improvement for {patience} epochs)"
                )
                break

        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            print(
                f"  MLP Epoch {epoch + 1}/{epochs} | "
                f"train_loss={history['train_loss'][-1]:.4f} | "
                f"val_loss={val_loss:.4f} | lr={current_lr:.2e}"
            )

    if best_state is None:
        raise ValueError(
            "train_mlp: no epochs completed; cannot produce a best model."
        )

    # --- final probabilities from the BEST model ---
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

    # --- save artifacts ---
    torch.save(best_state, results_dir / "mlp_best.pt")
    (results_dir / "mlp_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    np.save(results_dir / "mlp_val_probs.npy", val_probs)
    if test_probs is not None:
        np.save(results_dir / "mlp_test_probs.npy", test_probs)

    print(
        f"  Best val_loss={best_val_loss:.4f} after {epochs_trained} epochs; "
        f"saved to {results_dir}"
    )

    return {
        "best_state": best_state,
        "history": history,
        "val_probs": val_probs,
        "test_probs": test_probs,
        "epochs_trained": epochs_trained,
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict_mlp(model: Any, X: np.ndarray, device: Any = None) -> np.ndarray:
    """Predict pneumonia probabilities with a trained :class:`MLP`.

    Applies the sigmoid to the raw logits (the model itself outputs logits
    only) and returns a flat ``(N,)`` numpy array of probabilities.

    Args:
        model: A trained :class:`MLP` instance.
        X: Input features, shape ``(N, 64)``.
        device: Optional device override (string or ``torch.device``). When
            ``None``, the device of the model's first parameter is used (no
            unnecessary copies).

    Returns:
        Probabilities, shape ``(N,)``.
    """
    torch = _require_torch()

    X = np.asarray(X, dtype=np.float32)
    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float32).to(device)
        logits = model(x_t).squeeze(-1)
        probs = torch.sigmoid(logits).cpu().numpy()
    return probs
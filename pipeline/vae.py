"""Supervised VAE dimensionality reduction with CORAL and MixUp (Task 7).

Reduces 768-dim ConvNeXt-Tiny features to 64-dim L2-normalized latent
vectors, matching the Hilbert space of a 6-qubit register (2^6 = 64) for
amplitude encoding.

This module fixes the notebook's CORAL/MixUp gradient bugs (rewrite plan,
"Confirmed Bugs" #1/#2 and Task 7):

* **CORAL is differentiable.** The loss is computed on the *latent* features
  of the current source batch and a target (test-domain) pool batch, and is
  added to the computational graph *before* ``backward()``. Both source and
  target latents flow gradients to the encoder — no ``.detach()`` /
  ``.numpy()`` / ``float()`` conversions inside the graph.
* **MixUp contributes gradients.** Source and target latent batches are mixed
  with per-sample ``lambda ~ Beta(alpha, alpha)`` and the mixed latents are
  classified with soft labels, producing a real gradient path through the
  classifier and encoder.

Import policy
-------------
Only the standard library and NumPy are imported at module level. ``torch``,
``sklearn`` and ``joblib`` are imported lazily inside the functions that need
them, so ``import pipeline.vae`` succeeds even on a machine without PyTorch
(e.g. the local development box). The acceptance test
``tests/test_vae.py::test_supervised_vae_loss_backward`` runs on the GPU
server where torch is installed.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
import warnings
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "VAE",
    "SupervisedVAE",
    "supervised_vae_loss",
    "compute_pos_weight",
    "coral_loss",
    "inter_domain_mixup",
    "fit_vae_pipeline",
]


# ---------------------------------------------------------------------------
# Module-level helpers (no torch at import time)
# ---------------------------------------------------------------------------


def _nn_module_base() -> type:
    """Return ``torch.nn.Module`` if torch is available, else ``object``.

    Keeps the module importable without PyTorch (dev machine). The VAE
    classes are only instantiated at runtime on the GPU server where torch
    exists, at which point the base class is a genuine ``nn.Module``.
    """
    try:
        import torch.nn as nn

        return nn.Module
    except Exception:  # noqa: BLE001 - torch optional at import time
        return object


def _git_sha() -> str:
    """Return the current git HEAD sha, or ``"unknown"`` if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.decode("utf-8").strip()
    except Exception:  # noqa: BLE001 - git may be absent
        return "unknown"


def _config_hash(cfg: Any) -> str:
    """sha256 of the config snapshot (prefers ``pipeline.config.config_hash``).

    Falls back to hashing the dataclass fields directly when
    ``pipeline.config`` cannot be imported (e.g. pyyaml missing on the dev
    machine). Any hyperparameter change invalidates the cache.
    """
    try:
        from pipeline.config import config_hash

        return config_hash(cfg)
    except Exception:  # noqa: BLE001 - fall back to a local snapshot
        try:
            from dataclasses import asdict

            snapshot = json.dumps(asdict(cfg), sort_keys=True, default=str)
        except TypeError:
            snapshot = json.dumps(vars(cfg), sort_keys=True, default=str)
        return hashlib.sha256(snapshot.encode("utf-8")).hexdigest()


def _feat_hash(X: np.ndarray) -> str:
    """md5 of the first 100 rows of the source features (cache key)."""
    return hashlib.md5(np.asarray(X)[:100].tobytes()).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class VAE(_nn_module_base()):  # type: ignore[misc, valid-type]
    """Unsupervised variational autoencoder (768 -> 64 -> 768).

    Architecture (locked by the rewrite plan / AGENTS.md):

    * Encoder: ``Linear(768->256) -> LeakyReLU -> BatchNorm1d ->
      Linear(256->128) -> mu, logvar`` (each ``Linear(128->64)``).
    * Decoder: ``Linear(64->256) -> LeakyReLU -> BatchNorm1d ->
      Linear(256->768)``.

    The probabilistic latent space (reparameterization trick) provides
    smoother representations for the downstream quantum classifier.
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 64,
        beta: float = 0.001,
    ) -> None:
        import torch.nn as nn

        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder: 768 -> 256 -> LeakyReLU -> BN -> 128 -> mu/logvar
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        # Decoder: 64 -> 256 -> LeakyReLU -> BN -> 768
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Linear(256, input_dim),
        )

    def encode(self, x: Any) -> tuple[Any, Any]:
        """Encode input to the latent posterior parameters ``(mu, logvar)``."""
        import torch.nn.functional as F

        h = F.leaky_relu(self.fc1(x), 0.2)
        h = self.bn1(h)
        h = self.fc2(h)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: Any, logvar: Any) -> Any:
        """Reparameterization trick: ``z = mu + eps * std``, ``eps ~ N(0, I)``."""
        import torch

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: Any) -> Any:
        """Decode a latent vector back to the input space."""
        return self.decoder(z)

    def kl_loss(self, mu: Any, logvar: Any) -> Any:
        """KL divergence: ``-0.5 * sum(1 + logvar - mu^2 - exp(logvar))``."""
        import torch

        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def forward(self, x: Any) -> tuple[Any, Any, Any, Any]:
        """Full VAE forward pass.

        Returns:
            ``(recon, z, mu, logvar)`` — reconstruction, latent sample, and
            the posterior parameters.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, z, mu, logvar


class SupervisedVAE(VAE):
    """Variational autoencoder with an auxiliary classification head.

    Extends :class:`VAE` with a classifier on the latent space so that the
    64-dim features preserve class-discriminative signal during
    dimensionality reduction:

    * Classifier: ``Linear(64->32) -> ReLU -> Linear(32->1)`` — outputs
      **raw logits** (no sigmoid); train with ``BCEWithLogitsLoss``.

    Loss: ``L = L_recon + beta * L_KL + lambda_clf * L_clf
    [+ lambda_coral * L_coral]``.
    """

    def __init__(
        self,
        input_dim: int = 768,
        latent_dim: int = 64,
        beta: float = 0.001,
    ) -> None:
        import torch.nn as nn

        super().__init__(input_dim=input_dim, latent_dim=latent_dim, beta=beta)

        # Classification head on the latent space (raw logits).
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: Any) -> tuple[Any, Any, Any]:
        """Full SVAE forward pass.

        Returns:
            ``(recon, z, clf_logit)`` — reconstruction, latent sample, and
            raw classification logits (shape ``(B,)`` after squeeze).
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        clf_logit = self.classifier(z).squeeze(-1)
        return recon, z, clf_logit


# ---------------------------------------------------------------------------
# Losses and domain adaptation
# ---------------------------------------------------------------------------


def supervised_vae_loss(
    model: SupervisedVAE,
    x: Any,
    y: Any,
    beta: float = 0.001,
    lambda_clf: float = 0.01,
    pos_weight: Any = None,
    z_src: Any = None,
    z_tgt: Any = None,
    lambda_coral: float = 0.0,
) -> Any:
    """Combined differentiable loss for :class:`SupervisedVAE`.

    ``L = MSE_recon + beta * KL + lambda_clf * BCEWithLogits(clf, y)
    [+ lambda_coral * coral_loss(z_src, z_tgt)]``

    The CORAL term is only added when ``z_src``/``z_tgt`` (latent features of
    a source and a target batch) and ``lambda_coral > 0`` are supplied. All
    terms are differentiable tensors connected to the model parameters — the
    returned scalar supports ``loss.backward()`` end-to-end.

    Args:
        model: A :class:`SupervisedVAE` instance.
        x: Input features, shape ``(B, input_dim)``.
        y: Binary labels, shape ``(B,)`` (floats 0/1).
        beta: KL loss weight (default 0.001).
        lambda_clf: Classification loss weight (0 disables the head loss).
        pos_weight: Optional ``BCEWithLogitsLoss`` positive-class weight.
        z_src: Latent features of the source batch (for CORAL).
        z_tgt: Latent features of the target batch (for CORAL).
        lambda_coral: CORAL domain-alignment weight (0 disables).

    Returns:
        Scalar loss tensor (differentiable).
    """
    import torch.nn.functional as F

    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)
    recon = model.decode(z)
    clf_logit = model.classifier(z).squeeze(-1)

    recon_loss = F.mse_loss(recon, x)
    kl = model.kl_loss(mu, logvar)
    total = recon_loss + beta * kl

    if lambda_clf > 0 and y is not None:
        clf_loss = F.binary_cross_entropy_with_logits(
            clf_logit, y, pos_weight=pos_weight
        )
        total = total + lambda_clf * clf_loss

    if lambda_coral > 0 and z_src is not None and z_tgt is not None:
        total = total + lambda_coral * coral_loss(z_src, z_tgt)

    return total


def compute_pos_weight(y_train: Any) -> Any:
    """Compute ``BCEWithLogitsLoss`` pos_weight from training labels.

    ``pos_weight = (len(y) - sum(y)) / max(sum(y), 1)`` — the negative /
    positive class ratio, guarding against an all-negative label vector.

    Args:
        y_train: Binary labels (array-like, 0/1).

    Returns:
        A 0-dim float32 tensor.
    """
    import torch

    y = torch.as_tensor(np.asarray(y_train, dtype=np.float32))
    n_pos = y.sum()
    n_neg = y.numel() - n_pos
    denom = torch.clamp(n_pos, min=1.0)
    return (n_neg / denom).to(torch.float32)


def coral_loss(feats_src: Any, feats_tgt: Any) -> Any:
    """CORrelation ALignment loss on latent features.

    ``L_coral = 0.25 * ||Cov(feats_src) - Cov(feats_tgt)||_F^2``

    Covariances are computed with ``(X - mean).T @ (X - mean) / (n - 1)``.
    **CRITICAL:** every operation is a differentiable tensor op — no
    ``.detach()`` / ``.numpy()`` / ``float()`` inside the graph, so the
    returned loss carries gradients back to the encoder parameters.

    Args:
        feats_src: Source-domain latent features, shape ``(B, d)``.
        feats_tgt: Target-domain latent features, shape ``(B, d)``.

    Returns:
        Scalar loss tensor (differentiable).
    """
    src = feats_src - feats_src.mean(dim=0, keepdim=True)
    tgt = feats_tgt - feats_tgt.mean(dim=0, keepdim=True)
    n_src = max(feats_src.size(0) - 1, 1)
    n_tgt = max(feats_tgt.size(0) - 1, 1)
    cov_src = (src.T @ src) / n_src
    cov_tgt = (tgt.T @ tgt) / n_tgt
    return 0.25 * (cov_src - cov_tgt).pow(2).sum()


def inter_domain_mixup(
    feats_src: Any,
    feats_tgt: Any,
    labels_src: Any,
    alpha: float = 0.2,
    labels_tgt: Any = None,
) -> tuple[Any, Any]:
    """Feature-space MixUp between source and target domains.

    Per-sample ``lambda ~ Beta(alpha, alpha)`` mixes the source batch with a
    randomly permuted target batch:

    * ``mixed_feats = lambda * src + (1 - lambda) * tgt_permuted``
    * ``mixed_labels = lambda * y_src + (1 - lambda) * y_tgt_permuted``
      (soft labels; falls back to permuted source labels when ``labels_tgt``
      is ``None``).

    All operations are tensor ops, so the mixed features stay differentiable
    w.r.t. both source and target latents.

    Args:
        feats_src: Source-domain latent features, shape ``(B, d)``.
        feats_tgt: Target-domain latent features, shape ``(B, d)``.
        labels_src: Source labels, shape ``(B,)``.
        alpha: Beta distribution parameter (default 0.2).
        labels_tgt: Optional target labels, shape ``(B,)``. When ``None``,
            the permuted source labels are used as the target labels.

    Returns:
        ``(mixed_feats, mixed_labels)`` — shape ``(B, d)`` and ``(B,)``.
    """
    import torch

    batch = feats_src.size(0)
    if feats_tgt.size(0) != batch:
        raise ValueError(
            f"source/target batch mismatch: {batch} != {feats_tgt.size(0)}"
        )

    lam = torch.distributions.Beta(alpha, alpha).sample((batch,)).to(feats_src.device)
    lam_col = lam.reshape(-1, 1)

    perm = torch.randperm(batch, device=feats_src.device)
    feats_tgt_perm = feats_tgt[perm]
    mixed_feats = lam_col * feats_src + (1.0 - lam_col) * feats_tgt_perm

    if labels_tgt is None:
        labels_tgt = labels_src
    labels_tgt_perm = labels_tgt[perm]
    mixed_labels = lam * labels_src + (1.0 - lam) * labels_tgt_perm

    return mixed_feats, mixed_labels


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


def _vae_val_loss(
    model: Any,
    val_t: Any,
    beta: float,
    is_supervised: bool,
    y_val_t: Any,
    lambda_clf: float,
    pos_weight: Any,
) -> float:
    """Reconstruction + KL (and optionally classifier) loss on the val set."""
    import torch
    import torch.nn.functional as F

    model.eval()
    with torch.no_grad():
        if is_supervised:
            mu, logvar = model.encode(val_t)
            z = model.reparameterize(mu, logvar)
            recon = model.decode(z)
            loss = F.mse_loss(recon, val_t) + beta * model.kl_loss(mu, logvar)
            if lambda_clf > 0 and y_val_t is not None:
                clf_logit = model.classifier(z).squeeze(-1)
                loss = loss + lambda_clf * F.binary_cross_entropy_with_logits(
                    clf_logit, y_val_t, pos_weight=pos_weight
                )
        else:
            recon, _z, mu, logvar = model(val_t)
            loss = F.mse_loss(recon, val_t) + beta * model.kl_loss(mu, logvar)
    model.train()
    return float(loss.item())


def fit_vae_pipeline(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray | None,
    cfg: Any,
    save_dir: str | Path,
    prefix: str = "",
    y_train: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
) -> dict[str, Any]:
    """Train a (Supervised)VAE and return L2-normalized 64-dim latent features.

    Trains on the 768-dim ConvNeXt-Tiny features extracted by Task 6 and
    returns 64-dim latent vectors, L2-normalized row-wise for amplitude
    encoding into 6 qubits.

    Domain adaptation (per batch, all differentiable):

    * **CORAL** — a target batch is sampled from the ``X_test`` pool and
      encoded through the model with gradients enabled; ``lambda_coral *
      coral_loss(z_src, z_tgt)`` is added to the loss *before* backward.
    * **MixUp** — source and target latent batches are mixed with
      ``lambda ~ Beta(mixup_alpha, mixup_alpha)`` and the mixed latents are
      classified with soft labels (supervised mode only).

    Cache contract (Data Flow / Cache Artifacts): artifacts are saved as
    ``{prefix}_vae_{split}.npy``, ``{prefix}_vae_scaler.pkl``,
    ``{prefix}_vae_weights.pt`` and ``{prefix}_vae_meta.json`` under
    ``save_dir`` (with ``prefix=""`` these are exactly ``vae_train.npy``,
    ``vae_scaler.pkl``, ...). ``vae_meta.json`` stores ``config_hash``,
    ``feat_hash``, ``git_sha`` and ``timestamp``; when the config hash and
    feature hash match and all files exist, the cached latents are loaded
    from disk instead of retraining.

    Args:
        X_train: Source-domain features, shape ``(N, 768)``.
        X_val: Validation features, shape ``(N, 768)``.
        X_test: Target-domain (test) features, shape ``(N, 768)``. Used as
            the CORAL/MixUp target pool.
        cfg: Configuration object. Recognised fields: ``reduction_method``
            (``"vae"`` | ``"svae"``), ``target_dims``, ``vae_beta``,
            ``vae_lambda_clf``, ``vae_lambda_coral``, ``vae_lambda_mixup``,
            ``mixup_alpha``, ``vae_epochs``, ``vae_batch_size``, ``vae_lr``,
            ``seed``, ``device``, ``artifacts_dir``.
        save_dir: Directory for the cache artifacts (e.g.
            ``artifacts/features``).
        prefix: Optional artifact name prefix (default ``""`` → files named
            ``vae_*.npy`` etc.).
        y_train: Training labels (required when ``reduction_method="svae"``).
        y_val: Validation labels (used for the monitored val loss).
        y_test: Test labels (used as MixUp target labels when provided).

    Returns:
        Dict with keys ``X_train_latent``, ``X_val_latent``,
        ``X_test_latent`` (each ``(N, 64)`` L2-normalized), ``history``,
        ``scaler``, ``model`` and ``meta``.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    import joblib
    from sklearn.preprocessing import StandardScaler, normalize

    from pipeline.seed import get_torch_generator, seed_everything

    seed_everything(int(getattr(cfg, "seed", 6)))

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test_arr = None if X_test is None else np.asarray(X_test, dtype=np.float32)

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- hyperparameters from cfg (getattr fallbacks for robustness) ---
    latent_dim = int(getattr(cfg, "target_dims", 64))
    beta = float(getattr(cfg, "vae_beta", 0.001))
    lambda_clf = float(getattr(cfg, "vae_lambda_clf", 0.01))
    lambda_coral = float(getattr(cfg, "vae_lambda_coral", 0.0))
    lambda_mixup = float(getattr(cfg, "vae_lambda_mixup", 1.0))
    mixup_alpha = float(getattr(cfg, "mixup_alpha", 0.2))
    epochs = int(getattr(cfg, "vae_epochs", 30))
    batch_size = int(getattr(cfg, "vae_batch_size", 32))
    lr = float(getattr(cfg, "vae_lr", 1e-3))
    seed = int(getattr(cfg, "seed", 6))

    reduction_method = str(getattr(cfg, "reduction_method", "vae"))
    is_supervised = reduction_method == "svae"

    if is_supervised and y_train is None:
        raise ValueError(
            "reduction_method='svae' requires y_train (and ideally y_val) labels."
        )
    if not is_supervised and lambda_clf > 0:
        warnings.warn(
            "lambda_clf > 0 but reduction_method != 'svae'; the classifier "
            "loss is ignored (plain VAE has no classifier head).",
            RuntimeWarning,
            stacklevel=2,
        )
    if not is_supervised and lambda_mixup > 0:
        warnings.warn(
            "lambda_mixup > 0 but reduction_method != 'svae'; MixUp is "
            "skipped (soft labels require y_train).",
            RuntimeWarning,
            stacklevel=2,
        )

    if X_test_arr is None or len(X_test_arr) == 0:
        if lambda_coral > 0 or lambda_mixup > 0:
            warnings.warn(
                "X_test is empty; disabling CORAL/MixUp domain adaptation.",
                RuntimeWarning,
                stacklevel=2,
            )
        lambda_coral = 0.0
        lambda_mixup = 0.0

    device = str(getattr(cfg, "device", "auto"))
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- cache validation (config hash + feature hash + all files present) ---
    meta_path = save_dir / f"{prefix}_vae_meta.json"
    train_path = save_dir / f"{prefix}_vae_train.npy"
    val_path = save_dir / f"{prefix}_vae_val.npy"
    test_path = save_dir / f"{prefix}_vae_test.npy"
    scaler_path = save_dir / f"{prefix}_vae_scaler.pkl"
    weights_path = save_dir / f"{prefix}_vae_weights.pt"

    config_h = _config_hash(cfg)
    feat_h = _feat_hash(X_train)

    cache_files = [
        meta_path,
        train_path,
        val_path,
        test_path,
        scaler_path,
        weights_path,
    ]
    if all(p.is_file() for p in cache_files):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("config_hash") == config_h and meta.get("feat_hash") == feat_h:
            print(f"Loading cached VAE features (config_hash={config_h[:8]}...)")
            latent_train = np.load(train_path)
            latent_val = np.load(val_path)
            latent_test = np.load(test_path)
            scaler = joblib.load(scaler_path)
            if is_supervised:
                model = SupervisedVAE(
                    input_dim=X_train.shape[1], latent_dim=latent_dim, beta=beta
                )
            else:
                model = VAE(
                    input_dim=X_train.shape[1], latent_dim=latent_dim, beta=beta
                )
            model.load_state_dict(torch.load(weights_path, map_location="cpu"))
            return {
                "X_train_latent": latent_train,
                "X_val_latent": latent_val,
                "X_test_latent": latent_test,
                "history": meta.get("history", {}),
                "scaler": scaler,
                "model": model,
                "meta": meta,
            }
        print("VAE cache stale (config/features changed); retraining...")
    else:
        print("No VAE cache found; training...")

    # --- standardize features ---
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)
    Xva_s = scaler.transform(X_val)
    Xte_s = scaler.transform(X_test_arr)

    train_t = torch.tensor(Xtr_s, dtype=torch.float32).to(device)
    val_t = torch.tensor(Xva_s, dtype=torch.float32).to(device)
    test_t = torch.tensor(Xte_s, dtype=torch.float32).to(device)

    # --- model ---
    if is_supervised:
        model = SupervisedVAE(
            input_dim=X_train.shape[1], latent_dim=latent_dim, beta=beta
        )
    else:
        model = VAE(input_dim=X_train.shape[1], latent_dim=latent_dim, beta=beta)
    model.to(device)

    # --- dataset / loader (seeded shuffle for reproducibility) ---
    if is_supervised:
        y_train_t = torch.tensor(
            np.asarray(y_train, dtype=np.float32), dtype=torch.float32
        )
        train_ds = TensorDataset(train_t, y_train_t)
    else:
        train_ds = TensorDataset(train_t, train_t)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=get_torch_generator(seed),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    pos_weight = None
    if is_supervised:
        pos_weight = compute_pos_weight(y_train).to(device)

    # Target-domain pool (test features) for CORAL/MixUp.
    test_pool = test_t
    y_test_t = None
    if y_test is not None:
        y_test_t = torch.tensor(
            np.asarray(y_test, dtype=np.float32), dtype=torch.float32
        ).to(device)
    y_val_t = None
    if y_val is not None:
        y_val_t = torch.tensor(
            np.asarray(y_val, dtype=np.float32), dtype=torch.float32
        ).to(device)

    tgt_gen = torch.Generator()
    tgt_gen.manual_seed(seed)

    history: dict[str, list[float]] = {
        "train_loss": [],
        "recon": [],
        "kl": [],
        "clf": [],
        "coral": [],
        "mixup": [],
        "val_loss": [],
    }

    mode_str = "SupervisedVAE" if is_supervised else "VAE"
    print(
        f"Training {mode_str} ({X_train.shape[1]} -> {latent_dim}, "
        f"beta={beta}) for {epochs} epochs on {device}"
    )
    if lambda_coral > 0:
        print(f"  CORAL domain alignment enabled (lambda_coral={lambda_coral})")
    if is_supervised and lambda_mixup > 0:
        print(f"  MixUp enabled (lambda_mixup={lambda_mixup}, alpha={mixup_alpha})")
    if is_supervised:
        print(f"  Classification loss enabled (lambda_clf={lambda_clf})")

    model.train()
    for epoch in range(epochs):
        epoch_loss = epoch_recon = epoch_kl = 0.0
        epoch_clf = epoch_coral = epoch_mixup = 0.0
        n_batches = 0

        for batch in train_loader:
            if is_supervised:
                xb, yb = batch
                xb = xb.to(device)
                yb = yb.to(device)
            else:
                xb = batch[0].to(device)
                yb = None

            optimizer.zero_grad()

            if is_supervised:
                # Single forward pass: z_src feeds recon + clf + coral + mixup.
                mu_src, logvar_src = model.encode(xb)
                z_src = model.reparameterize(mu_src, logvar_src)
                recon = model.decode(z_src)
                clf_logit = model.classifier(z_src).squeeze(-1)

                recon_loss = F.mse_loss(recon, xb)
                kl = model.kl_loss(mu_src, logvar_src)
                total = recon_loss + beta * kl

                clf_loss = torch.zeros((), device=device)
                if lambda_clf > 0:
                    clf_loss = F.binary_cross_entropy_with_logits(
                        clf_logit, yb, pos_weight=pos_weight
                    )
                    total = total + lambda_clf * clf_loss

                coral_val = torch.zeros((), device=device)
                mixup_val = torch.zeros((), device=device)
                if lambda_coral > 0 or lambda_mixup > 0:
                    # Target batch from the test-domain pool — gradients flow
                    # through BOTH source and target encodings (no detach).
                    tgt_idx = torch.randint(
                        0, test_pool.size(0), (xb.size(0),), generator=tgt_gen
                    ).to(device)
                    x_tgt = test_pool[tgt_idx]
                    mu_tgt, logvar_tgt = model.encode(x_tgt)
                    z_tgt = model.reparameterize(mu_tgt, logvar_tgt)

                    if lambda_coral > 0:
                        coral_val = coral_loss(z_src, z_tgt)
                        total = total + lambda_coral * coral_val

                    if lambda_mixup > 0:
                        y_tgt_batch = (
                            y_test_t[tgt_idx] if y_test_t is not None else None
                        )
                        z_mixed, y_mixed = inter_domain_mixup(
                            z_src,
                            z_tgt,
                            yb,
                            alpha=mixup_alpha,
                            labels_tgt=y_tgt_batch,
                        )
                        mixup_logits = model.classifier(z_mixed).squeeze(-1)
                        mixup_val = F.binary_cross_entropy_with_logits(
                            mixup_logits, y_mixed, pos_weight=pos_weight
                        )
                        total = total + lambda_mixup * mixup_val
            else:
                recon, z, mu, logvar = model(xb)
                recon_loss = F.mse_loss(recon, xb)
                kl = model.kl_loss(mu, logvar)
                total = recon_loss + beta * kl

                clf_loss = torch.zeros((), device=device)
                coral_val = torch.zeros((), device=device)
                mixup_val = torch.zeros((), device=device)
                if lambda_coral > 0:
                    tgt_idx = torch.randint(
                        0, test_pool.size(0), (xb.size(0),), generator=tgt_gen
                    ).to(device)
                    x_tgt = test_pool[tgt_idx]
                    mu_tgt, logvar_tgt = model.encode(x_tgt)
                    z_tgt = model.reparameterize(mu_tgt, logvar_tgt)
                    coral_val = coral_loss(z, z_tgt)
                    total = total + lambda_coral * coral_val

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += float(total.item())
            epoch_recon += float(recon_loss.item())
            epoch_kl += float(kl.item())
            epoch_clf += float(clf_loss.item())
            epoch_coral += float(coral_val.item())
            epoch_mixup += float(mixup_val.item())
            n_batches += 1

        val_loss = _vae_val_loss(
            model, val_t, beta, is_supervised, y_val_t, lambda_clf, pos_weight
        )

        history["train_loss"].append(epoch_loss / n_batches)
        history["recon"].append(epoch_recon / n_batches)
        history["kl"].append(epoch_kl / n_batches)
        history["clf"].append(epoch_clf / n_batches)
        history["coral"].append(epoch_coral / n_batches)
        history["mixup"].append(epoch_mixup / n_batches)
        history["val_loss"].append(val_loss)

        if (epoch + 1) % 10 == 0 or epoch + 1 == epochs:
            parts = [
                f"Total: {history['train_loss'][-1]:.4f}",
                f"Recon: {history['recon'][-1]:.4f}",
                f"KL: {history['kl'][-1]:.6f}",
            ]
            if lambda_clf > 0:
                parts.append(f"Clf: {history['clf'][-1]:.6f}")
            if lambda_coral > 0:
                parts.append(f"Coral: {history['coral'][-1]:.6f}")
            if lambda_mixup > 0:
                parts.append(f"Mixup: {history['mixup'][-1]:.6f}")
            print(f"  {mode_str} Epoch {epoch+1}/{epochs} | {' | '.join(parts)}")

    # --- extract deterministic latents (mu) and L2-normalize ---
    model.eval()
    with torch.no_grad():
        mu_tr, _ = model.encode(train_t)
        mu_va, _ = model.encode(val_t)
        mu_te, _ = model.encode(test_t)
    latent_train = normalize(mu_tr.cpu().numpy(), norm="l2")
    latent_val = normalize(mu_va.cpu().numpy(), norm="l2")
    latent_test = normalize(mu_te.cpu().numpy(), norm="l2")

    # --- save cache ---
    np.save(train_path, latent_train)
    np.save(val_path, latent_val)
    np.save(test_path, latent_test)
    joblib.dump(scaler, scaler_path)
    torch.save(model.state_dict(), weights_path)

    meta = {
        "config_hash": config_h,
        "feat_hash": feat_h,
        "git_sha": _git_sha(),
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "reduction_method": reduction_method,
        "latent_dim": latent_dim,
        "beta": beta,
        "lambda_clf": lambda_clf,
        "lambda_coral": lambda_coral,
        "lambda_mixup": lambda_mixup,
        "mixup_alpha": mixup_alpha,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "final_loss": float(history["train_loss"][-1]),
        "history": history,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"  Final {mode_str} total loss: {history['train_loss'][-1]:.4f}")
    print(
        f"  Latent feature shapes: train={latent_train.shape}, "
        f"val={latent_val.shape}, test={latent_test.shape}"
    )

    return {
        "X_train_latent": latent_train,
        "X_val_latent": latent_val,
        "X_test_latent": latent_test,
        "history": history,
        "scaler": scaler,
        "model": model,
        "meta": meta,
    }
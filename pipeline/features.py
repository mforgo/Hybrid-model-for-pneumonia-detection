"""DANN_ConvNeXt feature extraction with caching.

Task 6 of the notebook → package rewrite. Implements the Domain-Adversarial
Neural Network (DANN) feature extraction stage of the hybrid QML pneumonia
pipeline:

* :class:`GradientReversalLayer` — the GRL of Ganin & Lempitsky (2015):
  identity in the forward pass, negated-and-scaled gradient in the backward
  pass. Forces the feature extractor to produce domain-invariant
  representations.
* :class:`DANN_ConvNeXt` — a frozen ``convnext_tiny`` backbone (classification
  head replaced with ``nn.Identity()``) plus a label predictor
  ``Linear(768→64)→ReLU→Linear(64→1)`` and a domain classifier
  ``Linear(768→64)→ReLU→Linear(64→1)`` fed through the GRL.
* :func:`get_grl_alpha` — the Ganin schedule
  ``α(p) = 2/(1 + exp(−10p)) − 1`` with ``p = epoch/total`` (plus a ``linear``
  schedule).
* :func:`train_dann_epoch` — one epoch over a loader of ``(x, y, domain)``
  batches; label loss on source samples only, domain loss on every sample.
* :func:`extract_dann_features` — polymorphic: forward-only extraction from a
  model + loader, or deterministic DANN fine-tuning on a numpy feature matrix
  (the ``test_extract_dann_deterministic`` acceptance contract).
* :class:`FeatureCache` + :func:`extract_or_load` — config-hash-validated
  caching of the extracted ``(N, 768)`` features.

Design decisions (LOCKED by the rewrite plan):

* The backbone is frozen by default (passthrough 768-dim features are the
  primary output). During DANN fine-tuning the backbone is **unfrozen** so the
  GRL can actually adapt the features — this matches the reference notebook,
  which trains the full model (``pneumonia_hybrid_qml.ipynb`` cell 14).
* The label/domain heads output **raw logits** (no sigmoid); the loss is
  ``BCEWithLogitsLoss``.
* ``forward(x, alpha)`` returns ``(label_logits, domain_logits, features)`` —
  the third element is required by feature extraction and mirrors the
  reference notebook's ``return label_preds, domain_preds, f``.
* All paths are relative to the repo root via ``cfg.artifacts_dir`` — no
  hardcoded ``/content/*`` paths.

Import policy
-------------
Only the standard library and NumPy are imported at module level. ``torch``
and ``torchvision`` are imported lazily inside the functions/classes that need
them, so ``import pipeline.features`` succeeds even on a machine without
PyTorch (e.g. the local development box). The heavy classes subclass
``torch.autograd.Function`` / ``torch.nn.Module`` when torch is available and
plain ``object`` otherwise (same convention as ``pipeline.data.XRayDataset``).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "GradientReversalLayer",
    "DANN_ConvNeXt",
    "get_grl_alpha",
    "train_dann_epoch",
    "extract_dann_features",
    "FeatureCache",
    "extract_or_load",
    "domain_labels_for",
]

#: Project-wide random seed (AGENTS.md global rule).
SEED = 6


# ---------------------------------------------------------------------------
# Lazy base classes (torch optional at import time)
# ---------------------------------------------------------------------------


def _function_base() -> type:
    """Return ``torch.autograd.Function`` if available, else ``object``.

    Lets :class:`GradientReversalLayer` be a genuine ``Function`` subclass on
    the GPU server while keeping the module importable without torch.
    """
    try:
        from torch.autograd import Function

        return Function
    except Exception:  # noqa: BLE001 - torch optional at import time
        return object


def _module_base() -> type:
    """Return ``torch.nn.Module`` if available, else ``object``."""
    try:
        import torch.nn as nn

        return nn.Module
    except Exception:  # noqa: BLE001 - torch optional at import time
        return object


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------


class GradientReversalLayer(_function_base()):  # type: ignore[misc, valid-type]
    """Gradient Reversal Layer (Ganin & Lempitsky, 2015).

    Forward pass is the identity; the backward pass negates the incoming
    gradient and scales it by ``alpha``::

        ∂L/∂x = −α · ∂L/∂x

    Placed between the feature extractor and the domain classifier, it makes
    the extractor learn features that fool the domain discriminator, i.e.
    domain-invariant representations.
    """

    @staticmethod
    def forward(ctx, x, alpha):
        """Identity forward; stash ``alpha`` for the backward pass."""
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """Negated gradient scaled by ``alpha``; no gradient for ``alpha``."""
        return grad_output.neg() * ctx.alpha, None


# ---------------------------------------------------------------------------
# DANN models
# ---------------------------------------------------------------------------


class DANN_ConvNeXt(_module_base()):  # type: ignore[misc, valid-type]
    """Frozen ConvNeXt-Tiny backbone + adversarial label/domain heads.

    The backbone is ``torchvision.models.convnext_tiny`` pre-trained on
    ImageNet with its classification head replaced by ``nn.Identity()``. The
    pooled 768-dim feature vector feeds two heads:

    * ``label_predictor`` — ``Linear(768→64)→ReLU→Linear(64→1)`` (raw logits).
    * ``domain_classifier`` — ``Linear(768→64)→ReLU→Linear(64→1)`` (raw
      logits), fed through the :class:`GradientReversalLayer`.

    Args:
        feature_dim: Backbone feature dimensionality (768 for ConvNeXt-Tiny).
        freeze_backbone: Freeze all backbone parameters on construction.
            The backbone is unfrozen by :func:`extract_or_load` during DANN
            fine-tuning so the GRL can adapt the features (reference notebook
            behaviour).

    Raises:
        RuntimeError: If the ImageNet weights cannot be downloaded/loaded
            (the GPU server needs network access on first use).
    """

    def __init__(self, feature_dim: int = 768, freeze_backbone: bool = True) -> None:
        super().__init__()
        import torch.nn as nn
        import torchvision
        from torchvision.models import ConvNeXt_Tiny_Weights

        try:
            backbone = torchvision.models.convnext_tiny(
                weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1
            )
        except Exception as exc:  # noqa: BLE001 - surface a clear download error
            raise RuntimeError(
                "Failed to load ConvNeXt-Tiny with IMAGENET1K_V1 weights. "
                "The GPU server needs network access to download the weights "
                "on first use (torch.hub)."
            ) from exc

        backbone.classifier = nn.Identity()  # remove the classification head
        self.backbone = backbone
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.label_predictor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, alpha: float = 1.0):
        """Return ``(label_logits, domain_logits, features)``.

        ``features`` is the pooled 768-dim backbone output — the third return
        value is what :func:`extract_dann_features` collects.
        """
        f = self.pool(self.backbone(x))
        label_logits = self.label_predictor(f)
        f_reversed = GradientReversalLayer.apply(f, alpha)
        domain_logits = self.domain_classifier(f_reversed)
        return label_logits, domain_logits, f


class _DANNHead(_module_base()):  # type: ignore[misc, valid-type]
    """Head-only DANN operating directly on pre-extracted feature vectors.

    Used by the numpy-array mode of :func:`extract_dann_features` (the
    ``test_extract_dann_deterministic`` acceptance contract): a trainable
    ``Linear(feature_dim, feature_dim)`` feature transform plays the role of
    the backbone, with the same label/domain heads and GRL as
    :class:`DANN_ConvNeXt`.
    """

    def __init__(self, feature_dim: int = 768) -> None:
        super().__init__()
        import torch.nn as nn

        self.feature_transform = nn.Linear(feature_dim, feature_dim)
        self.label_predictor = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, alpha: float = 1.0):
        f = self.feature_transform(x)
        label_logits = self.label_predictor(f)
        f_reversed = GradientReversalLayer.apply(f, alpha)
        domain_logits = self.domain_classifier(f_reversed)
        return label_logits, domain_logits, f


# ---------------------------------------------------------------------------
# GRL alpha schedule
# ---------------------------------------------------------------------------


def get_grl_alpha(epoch: int, total: int, schedule: str = "ganin") -> float:
    """Compute the GRL scaling factor ``alpha`` for a given epoch.

    Args:
        epoch: Current epoch (0-indexed).
        total: Total number of epochs.
        schedule: ``"ganin"`` (default) uses the original Ganin & Lempitsky
            schedule ``α = 2/(1 + exp(−10p)) − 1`` with ``p = epoch/total``;
            ``"linear"`` uses ``α = p``.

    Returns:
        The GRL scaling factor in ``[0, 1]``.

    Raises:
        ValueError: If ``schedule`` is not recognised.
    """
    p = float(epoch) / max(int(total), 1)
    if schedule == "ganin":
        return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
    if schedule == "linear":
        return p
    raise ValueError(f"Unknown GRL schedule: {schedule!r}")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _to_tensor(value: Any, device: Any, dtype: Any = None):
    """Move *value* to *device* (and optionally cast to *dtype*).

    Accepts torch tensors (moved with ``.to``) and array-likes (converted
    with ``torch.as_tensor``).
    """
    import torch

    if isinstance(value, torch.Tensor):
        t = value.to(device)
        if dtype is not None:
            t = t.to(dtype)
        return t
    return torch.as_tensor(value, device=device, dtype=dtype)


def _is_torch_module(obj: Any) -> bool:
    """``True`` if *obj* is a ``torch.nn.Module`` (torch optional)."""
    try:
        import torch.nn as nn

        return isinstance(obj, nn.Module)
    except Exception:  # noqa: BLE001 - torch optional at import time
        return False


def _features_from_output(out: Any):
    """Extract the feature tensor from a model's forward output.

    Accepts a plain tensor (backbone) or a ``(label_logits, domain_logits,
    features)`` tuple (:class:`DANN_ConvNeXt`).
    """
    if isinstance(out, (tuple, list)):
        if len(out) >= 3:
            return out[2]
        raise ValueError(
            "Model forward returned a tuple of length < 3; cannot locate the "
            "feature vector. Expected (label_logits, domain_logits, features)."
        )
    return out


def _resolve_device(cfg: Any):
    """Resolve ``cfg.device``; ``"auto"`` picks CUDA when available."""
    import torch

    device_str = str(getattr(cfg, "device", "auto") or "auto")
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def _git_sha() -> str:
    """Current git commit sha, or ``"unknown"`` when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return result.stdout.strip()
    except Exception:  # noqa: BLE001 - git is optional for caching
        return "unknown"


def _feat_hash(X_train: np.ndarray) -> str:
    """md5 of the first 100 rows of the training features (cache key)."""
    return hashlib.md5(np.ascontiguousarray(X_train[:100]).tobytes()).hexdigest()


# ---------------------------------------------------------------------------
# DANN training
# ---------------------------------------------------------------------------


def domain_labels_for(loader: Any, domain: Any):
    """Wrap a ``(x, y)`` loader to yield ``(x, y, domain)`` triples.

    ``domain`` is a scalar (0 = source, 1 = target) applied to every batch,
    or a per-sample array-like broadcastable to each batch.

    Example::

        source_loader = domain_labels_for(train_loader, 0)
        target_loader = domain_labels_for(test_loader, 1)
    """
    for x, y in loader:
        yield x, y, domain


def _combined_dann_loader(source_loader: Any, target_loader: Any):
    """Yield ``(x, y, domain)`` batches: source then target, cycling target.

    Mirrors the reference notebook's ``zip(train_loader, cycle(test_loader))``:
    every source batch is paired with a target batch so all training samples
    are covered. Target labels are hidden (``y=None``) — the domain adversary
    never sees the medical labels of the target domain.
    """
    from itertools import cycle

    target_iter = cycle(target_loader)
    for x_src, y_src in source_loader:
        yield x_src, y_src, 0
        x_tgt, _ = next(target_iter)
        yield x_tgt, None, 1


def train_dann_epoch(model: Any, loader: Any, opt: Any, alpha: float, device: Any) -> float:
    """Run one DANN training epoch over a loader of ``(x, y, domain)`` batches.

    Each batch is a triple ``(x, y, domain)`` where ``domain`` is 0 for the
    source domain (train) and 1 for the target domain (test). ``y`` may be
    ``None`` for target batches (medical labels are hidden from the domain
    adversary, matching the transductive UDA setup of the reference
    notebook). The label loss is computed only on source samples; the domain
    loss is computed on every sample. Both heads output raw logits, so the
    loss is ``BCEWithLogitsLoss``.

    Args:
        model: A :class:`DANN_ConvNeXt` (or any model whose forward returns
            ``(label_logits, domain_logits, features)``).
        loader: Iterable of ``(x, y, domain)`` batches. Build it with
            :func:`domain_labels_for` or :func:`_combined_dann_loader`.
        opt: ``torch.optim.Optimizer`` over the model parameters.
        alpha: GRL scaling factor for this epoch (see :func:`get_grl_alpha`).
        device: Device to move batches to.

    Returns:
        Mean total loss over the epoch.

    Raises:
        ValueError: If a batch does not contain a domain label.
    """
    import torch
    import torch.nn as nn

    model.train()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_samples = 0

    for batch in loader:
        if len(batch) < 3:
            raise ValueError(
                "train_dann_epoch expects batches of the form (x, y, domain); "
                f"got a batch with {len(batch)} elements. Wrap loaders with "
                "domain_labels_for(loader, domain) to add domain labels."
            )
        x, y, domain = batch[:3]
        x = _to_tensor(x, device=device)
        domain = _to_tensor(domain, device=device, dtype=torch.float32)
        if domain.dim() == 0:
            domain = domain.expand(x.size(0))

        opt.zero_grad()
        label_logits, domain_logits, _ = model(x, alpha)
        loss = criterion(domain_logits.squeeze(-1), domain)
        if y is not None:
            y = _to_tensor(y, device=device, dtype=torch.float32)
            src_mask = domain == 0
            if bool(src_mask.any()):
                loss = loss + criterion(
                    label_logits.squeeze(-1)[src_mask], y[src_mask]
                )
        loss.backward()
        opt.step()

        total_loss += float(loss.item()) * x.size(0)
        n_samples += x.size(0)

    return total_loss / max(n_samples, 1)


def _finetune_dann(cfg: Any, model: Any, loaders: dict[str, Any], device: Any) -> Any:
    """Fine-tune the DANN model on source (train) vs target (test) domains.

    Unfreezes the backbone so the GRL can adapt the features (reference
    notebook trains the full model), then runs ``cfg.dann_epochs`` epochs with
    the Ganin alpha schedule and Adam at ``cfg.dann_lr``.
    """
    import torch

    # Unfreeze the backbone: the GRL must be able to adapt the features for
    # domain invariance (reference notebook trains the full model).
    for param in model.backbone.parameters():
        param.requires_grad = True

    dann_epochs = int(getattr(cfg, "dann_epochs", 10))
    dann_lr = float(getattr(cfg, "dann_lr", 1e-4))
    opt = torch.optim.Adam(model.parameters(), lr=dann_lr)

    source_loader = loaders["train_loader"]
    target_loader = loaders["test_loader"]

    for epoch in range(dann_epochs):
        alpha = get_grl_alpha(epoch, dann_epochs, schedule="ganin")
        dann_loader = _combined_dann_loader(source_loader, target_loader)
        mean_loss = train_dann_epoch(model, dann_loader, opt, alpha, device)
        print(f"  DANN epoch {epoch + 1}/{dann_epochs} (alpha={alpha:.3f}): loss={mean_loss:.4f}")

    return model


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _extract_forward(model: Any, loader: Any, device: Any) -> tuple[np.ndarray, np.ndarray]:
    """Forward-only feature extraction: run *model* over *loader* in eval mode.

    Returns ``(features (N, feature_dim), labels (N,))`` in deterministic
    loader order (val/test loaders do not shuffle; the train loader shuffles
    with a seeded generator from ``pipeline.data.get_loaders``).
    """
    import torch

    model.eval()
    feats: list[np.ndarray] = []
    lbls: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            x = _to_tensor(batch[0], device=device)
            labels = batch[1]
            out = model(x)
            f = _features_from_output(out)
            feats.append(f.cpu().numpy())
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            lbls.append(np.asarray(labels))

    if not feats:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(feats), np.concatenate(lbls)


def _extract_dann_features_numpy(
    X: np.ndarray,
    y: np.ndarray,
    domain: np.ndarray,
    seed: int = SEED,
    epochs: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic DANN fine-tuning on a pre-extracted feature matrix.

    Trains a head-only DANN (:class:`_DANNHead`) on ``X`` for ``epochs``
    epochs and returns the DANN-refined features ``(N, feature_dim)`` together
    with ``y``. Everything is seeded with ``seed`` (model init, DataLoader
    shuffle) so repeated calls produce bit-identical features — the
    ``test_extract_dann_deterministic`` acceptance contract.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from pipeline.seed import get_torch_generator, seed_everything

    seed_everything(seed)

    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.asarray(y)
    domain = np.asarray(domain, dtype=np.float32)

    model = _DANNHead(feature_dim=X.shape[1])
    device = torch.device("cpu")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
        torch.from_numpy(domain),
    )
    batch_size = min(16, len(X)) if len(X) else 1
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=get_torch_generator(seed),
    )

    for epoch in range(int(epochs)):
        alpha = get_grl_alpha(epoch, int(epochs), schedule="ganin")
        train_dann_epoch(model, loader, opt, alpha, device)

    model.eval()
    with torch.no_grad():
        feats = model.feature_transform(torch.from_numpy(X)).numpy()
    return feats, y


def extract_dann_features(
    X: Any,
    y: Any,
    domain: Any = None,
    seed: int = SEED,
    epochs: int = 1,
    model: Any = None,
    loader: Any = None,
    device: Any = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract 768-dim features with optional DANN fine-tuning.

    Two calling conventions are supported:

    **Forward-only mode** (model + loader)::

        feats, labels = extract_dann_features(model, loader, device)

    Runs ``model`` over ``loader`` in ``eval()`` mode and returns the backbone
    features ``(N, feature_dim)`` and labels ``(N,)`` in loader order.
    ``model`` may be a :class:`DANN_ConvNeXt` (whose forward returns
    ``(label_logits, domain_logits, features)``) or a plain backbone that
    returns a feature tensor directly.

    **Feature mode** (numpy arrays)::

        feats, labels = extract_dann_features(X, y, domain, seed=6, epochs=1)

    Trains a head-only DANN on the pre-extracted feature matrix ``X`` for
    ``epochs`` epochs (deterministic given ``seed``) and returns the
    DANN-refined features ``(N, feature_dim)`` together with ``y``. This is
    the acceptance contract exercised by ``test_extract_dann_deterministic``.

    Returns:
        ``(features (N, feature_dim), labels (N,))``.
    """
    if model is not None and loader is not None:
        return _extract_forward(model, loader, device)
    if _is_torch_module(X):
        # Positional forward-only call: extract_dann_features(model, loader, device)
        return _extract_forward(X, y, domain)
    if domain is None:
        raise ValueError(
            "extract_dann_features feature mode requires domain labels "
            "(extract_dann_features(X, y, domain, seed=..., epochs=...))."
        )
    return _extract_dann_features_numpy(X, y, domain, seed=seed, epochs=epochs)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@dataclass
class FeatureCache:
    """Cache metadata for the extracted feature artifacts.

    Attributes:
        feat_hash: md5 of the first 100 rows of the training features —
            detects input-data changes.
        config_hash: sha256 of the config snapshot — any hyperparameter
            change invalidates the cache.
        git_sha: Git commit the features were extracted from.
        timestamp: ISO-8601 extraction time (UTC).
    """

    feat_hash: str
    config_hash: str
    git_sha: str
    timestamp: str

    def is_valid(self, cfg_hash: str) -> bool:
        """``True`` if the cached features were produced by this config."""
        return self.config_hash == cfg_hash

    def save(self, path: str | Path) -> None:
        """Persist the cache metadata as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "FeatureCache":
        """Load cache metadata from a JSON file."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)


def _load_features_from_disk(feat_dir: Path) -> dict[str, np.ndarray]:
    """Load the six cached feature/label arrays from *feat_dir*."""
    return {
        "X_train": np.load(feat_dir / "convnext_tiny_train.npy"),
        "y_train": np.load(feat_dir / "y_train.npy"),
        "X_val": np.load(feat_dir / "convnext_tiny_val.npy"),
        "y_val": np.load(feat_dir / "y_val.npy"),
        "X_test": np.load(feat_dir / "convnext_tiny_test.npy"),
        "y_test": np.load(feat_dir / "y_test.npy"),
    }


def _build_dann_model(cfg: Any, device: Any = None) -> Any:
    """Build a :class:`DANN_ConvNeXt` with a frozen backbone, seeded."""
    from pipeline.seed import seed_everything

    seed_everything(int(getattr(cfg, "seed", SEED)))
    model = DANN_ConvNeXt(
        feature_dim=int(getattr(cfg, "feature_dim", 768)),
        freeze_backbone=True,
    )
    if device is not None:
        model = model.to(device)
    return model


def extract_or_load(cfg: Any, loaders: dict[str, Any]) -> dict[str, np.ndarray]:
    """Extract (or load cached) 768-dim ConvNeXt-Tiny features.

    First extracts the raw 768-dim features from the frozen backbone over the
    train/val/test loaders; then, if ``cfg.use_dann`` is true and
    ``cfg.dann_epochs > 0``, fine-tunes the DANN on top (source = train,
    target = test) and re-extracts the improved features. Both paths produce
    the same ``(N, 768)`` dict contract.

    Artifacts (relative to ``cfg.artifacts_dir``)::

        artifacts/features/convnext_tiny_{train,val,test}.npy   (N, 768)
        artifacts/features/y_{train,val,test}.npy               (N,)
        artifacts/features/dann_convnext_finetuned.pt           (only if DANN fine-tuned)
        artifacts/features/feature_meta.json                    (FeatureCache)

    Cache validity: ``feature_meta.json`` exists AND its ``config_hash``
    equals ``config_hash(cfg)`` AND all six ``.npy`` files exist → load from
    disk instead of re-extracting. Any config change invalidates the cache.

    Args:
        cfg: Configuration object (``pipeline.config.Config``).
        loaders: Dict from ``pipeline.data.get_loaders`` with keys
            ``train_loader``, ``val_loader``, ``test_loader``.

    Returns:
        Dict with keys ``X_train, y_train, X_val, y_val, X_test, y_test`` —
        numpy arrays of shape ``(N, 768)`` and ``(N,)`` respectively.
    """
    from pipeline.config import config_hash
    from pipeline.seed import seed_everything

    cfg_hash = config_hash(cfg)
    feat_dir = (
        Path(str(getattr(cfg, "artifacts_dir", "artifacts") or "artifacts")) / "features"
    )
    meta_path = feat_dir / "feature_meta.json"
    splits = ("train", "val", "test")

    # Cache hit: metadata present, config hash matches, all arrays exist.
    if meta_path.is_file():
        try:
            cache = FeatureCache.load(meta_path)
        except Exception:  # noqa: BLE001 - corrupt metadata -> re-extract
            cache = None
        if cache is not None and cache.is_valid(cfg_hash) and all(
            (feat_dir / f"convnext_tiny_{s}.npy").is_file()
            and (feat_dir / f"y_{s}.npy").is_file()
            for s in splits
        ):
            print(
                f"[features] Cache hit (config_hash={cfg_hash[:12]}...) — "
                f"loading from {feat_dir}"
            )
            return _load_features_from_disk(feat_dir)

    # Cache miss: extract raw features from the frozen backbone.
    print(f"[features] Cache miss — extracting ConvNeXt-Tiny features to {feat_dir}")
    seed_everything(int(getattr(cfg, "seed", SEED)))
    device = _resolve_device(cfg)
    model = _build_dann_model(cfg, device=device)

    X_train, y_train = _extract_forward(model, loaders["train_loader"], device)
    X_val, y_val = _extract_forward(model, loaders["val_loader"], device)
    X_test, y_test = _extract_forward(model, loaders["test_loader"], device)

    # Optional DANN fine-tuning on top of the frozen-backbone features.
    use_dann = bool(getattr(cfg, "use_dann", True))
    dann_epochs = int(getattr(cfg, "dann_epochs", 0))
    if use_dann and dann_epochs > 0:
        import torch

        print(f"[features] Fine-tuning DANN for {dann_epochs} epochs (ganin schedule)")
        model = _finetune_dann(cfg, model, loaders, device)
        X_train, y_train = _extract_forward(model, loaders["train_loader"], device)
        X_val, y_val = _extract_forward(model, loaders["val_loader"], device)
        X_test, y_test = _extract_forward(model, loaders["test_loader"], device)
        torch.save(model.state_dict(), feat_dir / "dann_convnext_finetuned.pt")

    # Persist artifacts + cache metadata.
    feat_dir.mkdir(parents=True, exist_ok=True)
    for split, X, y in zip(splits, (X_train, X_val, X_test), (y_train, y_val, y_test)):
        np.save(feat_dir / f"convnext_tiny_{split}.npy", X)
        np.save(feat_dir / f"y_{split}.npy", y)

    cache = FeatureCache(
        feat_hash=_feat_hash(X_train),
        config_hash=cfg_hash,
        git_sha=_git_sha(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    cache.save(meta_path)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }
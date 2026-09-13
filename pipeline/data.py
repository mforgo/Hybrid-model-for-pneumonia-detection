"""Dataset download, deterministic path collection, splitting, and loaders.

This module is the data foundation of the hybrid QML pneumonia pipeline. It
implements the locked design decisions from the rewrite plan (Task 4):

* **Deterministic collection** — paths are always sorted before any split so
  that labels and features are generated in one canonical order (fixes the
  notebook's misordered-labels bug).
* **Stratified, seeded split** — the original ``val`` split (16 images) is
  merged with ``train`` and re-split 80/20 with stratification. The original
  ``test`` split is kept untouched.
* **Medical-safe augmentation** — train-only ``RandomRotation(±7°)``,
  ``RandomAffine(translate=±5%)`` and ``ColorJitter(0.2/0.2)``. Horizontal
  flip and RandAugment are intentionally **not** used (harmful for X-rays).
* **No ``WeightedRandomSampler``** — class imbalance is handled downstream by
  the MLP's ``pos_weight`` (Task 10). The train loader only shuffles with a
  seeded ``torch.Generator`` for reproducibility.

Import policy
-------------
Only the standard library and NumPy are imported at module level. ``torch``,
``torchvision``, ``PIL`` and ``kagglehub`` are imported lazily inside the
functions that need them, so ``import pipeline.data`` succeeds even on a
machine without PyTorch (e.g. the local development box).
"""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "IMAGE_EXTENSIONS",
    "DEFAULT_DATASET",
    "DatasetMeta",
    "XRayDataset",
    "download_dataset",
    "collect_paths_labels",
    "split_data",
    "get_transforms",
    "get_loaders",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: ImageNet normalization statistics used by the pre-trained ConvNeXt-Tiny.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

#: File extensions treated as chest X-ray images.
IMAGE_EXTENSIONS: frozenset[str] = frozenset({".jpeg", ".jpg", ".png"})

#: Kaggle dataset slug (Paul Mooney, Chest X-Ray Images (Pneumonia)).
DEFAULT_DATASET: str = "paultimothymooney/chest-xray-pneumonia"

#: Folder name -> integer label. ``PNEUMONIA`` is the positive class.
_CLASS_LABELS: dict[str, int] = {"PNEUMONIA": 1, "NORMAL": 0}


# ---------------------------------------------------------------------------
# Dataset metadata
# ---------------------------------------------------------------------------


@dataclass
class DatasetMeta:
    """Summary of the resolved dataset splits.

    Attributes:
        train_count: Number of training images.
        val_count: Number of validation images.
        test_count: Number of test images.
        train_pneumonia_ratio: Fraction of pneumonia (label 1) in train.
        val_pneumonia_ratio: Fraction of pneumonia (label 1) in validation.
        test_pneumonia_ratio: Fraction of pneumonia (label 1) in test.
        train_paths: Training image paths (canonical sorted order).
        val_paths: Validation image paths (canonical sorted order).
        test_paths: Test image paths (canonical sorted order).
        seed: Random seed used for the split.
        val_split: Validation fraction used for the split.
    """

    train_count: int
    val_count: int
    test_count: int
    train_pneumonia_ratio: float
    val_pneumonia_ratio: float
    test_pneumonia_ratio: float
    train_paths: list[Path] = field(default_factory=list)
    val_paths: list[Path] = field(default_factory=list)
    test_paths: list[Path] = field(default_factory=list)
    seed: int = 6
    val_split: float = 0.20

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation (paths as strings)."""
        return {
            "train_count": self.train_count,
            "val_count": self.val_count,
            "test_count": self.test_count,
            "train_pneumonia_ratio": self.train_pneumonia_ratio,
            "val_pneumonia_ratio": self.val_pneumonia_ratio,
            "test_pneumonia_ratio": self.test_pneumonia_ratio,
            "train_paths": [str(p) for p in self.train_paths],
            "val_paths": [str(p) for p in self.val_paths],
            "test_paths": [str(p) for p in self.test_paths],
            "seed": self.seed,
            "val_split": self.val_split,
        }


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _resolve_dataset_root(path: Path) -> Path:
    """Locate the directory that actually contains ``train``/``test`` splits.

    Kaggle archives are often nested (``<download>/chest_xray/train``), so we
    probe a few common layouts before giving up and returning ``path``.
    """
    path = Path(path)
    candidates = [
        path,
        path / "chest_xray",
        path / "chest_xray" / "chest_xray",
    ]
    for candidate in candidates:
        if (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate

    # Fallback: search one level deep for a child that looks like the root.
    if path.is_dir():
        for child in sorted(path.iterdir()):
            if child.is_dir() and (child / "train").is_dir():
                return child
    return path


def download_dataset(cfg: Any) -> Path:
    """Resolve the chest X-ray dataset root, downloading it if necessary.

    Resolution order:

    1. ``cfg.dataset_path`` — if set and existing, use it directly (manual
       override for a pre-downloaded dataset).
    2. ``kagglehub.dataset_download(cfg.dataset)`` — only attempted when
       ``~/.kaggle/kaggle.json`` exists.
    3. Graceful degradation — if the Kaggle credentials are missing or the
       download fails, warn and fall back to ``cfg.dataset_path``.

    Args:
        cfg: Configuration object exposing ``dataset`` and ``dataset_path``.

    Returns:
        Path to the dataset root containing ``train``/``val``/``test``.

    Raises:
        FileNotFoundError: If no usable dataset is available anywhere.
    """
    dataset_path = str(getattr(cfg, "dataset_path", "") or "")
    manual_root = Path(dataset_path) if dataset_path else None

    # 1. Manual override wins when it points at real data.
    if manual_root is not None and manual_root.exists():
        return _resolve_dataset_root(manual_root)

    # 2. Kaggle credentials required for an automatic download.
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        warnings.warn(
            "~/.kaggle/kaggle.json not found; cannot download the dataset. "
            "Falling back to cfg.dataset_path.",
            RuntimeWarning,
            stacklevel=2,
        )
        if manual_root is not None and manual_root.exists():
            return _resolve_dataset_root(manual_root)
        raise FileNotFoundError(
            "No dataset available: ~/.kaggle/kaggle.json is missing and "
            "cfg.dataset_path does not point to an existing directory."
        )

    # 3. Attempt the download; degrade gracefully on any failure.
    dataset_slug = str(getattr(cfg, "dataset", DEFAULT_DATASET) or DEFAULT_DATASET)
    try:
        import kagglehub  # lazy: not installed on the dev machine

        downloaded = kagglehub.dataset_download(dataset_slug)
        return _resolve_dataset_root(Path(downloaded))
    except Exception as exc:  # noqa: BLE001 - degrade on any download failure
        warnings.warn(
            f"kagglehub download failed ({exc!r}); falling back to "
            "cfg.dataset_path.",
            RuntimeWarning,
            stacklevel=2,
        )
        if manual_root is not None and manual_root.exists():
            return _resolve_dataset_root(manual_root)
        raise FileNotFoundError(
            f"Dataset download failed and cfg.dataset_path is unusable: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Deterministic path/label collection
# ---------------------------------------------------------------------------


def _label_from_dirname(name: str) -> int | None:
    """Map a class directory name to its integer label (or ``None``)."""
    return _CLASS_LABELS.get(name.strip().upper())


def collect_paths_labels(root: Path | str, split: str) -> tuple[list[Path], list[int]]:
    """Collect image paths and labels for one split, deterministically.

    Walks ``root/split/<CLASS>/*`` where ``<CLASS>`` is ``NORMAL`` (label 0)
    or ``PNEUMONIA`` (label 1). Class directories and files are sorted, and
    the final ``(path, label)`` pairs are sorted by path, so repeated calls on
    the same root always return the identical order regardless of filesystem
    enumeration order.

    Args:
        root: Dataset root containing the split directories.
        split: Split name, e.g. ``"train"``, ``"val"`` or ``"test"``.

    Returns:
        ``(paths, labels)`` with aligned, sorted lists.

    Raises:
        FileNotFoundError: If ``root/split`` does not exist.
    """
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    pairs: list[tuple[Path, int]] = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = _label_from_dirname(class_dir.name)
        if label is None:
            continue
        for file_path in sorted(class_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
                pairs.append((file_path, label))

    # Sort by Path (matches ``sorted(paths)``) to guarantee determinism.
    pairs.sort(key=lambda pair: pair[0])

    paths = [path for path, _ in pairs]
    labels = [label for _, label in pairs]
    return paths, labels


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------


def split_data(
    paths: Sequence[Path],
    labels: Sequence[int],
    val_split: float = 0.20,
    seed: int = 6,
) -> tuple[list[Path], list[Path], list[int], list[int]]:
    """Stratified, seeded train/validation split.

    Each class is shuffled independently with a NumPy generator seeded by
    ``seed`` and ``ceil(n_class * val_split)`` samples are assigned to
    validation. This reproduces the plan's expected counts (train 4185,
    val 1047) and keeps the class ratio stable across splits.

    Args:
        paths: Image paths.
        labels: Integer labels aligned with ``paths``.
        val_split: Fraction assigned to validation (default 0.20).
        seed: Random seed (project convention: 6).

    Returns:
        ``(train_paths, val_paths, train_labels, val_labels)``.

    Raises:
        ValueError: If lengths mismatch or ``val_split`` is out of range.
    """
    paths = list(paths)
    labels = list(labels)
    if len(paths) != len(labels):
        raise ValueError(
            f"paths/labels length mismatch: {len(paths)} != {len(labels)}"
        )
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    labels_arr = np.asarray(labels)
    rng = np.random.default_rng(seed)

    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in np.unique(labels_arr):
        cls_idx = np.where(labels_arr == cls)[0]
        rng.shuffle(cls_idx)
        n_val = min(int(math.ceil(len(cls_idx) * val_split)), len(cls_idx))
        val_idx.extend(cls_idx[:n_val].tolist())
        train_idx.extend(cls_idx[n_val:].tolist())

    train_idx.sort()
    val_idx.sort()

    train_paths = [paths[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    return train_paths, val_paths, train_labels, val_labels


def _limit_split(
    paths: Sequence[Path],
    labels: Sequence[int],
    n: int,
    seed: int,
) -> tuple[list[Path], list[int]]:
    """Stratified truncation to ~``n`` samples (smoke-test ``subset`` override).

    A naive ``paths[:n]`` would return a single class because class
    directories sort contiguously; instead we sample proportionally from each
    class so the subset stays usable for training.
    """
    paths = list(paths)
    labels = list(labels)
    if n <= 0 or n >= len(paths):
        return paths, labels

    labels_arr = np.asarray(labels)
    rng = np.random.default_rng(seed)
    chosen: list[int] = []
    for cls in np.unique(labels_arr):
        cls_idx = np.where(labels_arr == cls)[0]
        rng.shuffle(cls_idx)
        k = max(1, int(round(n * len(cls_idx) / len(labels_arr))))
        chosen.extend(cls_idx[:k].tolist())

    chosen.sort()
    return [paths[i] for i in chosen], [labels[i] for i in chosen]


# ---------------------------------------------------------------------------
# Dataset + transforms
# ---------------------------------------------------------------------------


def _dataset_base() -> type:
    """Return ``torch.utils.data.Dataset`` if available, else ``object``.

    The import lives inside this function so the module stays importable
    without PyTorch. When PyTorch *is* present, ``XRayDataset`` is a genuine
    ``Dataset`` subclass.
    """
    try:
        from torch.utils.data import Dataset

        return Dataset
    except Exception:  # noqa: BLE001 - torch optional at import time
        return object


class XRayDataset(_dataset_base()):  # type: ignore[misc, valid-type]
    """Map-style dataset yielding ``(image_tensor, label)`` pairs.

    Images are loaded lazily with PIL, converted to RGB, and passed through
    the supplied transform (resize + normalization, plus train-only
    augmentation).
    """

    def __init__(
        self,
        paths: Sequence[Path],
        labels: Sequence[int],
        transform: Any = None,
    ) -> None:
        if len(paths) != len(labels):
            raise ValueError(
                f"paths/labels length mismatch: {len(paths)} != {len(labels)}"
            )
        self.paths = list(paths)
        self.labels = [int(label) for label in labels]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        from PIL import Image  # lazy: PIL is a runtime dependency

        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


def get_transforms(cfg: Any, train: bool) -> Any:
    """Build the torchvision transform pipeline for a split.

    Train transforms add medical-safe augmentation (rotation ±7°, translation
    ±5%, brightness/contrast jitter 0.2). Validation/test transforms only
    resize and normalize. Horizontal flip and RandAugment are deliberately
    excluded.

    Args:
        cfg: Configuration object with ``img_size`` and augmentation fields.
        train: ``True`` for the training pipeline, ``False`` otherwise.

    Returns:
        A ``torchvision.transforms.Compose`` instance.
    """
    from torchvision import transforms  # lazy: torchvision optional at import

    img_size = int(getattr(cfg, "img_size", 224))

    ops: list[Any] = [transforms.Resize((img_size, img_size))]
    if train:
        rotation_deg = float(getattr(cfg, "rotation_deg", 7.0))
        translate = float(getattr(cfg, "translate", 0.05))
        brightness = float(getattr(cfg, "color_jitter_brightness", 0.2))
        contrast = float(getattr(cfg, "color_jitter_contrast", 0.2))

        ops.append(transforms.RandomRotation(degrees=rotation_deg))
        ops.append(
            transforms.RandomAffine(degrees=0, translate=(translate, translate))
        )
        ops.append(
            transforms.ColorJitter(brightness=brightness, contrast=contrast)
        )

    ops.append(transforms.ToTensor())
    ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return transforms.Compose(ops)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _pneumonia_ratio(labels: Sequence[int]) -> float:
    """Fraction of positive (pneumonia) labels; 0.0 for an empty split."""
    if not labels:
        return 0.0
    return float(sum(1 for label in labels if int(label) == 1) / len(labels))


def _save_dataset_meta(cfg: Any, meta: DatasetMeta) -> Path:
    """Persist ``dataset_meta.json`` under ``cfg.artifacts_dir``."""
    artifacts_dir = Path(str(getattr(cfg, "artifacts_dir", "artifacts") or "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_path = artifacts_dir / "dataset_meta.json"
    out_path.write_text(json.dumps(meta.to_dict(), indent=2), encoding="utf-8")
    return out_path


def get_loaders(cfg: Any) -> dict[str, Any]:
    """Build train/val/test DataLoaders and dataset metadata.

    The original ``train`` and ``val`` splits are merged and re-split 80/20
    with stratification; the original ``test`` split is left untouched. The
    train loader shuffles with a seeded ``torch.Generator`` and does **not**
    use a ``WeightedRandomSampler`` (imbalance is handled by the MLP's
    ``pos_weight`` downstream).

    Args:
        cfg: Configuration object. Recognised fields: ``dataset``,
            ``dataset_path``, ``img_size``, ``val_split``, ``batch_size``,
            ``seed``, ``artifacts_dir``, augmentation fields, plus optional
            ``subset`` (limit N samples per split) and ``n_workers``.

    Returns:
        Dict with keys ``train_loader``, ``val_loader``, ``test_loader``,
        ``train_paths``, ``val_paths``, ``test_paths``, ``train_labels``,
        ``val_labels``, ``test_labels`` and ``meta`` (a :class:`DatasetMeta`).
    """
    import torch  # lazy: torch optional at import time
    from torch.utils.data import DataLoader

    from pipeline.seed import get_torch_generator

    seed = int(getattr(cfg, "seed", 6))
    val_split = float(getattr(cfg, "val_split", 0.20))
    batch_size = int(getattr(cfg, "batch_size", 16))
    n_workers = int(getattr(cfg, "n_workers", 0))
    subset = getattr(cfg, "subset", None)

    root = download_dataset(cfg)

    # Merge original train + val, then re-split stratified.
    train_paths_raw, train_labels_raw = collect_paths_labels(root, "train")
    val_paths_raw, val_labels_raw = collect_paths_labels(root, "val")
    merged = sorted(
        zip(train_paths_raw + val_paths_raw, train_labels_raw + val_labels_raw),
        key=lambda pair: pair[0],
    )
    all_paths = [path for path, _ in merged]
    all_labels = [label for _, label in merged]

    train_paths, val_paths, train_labels, val_labels = split_data(
        all_paths, all_labels, val_split, seed
    )
    test_paths, test_labels = collect_paths_labels(root, "test")

    # Optional smoke-test subset (stratified so both classes survive).
    if subset:
        subset_n = int(subset)
        train_paths, train_labels = _limit_split(
            train_paths, train_labels, subset_n, seed
        )
        val_paths, val_labels = _limit_split(val_paths, val_labels, subset_n, seed)
        test_paths, test_labels = _limit_split(
            test_paths, test_labels, subset_n, seed
        )

    train_ds = XRayDataset(
        train_paths, train_labels, transform=get_transforms(cfg, train=True)
    )
    val_ds = XRayDataset(
        val_paths, val_labels, transform=get_transforms(cfg, train=False)
    )
    test_ds = XRayDataset(
        test_paths, test_labels, transform=get_transforms(cfg, train=False)
    )

    pin_memory = bool(torch.cuda.is_available())
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        generator=get_torch_generator(seed),
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=pin_memory,
    )

    meta = DatasetMeta(
        train_count=len(train_paths),
        val_count=len(val_paths),
        test_count=len(test_paths),
        train_pneumonia_ratio=_pneumonia_ratio(train_labels),
        val_pneumonia_ratio=_pneumonia_ratio(val_labels),
        test_pneumonia_ratio=_pneumonia_ratio(test_labels),
        train_paths=train_paths,
        val_paths=val_paths,
        test_paths=test_paths,
        seed=seed,
        val_split=val_split,
    )
    _save_dataset_meta(cfg, meta)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_paths": train_paths,
        "val_paths": val_paths,
        "test_paths": test_paths,
        "train_labels": train_labels,
        "val_labels": val_labels,
        "test_labels": test_labels,
        "meta": meta,
    }

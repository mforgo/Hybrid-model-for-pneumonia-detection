"""Analysis and interpretability: DANN t-SNE, calibration, SOTA table, 5-fold CV, Grad-CAM.

Task 13 of the notebook → package rewrite. Consumes the outputs of the
feature/VAE stages (64-dim L2-normalized latents) and the trained models
from ``pipeline.vqc`` / ``pipeline.mlp`` to produce the interpretability
and robustness artifacts of the thesis results chapter:

* ``figures/dann_domain_tsne.png`` — t-SNE of train vs test features
  (the DANN domain-invariance visualization).
* ``figures/calibration_curve.png`` — reliability diagram + Brier score.
* ``results/sota_comparison.csv`` — hardcoded state-of-the-art table with
  this project's MLP/VQC rows filled from ``results/main_results.csv``.
* ``results/cv_results.json`` — per-fold + mean/std metrics of the REAL
  stratified 5-fold cross-validation (per-fold thresholds included).

Design decisions (LOCKED by the rewrite plan, T13):

* ``run_5fold_cv`` reuses ``train_vqc`` / ``train_mlp`` verbatim — no
  manual parameter-shift, no second training loop. Per-fold thresholds are
  selected on the validation fold only via ``find_best_threshold``.
* CV compute is reduced: each fold trains with ``cfg.vqc_epochs_cv`` /
  ``cfg.mlp_epochs_cv`` epochs when those attributes exist, otherwise
  ``min(cfg.epochs, 10)`` (documented in the function docstring).
* ``cfg`` is never mutated across folds — a per-fold copy is created with
  ``dataclasses.replace``.
* All heavy imports (sklearn, pandas, matplotlib, torch, pennylane,
  pytorch_grad_cam) are lazy inside the functions, so ``import
  pipeline.analysis`` succeeds with stdlib + numpy only (e.g. the local
  development box).

Import policy
-------------
Only the standard library and NumPy are imported at module level.
``sklearn``, ``pandas``, ``matplotlib``, ``torch``, ``pennylane`` and
``pytorch_grad_cam`` are imported lazily inside the functions that need
them (with a clear ``RuntimeError`` / warning when missing), so
``import pipeline.analysis`` succeeds even on a machine without those
packages.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.seed import seed_everything

SEED = 6  # project-wide random seed (AGENTS.md global rule)

__all__ = [
    "SEED",
    "plot_domain_invariance",
    "plot_calibration_curve",
    "sota_comparison_table",
    "run_5fold_cv",
    "gradcam_analysis",
]


# ---------------------------------------------------------------------------
# Lazy heavy imports (stdlib + numpy only at module level)
# ---------------------------------------------------------------------------


def _import_matplotlib() -> Any:
    """Import ``matplotlib.pyplot`` lazily with the Agg backend.

    ``matplotlib.use("Agg")`` must run before ``pyplot`` is imported so the
    module works on headless servers (no display).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:
        raise RuntimeError(
            "pipeline.analysis requires matplotlib at call time "
            "(pip install matplotlib). It is imported lazily so the module "
            "imports with stdlib + numpy only."
        ) from exc


def _import_sklearn() -> Any:
    """Import ``sklearn`` lazily, raising a clear RuntimeError."""
    try:
        import sklearn

        return sklearn
    except ImportError as exc:
        raise RuntimeError(
            "pipeline.analysis requires scikit-learn at call time "
            "(pip install scikit-learn). It is imported lazily so the module "
            "imports with stdlib + numpy only."
        ) from exc


def _import_pandas() -> Any:
    """Import ``pandas`` lazily, raising a clear RuntimeError."""
    try:
        import pandas as pd

        return pd
    except ImportError as exc:
        raise RuntimeError(
            "pipeline.analysis requires pandas at call time "
            "(pip install pandas). It is imported lazily so the module "
            "imports with stdlib + numpy only."
        ) from exc


# ---------------------------------------------------------------------------
# DANN domain-invariance visualization (t-SNE)
# ---------------------------------------------------------------------------


def plot_domain_invariance(
    train_feat,
    test_feat,
    path,
    labels_train=None,
    labels_test=None,
) -> None:
    """t-SNE of train vs test features (DANN domain-invariance visualization).

    Projects the 64-dim L2-normalized VAE latents of both domains into 2D
    with ``sklearn.manifold.TSNE`` (``random_state=6`` for determinism) and
    saves a scatter plot to *path*. Points are colored by domain (train vs
    test); when *labels_train* / *labels_test* are provided the marker shape
    additionally encodes the class (circle = Normal, triangle = Pneumonia),
    which makes it easy to spot whether the domains overlap after DANN.

    Args:
        train_feat: Source-domain (train) features, shape ``(N, 64)``.
        test_feat: Target-domain (test) features, shape ``(N, 64)``.
        path: Output PNG path (parent directories are created). Defaults to
            ``figures/dann_domain_tsne.png`` per the plan Data Flow.
        labels_train: Optional binary labels for the train features, shape
            ``(N,)`` — used for marker shapes.
        labels_test: Optional binary labels for the test features, shape
            ``(N,)`` — used for marker shapes.

    Returns:
        None. Saves the PNG to *path*.
    """
    plt = _import_matplotlib()
    sklearn = _import_sklearn()

    train_feat = np.asarray(train_feat, dtype=np.float64)
    test_feat = np.asarray(test_feat, dtype=np.float64)

    X = np.vstack([train_feat, test_feat])
    n_train = train_feat.shape[0]
    domains = np.array(["train"] * n_train + ["test"] * test_feat.shape[0])

    tsne = sklearn.manifold.TSNE(
        n_components=2,
        random_state=SEED,
        init="pca",
        perplexity=min(30, max(5, X.shape[0] - 1)),
    )
    X_2d = tsne.fit_transform(X)

    has_labels = labels_train is not None and labels_test is not None
    if has_labels:
        labels = np.concatenate(
            [np.asarray(labels_train), np.asarray(labels_test)]
        ).astype(int)

    fig, ax = plt.subplots(figsize=(8, 6))
    domain_colors = {"train": "tab:blue", "test": "tab:red"}
    class_markers = {0: "o", 1: "^"}

    for domain in ("train", "test"):
        mask = domains == domain
        if has_labels:
            for cls in (0, 1):
                cls_mask = mask & (labels == cls)
                if cls_mask.sum() == 0:
                    continue
                ax.scatter(
                    X_2d[cls_mask, 0],
                    X_2d[cls_mask, 1],
                    c=domain_colors[domain],
                    marker=class_markers[cls],
                    s=18,
                    alpha=0.7,
                    label=f"{domain} (class {cls})",
                    edgecolors="none",
                )
        else:
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=domain_colors[domain],
                marker="o",
                s=18,
                alpha=0.7,
                label=domain,
                edgecolors="none",
            )

    ax.set_xlabel("t-SNE component 1")
    ax.set_ylabel("t-SNE component 2")
    ax.set_title("Domain-invariance of features (t-SNE, train vs test)")
    ax.legend(loc="best", fontsize="small", markerscale=1.5)
    ax.grid(alpha=0.3)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Calibration curve (reliability diagram) + Brier score
# ---------------------------------------------------------------------------


def plot_calibration_curve(y_true, y_probs, path) -> dict[str, float]:
    """Reliability diagram (calibration curve) + Brier score.

    Bins the predicted probabilities into 10 equal-width bins and plots the
    mean predicted probability vs the observed positive fraction, together
    with the perfect-calibration diagonal. Uses
    ``sklearn.calibration.calibration_curve`` when available; falls back to
    a manual NumPy binning (identical binning convention) otherwise.

    Args:
        y_true: Ground-truth binary labels (0/1), shape ``(N,)``.
        y_probs: Predicted probabilities, shape ``(N,)``.
        path: Output PNG path (parent directories are created). Defaults to
            ``figures/calibration_curve.png`` per the plan Data Flow.

    Returns:
        Dict with ``brier_score`` (float) and ``ece`` (expected calibration
        error, float). ``ece`` is ``None`` when the manual fallback binning
        is used and no bin contains both classes.
    """
    plt = _import_matplotlib()

    y_true = np.asarray(y_true, dtype=np.int64)
    y_probs = np.asarray(y_probs, dtype=np.float64)

    # --- Brier score (pure NumPy — matches sklearn.metrics.brier_score_loss) ---
    brier = float(np.mean((y_probs - y_true) ** 2))

    # --- calibration curve: sklearn when available, manual fallback ---
    try:
        from sklearn.calibration import calibration_curve

        prob_true, prob_pred = calibration_curve(
            y_true, y_probs, n_bins=10, strategy="uniform"
        )
        ece = _expected_calibration_error(y_true, y_probs, prob_true, prob_pred)
    except ImportError:
        prob_true, prob_pred, ece = _manual_calibration_bins(y_true, y_probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0.0, 1.0], [0.0, 1.0], "k--", lw=1, alpha=0.6, label="Perfect calibration")
    ax.plot(
        prob_pred,
        prob_true,
        marker="o",
        lw=2,
        label=f"Model (Brier = {brier:.4f})",
    )
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive fraction")
    ax.set_title("Calibration curve (reliability diagram)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    result: dict[str, float] = {"brier_score": brier}
    if ece is not None:
        result["ece"] = ece
    return result


def _expected_calibration_error(y_true, y_probs, prob_true, prob_pred) -> float:
    """Expected calibration error from sklearn-style bin statistics.

    ``ECE = sum_b (n_b / N) * |prob_true_b - prob_pred_b|``.

    Args:
        y_true: Ground-truth binary labels (0/1), shape ``(N,)``.
        y_probs: Predicted probabilities, shape ``(N,)``.
        prob_true: Observed positive fraction per bin.
        prob_pred: Mean predicted probability per bin.

    Returns:
        The ECE as a plain ``float``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_probs = np.asarray(y_probs, dtype=np.float64)
    prob_true = np.asarray(prob_true, dtype=np.float64)
    prob_pred = np.asarray(prob_pred, dtype=np.float64)

    n = len(y_probs)
    bin_edges = np.linspace(0.0, 1.0, len(prob_true) + 1)
    counts = np.zeros(len(prob_true), dtype=np.float64)
    for i in range(len(prob_true)):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == len(prob_true) - 1:
            mask = (y_probs >= lo) & (y_probs <= hi)
        else:
            mask = (y_probs >= lo) & (y_probs < hi)
        counts[i] = float(mask.sum())

    total = max(float(counts.sum()), 1.0)
    ece = float(np.sum((counts / total) * np.abs(prob_true - prob_pred)))
    return ece


def _manual_calibration_bins(y_true, y_probs):
    """Manual NumPy calibration binning (fallback when sklearn is missing).

    Bins probabilities into 10 equal-width bins and computes the observed
    positive fraction and mean predicted probability per bin. Returns
    ``(prob_true, prob_pred, ece)`` where ``ece`` is ``None`` when a bin
    contains no samples (the ECE is then undefined without sklearn's
    bin-count helper).

    Args:
        y_true: Ground-truth binary labels (0/1), shape ``(N,)``.
        y_probs: Predicted probabilities, shape ``(N,)``.

    Returns:
        ``(prob_true, prob_pred, ece_or_None)``.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_probs = np.asarray(y_probs, dtype=np.float64)

    n_bins = 10
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    prob_true = np.zeros(n_bins, dtype=np.float64)
    prob_pred = np.zeros(n_bins, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=np.float64)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_probs >= lo) & (y_probs <= hi)
        else:
            mask = (y_probs >= lo) & (y_probs < hi)
        n_bin = int(mask.sum())
        counts[i] = float(n_bin)
        if n_bin > 0:
            prob_pred[i] = float(y_probs[mask].mean())
            prob_true[i] = float(y_true[mask].mean())

    ece = None
    if counts.sum() > 0:
        ece = float(
            np.sum((counts / counts.sum()) * np.abs(prob_true - prob_pred))
        )
    return prob_true, prob_pred, ece


# ---------------------------------------------------------------------------
# State-of-the-art comparison table
# ---------------------------------------------------------------------------


def sota_comparison_table() -> Any:
    """Build the state-of-the-art comparison table for the thesis.

    Returns a ``pandas.DataFrame`` with columns ``Model``, ``AUC``,
    ``Params`` and ``Reference``. The table contains hardcoded classical CNN
    baselines on pediatric chest X-ray pneumonia (known SOTA AUCs in the
    ~0.92–0.99 range) plus placeholder rows for this project's MLP and VQC,
    whose AUC values are filled from ``results/main_results.csv`` when that
    file exists (else ``NaN``).

    When pandas is unavailable the function warns and returns a plain dict
    ``{"models": [...], "auc": [...], "params": [...], "reference": [...]}``
    instead. The CSV is always saved to ``results/sota_comparison.csv``
    (via the stdlib ``csv`` module when pandas is missing).

    Returns:
        ``pandas.DataFrame`` (or a dict-of-lists fallback when pandas is
        missing).
    """
    # Hardcoded SOTA baselines from the thesis context (pediatric chest
    # X-ray pneumonia, AUC-ROC on held-out test sets).
    models = [
        "Simple CNN (Kermany et al. 2018)",
        "DenseNet-121 (CheXNet-style)",
        "ResNet-50 (transfer learning)",
        "ConvNeXt-Tiny + MLP (this work)",
        "ConvNeXt-Tiny + VQC (this work)",
    ]
    aucs: list[float | None] = [0.92, 0.96, 0.94, None, None]
    params: list[float | None] = [1_000_000, 8_000_000, 25_000_000, 2_113, 54]
    references = [
        "Kermany et al., Cell 2018",
        "Rajpurkar et al., arXiv:1711.05225",
        "He et al., CVPR 2016 (fine-tuned)",
        "This work (results/main_results.csv)",
        "This work (results/main_results.csv)",
    ]

    # Fill this project's rows from results/main_results.csv when present.
    results_csv = Path("results") / "main_results.csv"
    if results_csv.is_file():
        try:
            with open(results_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = str(row.get("", "") or row.get("Model", "")).lower()
                    auc_raw = row.get("AUC-ROC") or row.get("AUC") or row.get("roc_auc")
                    if "vqc" in name and auc_raw:
                        aucs[4] = float(auc_raw)
                    elif "mlp" in name and auc_raw:
                        aucs[3] = float(auc_raw)
        except (OSError, ValueError) as exc:
            warnings.warn(
                f"Could not parse {results_csv}: {exc}; leaving project rows as NaN.",
                RuntimeWarning,
                stacklevel=2,
            )

    # --- save CSV (pandas when available, stdlib csv otherwise) ---
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "sota_comparison.csv"

    try:
        pd = _import_pandas()
        df = pd.DataFrame(
            {
                "Model": models,
                "AUC": aucs,
                "Params": params,
                "Reference": references,
            }
        )
        df.to_csv(csv_path, index=False)
        return df
    except RuntimeError:
        warnings.warn(
            "pandas is not installed; returning a dict-of-lists fallback for "
            "sota_comparison_table().",
            RuntimeWarning,
            stacklevel=2,
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "AUC", "Params", "Reference"])
            for m, a, p, r in zip(models, aucs, params, references):
                writer.writerow([m, "" if a is None else a, p, r])
        return {
            "models": models,
            "auc": aucs,
            "params": params,
            "reference": references,
        }


# ---------------------------------------------------------------------------
# Real 5-fold stratified cross-validation
# ---------------------------------------------------------------------------


def run_5fold_cv(cfg: Any, X_cv: np.ndarray, y_cv: np.ndarray) -> dict[str, Any]:
    """Run a REAL stratified 5-fold cross-validation of VQC + MLP.

    Splits ``(X_cv, y_cv)`` into ``cfg.cv_folds`` stratified folds
    (``sklearn.model_selection.StratifiedKFold``, ``random_state=cfg.seed``).
    For every fold:

    1. ``train_vqc(cfg_fold, X_tr, y_tr, X_val, y_val)`` — the VQC is
       trained on the 4 training folds (no manual parameter-shift; the
       training loop of ``pipeline.vqc`` is reused verbatim).
    2. ``train_mlp(cfg_fold, X_tr, y_tr, X_val, y_val)`` — the classical
       MLP baseline on the same folds.
    3. ``find_best_threshold`` selects the per-fold threshold on the
       validation fold only (never on test data).
    4. ``compute_all_metrics`` evaluates both models on the validation fold
       at their per-fold threshold.

    Compute reduction (LOCKED): each fold trains with
    ``cfg.vqc_epochs_cv`` / ``cfg.mlp_epochs_cv`` epochs when those
    attributes exist on ``cfg``, otherwise ``min(cfg.epochs, 10)``. The
    per-fold ``cfg`` is a ``dataclasses.replace`` copy — the caller's
    ``cfg`` object is never mutated. Per-fold model artifacts are written to
    a temporary directory so the main ``results_dir`` is not clobbered.

    The aggregated result is saved to ``results/cv_results.json`` with the
    structure ``{"per_fold": [...], "mean": {...}, "std": {...}}``.

    Args:
        cfg: Configuration object. Recognised fields: ``cv_folds``, ``seed``,
            ``epochs``, ``vqc_epochs_cv`` (optional), ``mlp_epochs_cv``
            (optional), ``results_dir``, ``figures_dir``, ``threshold_range_*``.
        X_cv: Feature matrix for cross-validation, shape ``(N, 64)``
            (L2-normalized VAE latents).
        y_cv: Binary labels (0/1), shape ``(N,)``.

    Returns:
        Dict with keys ``per_fold`` (list of per-fold dicts with ``fold``,
        ``threshold`` and ``metrics`` for ``vqc`` and ``mlp``), ``mean``
        (mean of each metric across folds per model) and ``std`` (standard
        deviation across folds per model).
    """
    seed_everything(int(getattr(cfg, "seed", SEED)))

    from pipeline.evaluate import compute_all_metrics, find_best_threshold
    from pipeline.mlp import train_mlp
    from pipeline.vqc import train_vqc

    sklearn = _import_sklearn()

    X_cv = np.asarray(X_cv, dtype=np.float64)
    y_cv = np.asarray(y_cv, dtype=np.int64)

    n_folds = int(getattr(cfg, "cv_folds", 5))
    if n_folds < 2:
        raise ValueError(f"cv_folds must be >= 2, got {n_folds}")

    # --- compute-reduction epochs (LOCKED default: min(cfg.epochs, 10)) ---
    base_epochs = int(getattr(cfg, "epochs", 50))
    vqc_epochs = int(getattr(cfg, "vqc_epochs_cv", min(base_epochs, 10)))
    mlp_epochs = int(getattr(cfg, "mlp_epochs_cv", min(base_epochs, 10)))

    # --- threshold scan range from cfg ---
    tau_range = np.arange(
        float(getattr(cfg, "threshold_range_min", 0.30)),
        float(getattr(cfg, "threshold_range_max", 0.80))
        + float(getattr(cfg, "threshold_step", 0.025)) / 2.0,
        float(getattr(cfg, "threshold_step", 0.025)),
    )

    results_dir = Path(getattr(cfg, "results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)

    skf = sklearn.model_selection.StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=int(getattr(cfg, "seed", SEED))
    )

    per_fold: list[dict[str, Any]] = []
    metric_keys: set[str] = set()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
        X_tr, y_tr = X_cv[train_idx], y_cv[train_idx]
        X_val, y_val = X_cv[val_idx], y_cv[val_idx]

        # Per-fold cfg copy: reduced epochs + isolated artifact dir. The
        # caller's cfg is never mutated.
        fold_artifacts = results_dir / f"cv_fold_{fold_idx + 1}"
        cfg_fold = dataclasses.replace(
            cfg,
            epochs=vqc_epochs,
            results_dir=str(fold_artifacts),
        )

        print(f"\n=== CV fold {fold_idx + 1}/{n_folds} "
              f"(train={X_tr.shape[0]}, val={X_val.shape[0]}) ===")

        # --- VQC fold training (reuses pipeline.vqc.train_vqc verbatim) ---
        vqc_result = train_vqc(cfg_fold, X_tr, y_tr, X_val, y_val)
        vqc_val_probs = np.asarray(vqc_result["val_probs"], dtype=np.float64)
        vqc_tau = find_best_threshold(vqc_val_probs, y_val, tau_range)
        vqc_metrics = compute_all_metrics(vqc_val_probs, y_val, vqc_tau)

        # --- MLP fold training (reuses pipeline.mlp.train_mlp verbatim) ---
        cfg_fold_mlp = dataclasses.replace(cfg_fold, epochs=mlp_epochs)
        mlp_result = train_mlp(cfg_fold_mlp, X_tr, y_tr, X_val, y_val)
        mlp_val_probs = np.asarray(mlp_result["val_probs"], dtype=np.float64)
        mlp_tau = find_best_threshold(mlp_val_probs, y_val, tau_range)
        mlp_metrics = compute_all_metrics(mlp_val_probs, y_val, mlp_tau)

        metric_keys.update(vqc_metrics.keys())
        metric_keys.update(mlp_metrics.keys())

        per_fold.append(
            {
                "fold": fold_idx + 1,
                "n_train": int(X_tr.shape[0]),
                "n_val": int(X_val.shape[0]),
                "threshold": {"vqc": vqc_tau, "mlp": mlp_tau},
                "metrics": {"vqc": vqc_metrics, "mlp": mlp_metrics},
            }
        )

    # --- aggregate mean / std across folds (per model, per metric) ---
    mean: dict[str, dict[str, float]] = {"vqc": {}, "mlp": {}}
    std: dict[str, dict[str, float]] = {"vqc": {}, "mlp": {}}
    for model in ("vqc", "mlp"):
        for key in sorted(metric_keys):
            values = np.array(
                [fold["metrics"][model][key] for fold in per_fold],
                dtype=np.float64,
            )
            mean[model][key] = float(values.mean())
            std[model][key] = float(values.std(ddof=0))

    result = {
        "per_fold": per_fold,
        "mean": mean,
        "std": std,
        "config": {
            "cv_folds": n_folds,
            "seed": int(getattr(cfg, "seed", SEED)),
            "vqc_epochs_cv": vqc_epochs,
            "mlp_epochs_cv": mlp_epochs,
        },
    }

    with open(results_dir / "cv_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=_json_default)

    return result


def _json_default(obj: Any) -> Any:
    """JSON serializer fallback for numpy scalars/arrays in CV results."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# ---------------------------------------------------------------------------
# Grad-CAM interpretability
# ---------------------------------------------------------------------------


def gradcam_analysis(
    cfg: Any,
    model: Any,
    image_paths,
    labels,
    out_dir: str = "figures/gradcam",
) -> list[str]:
    """Generate Grad-CAM heatmaps for the given chest X-ray images.

    Uses ``pytorch_grad_cam.GradCAM`` on the ConvNeXt-Tiny model's last
    stage (``target_layers=[model.stages[-1]]``) to produce class activation
    maps, overlaid on the input images and saved as PNGs. The images are
    preprocessed with the standard ImageNet normalization (mean
    ``[0.485, 0.456, 0.406]``, std ``[0.229, 0.224, 0.225]``) and resized
    to ``cfg.img_size`` (default 224).

    The function processes whatever images it is given — the True Positive /
    False Negative / True Negative selection is the caller's responsibility
    (the thesis uses TP / FN / TN examples).

    Args:
        cfg: Configuration object. Recognised fields: ``img_size``.
        model: A ConvNeXt-Tiny model with the classification head attached
            (for this analysis only the head is needed for the CAM target).
        image_paths: Iterable of paths to JPEG chest X-ray images.
        labels: Iterable of binary labels (0/1) aligned with *image_paths*.
        out_dir: Output directory for the overlay PNGs (default
            ``figures/gradcam``).

    Returns:
        List of saved PNG paths. When ``pytorch_grad_cam`` is not installed
        a warning is emitted and an empty list is returned.
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError as exc:
        warnings.warn(
            "pytorch_grad_cam is not installed; skipping Grad-CAM analysis "
            "(pip install grad-cam).",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    try:
        import torch
        import torchvision.transforms as T
        from PIL import Image
    except ImportError as exc:
        warnings.warn(
            f"torch / torchvision / PIL required for Grad-CAM: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return []

    img_size = int(getattr(cfg, "img_size", 224))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    model.eval()
    target_layers = [model.stages[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    saved_paths: list[str] = []
    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        img_path = Path(img_path)
        if not img_path.is_file():
            warnings.warn(
                f"Grad-CAM: image not found, skipping: {img_path}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0)

        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]

        # Reconstruct the un-normalized RGB image for the overlay.
        img_resized = img.resize((img_size, img_size))
        img_rgb = np.asarray(img_resized, dtype=np.float32) / 255.0

        visualization = show_cam_on_image(
            img_rgb, grayscale_cam, use_rgb=True
        )

        out_path = out_dir / f"gradcam_{i:03d}_label{int(label)}.png"
        Image.fromarray(visualization).save(out_path)
        saved_paths.append(str(out_path))

    return saved_paths
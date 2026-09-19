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
    "domain_discriminator_accuracy",
    "run_dann_validation",
    "ablation_shift",
    "run_analysis",
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
    params: list[float | None] = [1_000_000, 8_000_000, 25_000_000, 2_113, 62]
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
                    name = str(
                        row.get("model") or row.get("Model") or row.get("", "") or ""
                    ).lower()
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
# DANN quantitative validation (domain-discriminator accuracy)
# ---------------------------------------------------------------------------


def domain_discriminator_accuracy(
    X_source: np.ndarray,
    X_target: np.ndarray,
    n_folds: int = 5,
    seed: int = 6,
) -> dict[str, float] | None:
    """Train a domain discriminator and report its cross-validated accuracy.

    The quantitative DANN proof: a ``LogisticRegression`` is trained to
    classify domain 0 (source/train) vs domain 1 (target/test) on the given
    feature matrix. If the features are domain-invariant (the DANN goal) the
    discriminator cannot do better than chance and the accuracy approaches
    0.5; a large pre-DANN accuracy that drops toward 0.5 post-DANN is the
    quantitative evidence that DANN mitigated the dataset shift.

    ``sklearn`` is imported lazily. When it is missing the function warns and
    returns ``None`` instead of raising — the analysis stage must stay alive
    on machines without scikit-learn (documented choice).

    Args:
        X_source: Source-domain (train) features, shape ``(N, 768)``.
        X_target: Target-domain (test) features, shape ``(M, 768)``.
        n_folds: Number of stratified cross-validation folds (default 5).
        seed: Random seed for the fold split and the classifier (default 6).

    Returns:
        Dict with ``acc_mean`` (mean fold accuracy), ``acc_std`` (std across
        folds) and ``n_folds``, or ``None`` when scikit-learn is unavailable.
    """
    seed_everything(int(seed))
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
    except ImportError as exc:
        warnings.warn(
            "scikit-learn is not installed; skipping domain-discriminator "
            "accuracy (pip install scikit-learn).",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    X_source = np.asarray(X_source, dtype=np.float64)
    X_target = np.asarray(X_target, dtype=np.float64)
    X = np.vstack([X_source, X_target])
    y = np.concatenate(
        [
            np.zeros(X_source.shape[0], dtype=np.int64),
            np.ones(X_target.shape[0], dtype=np.int64),
        ]
    )

    n_folds = int(n_folds)
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=int(seed))
    accs: list[float] = []
    for train_idx, test_idx in skf.split(X, y):
        clf = LogisticRegression(random_state=int(seed), max_iter=1000)
        clf.fit(X[train_idx], y[train_idx])
        accs.append(float(clf.score(X[test_idx], y[test_idx])))

    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs, ddof=0)),
        "n_folds": n_folds,
    }


def _plot_dann_discriminator(result: dict[str, Any], path: Path) -> None:
    """Bar chart of domain-discriminator accuracy per feature stage.

    Plots ``acc_mean`` with ``acc_std`` error bars for the stages present in
    *result* (``pre_dann``, ``post_dann``, ``post_vae``) and a dashed 0.5
    chance line. Stages whose discriminator result is ``None`` (sklearn
    missing) are skipped.
    """
    plt = _import_matplotlib()

    stages = [
        k for k in ("pre_dann", "post_dann", "post_vae")
        if k in result and result[k] is not None
    ]
    if not stages:
        warnings.warn(
            "No domain-discriminator results to plot.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    labels = {
        "pre_dann": "Pre-DANN (raw)",
        "post_dann": "Post-DANN",
        "post_vae": "Post-VAE",
    }
    means = [float(result[s]["acc_mean"]) for s in stages]
    stds = [float(result[s]["acc_std"]) for s in stages]

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(stages))
    ax.bar(x, means, yerr=stds, capsize=4, color="tab:blue", alpha=0.85)
    ax.axhline(0.5, color="k", ls="--", lw=1, alpha=0.7, label="Chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s] for s in stages])
    ax.set_ylabel("Domain-discriminator accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Domain-discriminator accuracy by feature stage")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="y")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_dann_validation(cfg: Any) -> dict[str, Any]:
    """Quantify the DANN effect with a domain discriminator.

    Loads the pre-DANN raw features (``convnext_tiny_raw_{train,test}.npy``)
    and the post-DANN features (``convnext_tiny_{train,test}.npy``) from
    ``cfg.artifacts_dir/features`` and runs
    :func:`domain_discriminator_accuracy` on both, plus on the VAE features
    (``vae_{train,test}.npy``) when present. The pre→post accuracy drop
    toward 0.5 is the quantitative proof that DANN mitigated the dataset
    shift (74.2% → 62.5% pneumonia prevalence).

    Saves ``results/dann_discriminator.json`` and
    ``figures/dann_discriminator.png`` (bar chart with a 0.5 chance line).

    Graceful degradation: when the raw (pre-DANN) files are missing — raw
    saving is new, so older feature caches lack them — a warning is emitted
    and only the post-DANN part is returned. When the post-DANN files are
    missing the function warns and returns ``{}``.

    Args:
        cfg: Configuration object. Recognised fields: ``artifacts_dir``,
            ``results_dir``, ``figures_dir``.

    Returns:
        Dict with keys ``pre_dann``, ``post_dann`` and (when VAE features
        exist) ``post_vae``, each mapping to the
        :func:`domain_discriminator_accuracy` result (or ``None`` when
        sklearn is unavailable).
    """
    feat_dir = Path(getattr(cfg, "artifacts_dir", "artifacts")) / "features"
    results_dir = Path(getattr(cfg, "results_dir", "results"))
    figures_dir = Path(getattr(cfg, "figures_dir", "figures"))

    post_train = feat_dir / "convnext_tiny_train.npy"
    post_test = feat_dir / "convnext_tiny_test.npy"
    if not (post_train.is_file() and post_test.is_file()):
        warnings.warn(
            f"run_dann_validation: post-DANN features not found in {feat_dir}; "
            "skipping DANN validation.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {}

    result: dict[str, Any] = {}
    result["post_dann"] = domain_discriminator_accuracy(
        np.load(post_train), np.load(post_test)
    )

    raw_train = feat_dir / "convnext_tiny_raw_train.npy"
    raw_test = feat_dir / "convnext_tiny_raw_test.npy"
    if raw_train.is_file() and raw_test.is_file():
        result["pre_dann"] = domain_discriminator_accuracy(
            np.load(raw_train), np.load(raw_test)
        )
    else:
        warnings.warn(
            "run_dann_validation: pre-DANN raw features "
            "(convnext_tiny_raw_{train,test}.npy) not found in "
            f"{feat_dir}; returning the post-DANN part only. Re-run the "
            "features stage with save_raw_features=True to enable the "
            "pre/post comparison.",
            RuntimeWarning,
            stacklevel=2,
        )

    vae_train = feat_dir / "vae_train.npy"
    vae_test = feat_dir / "vae_test.npy"
    if vae_train.is_file() and vae_test.is_file():
        result["post_vae"] = domain_discriminator_accuracy(
            np.load(vae_train), np.load(vae_test)
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "dann_discriminator.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=_json_default)

    try:
        _plot_dann_discriminator(result, figures_dir / "dann_discriminator.png")
    except RuntimeError as exc:
        warnings.warn(
            f"Could not plot DANN discriminator figure: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )

    return result


# ---------------------------------------------------------------------------
# Dataset-shift vs expressibility ablation (two-run manifest comparison)
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    """Coerce *value* to ``float``, returning ``None`` when not numeric."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _balanced_accuracy_np(labels, preds) -> float:
    """Balanced accuracy ``(recall + specificity) / 2`` (matches sklearn)."""
    labels = np.asarray(labels, dtype=np.int64)
    preds = np.asarray(preds, dtype=np.int64)
    tp = float(np.sum((preds == 1) & (labels == 1)))
    tn = float(np.sum((preds == 0) & (labels == 0)))
    fp = float(np.sum((preds == 1) & (labels == 0)))
    fn = float(np.sum((preds == 0) & (labels == 1)))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (recall + specificity)


def _manifest_balance(manifest: dict[str, Any], run_dir: Path) -> str:
    """Resolve the ``balance`` identifier of a run manifest.

    Prefers the manifest's stored config fields (``config.class_balance`` /
    ``config.balance`` — future-proofing for enriched manifests), then a
    top-level ``class_balance`` key, then infers from the results directory
    name: ``"balanced"`` when the name contains ``"balanced"``, else
    ``"imbalanced"``.
    """
    config = manifest.get("config", {})
    if isinstance(config, dict):
        for key in ("class_balance", "balance"):
            if config.get(key) is not None:
                return str(config[key])
    if manifest.get("class_balance") is not None:
        return str(manifest["class_balance"])
    name = str(run_dir).lower()
    return "balanced" if "balanced" in name else "imbalanced"


def _manifest_artifact(manifest: dict[str, Any], run_dir: Path, name: str) -> Path:
    """Resolve an artifact path from the manifest, falling back to *run_dir*."""
    artifacts = manifest.get("artifacts", {})
    if isinstance(artifacts, dict) and name in artifacts:
        return Path(artifacts[name])
    return run_dir / name


def _test_metrics(
    model: str,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    run_dir: Path,
    cfg: Any,
) -> tuple[float | None, float | None]:
    """Test-set AUC and balanced accuracy for one model of a run.

    Prefers test metrics stored in the manifest (``test_metrics`` section or
    ``test_roc_auc`` / ``test_balanced_accuracy`` keys — future-proofing),
    then computes them from the saved ``{model}_test_probs.npy`` and the
    ``y_test.npy`` labels (pure analysis/IO — no training). The manifest
    threshold (selected on validation) is applied to the test probabilities
    for the balanced accuracy, matching the locked evaluation protocol.
    Returns ``(None, None)`` when neither source is available.
    """
    test_metrics = manifest.get("test_metrics", {})
    if isinstance(test_metrics, dict) and model in test_metrics:
        tm = test_metrics[model]
        return _safe_float(tm.get("roc_auc")), _safe_float(tm.get("balanced_accuracy"))
    if "test_roc_auc" in metrics:
        return (
            _safe_float(metrics.get("test_roc_auc")),
            _safe_float(metrics.get("test_balanced_accuracy")),
        )

    probs_path = _manifest_artifact(manifest, run_dir, f"{model}_test_probs.npy")
    y_test_path = (
        Path(getattr(cfg, "artifacts_dir", "artifacts")) / "features" / "y_test.npy"
    )
    if not (probs_path.is_file() and y_test_path.is_file()):
        return None, None

    probs = np.load(probs_path)
    y_test = np.load(y_test_path)
    if len(probs) != len(y_test):
        return None, None

    test_auc: float | None = None
    try:
        from sklearn.metrics import roc_auc_score

        test_auc = float(roc_auc_score(y_test, probs))
    except (ImportError, ValueError):
        test_auc = None

    tau = _safe_float(metrics.get("threshold"))
    bal_acc_test: float | None = None
    if tau is not None:
        bal_acc_test = _balanced_accuracy_np(y_test, (probs > tau).astype(int))

    return test_auc, bal_acc_test


def _ablation_row(
    model: str,
    balance: str,
    metrics: dict[str, Any],
    manifest: dict[str, Any],
    run_dir: Path,
    cfg: Any,
) -> dict[str, Any]:
    """Build one ``{model, balance, val_auc, test_auc, auc_drop, bal_acc_test}`` row."""
    val_auc = _safe_float(metrics.get("roc_auc"))
    test_auc, bal_acc_test = _test_metrics(model, metrics, manifest, run_dir, cfg)
    auc_drop = val_auc - test_auc if (val_auc is not None and test_auc is not None) else None
    return {
        "model": model,
        "balance": balance,
        "val_auc": val_auc,
        "test_auc": test_auc,
        "auc_drop": auc_drop,
        "bal_acc_test": bal_acc_test,
    }


def ablation_shift(cfg: Any) -> dict[str, Any] | None:
    """Decompose the val→test AUC drop into dataset shift vs expressibility.

    Reads two run manifests — the current run (``cfg.results_dir/
    run_manifest.json``) and a comparison run (``cfg.ablation_compare_dir/
    run_manifest.json``) — and produces a per-model table with the validation
    AUC, test AUC, the val→test AUC drop and the test balanced accuracy for
    each ``balance`` identifier (``"balanced"`` vs ``"imbalanced"``). Running
    the pipeline twice (imbalanced vs ``class_balance=undersample``, different
    ``results_dir``) and comparing the AUC drops decomposes how much of the
    VQC's 0.969→0.860 drop is dataset shift vs model expressibility.

    The manifest stores per-model validation metrics (``roc_auc``,
    ``balanced_accuracy``, ``threshold``) and the ``config_hash``; the test
    metrics are computed from the saved ``{model}_test_probs.npy`` arrays and
    ``y_test.npy`` (pure analysis/IO — no training). The ``balance``
    identifier is read from the manifest's config fields when stored, else
    inferred from the results directory name containing ``"balanced"``.

    Saves ``results/ablation_shift.csv`` (pandas when available, stdlib
    ``csv`` otherwise).

    Args:
        cfg: Configuration object. Recognised fields: ``results_dir``,
            ``ablation_compare_dir`` (may be absent/empty — the ablation is
            then skipped), ``artifacts_dir``.

    Returns:
        Dict with ``rows`` (list of per-model row dicts), ``current_run`` and
        ``compare_run``, or ``None`` when ``ablation_compare_dir`` is empty or
        either manifest is missing.
    """
    results_dir = Path(getattr(cfg, "results_dir", "results"))
    compare_dir_raw = getattr(cfg, "ablation_compare_dir", "")
    if not compare_dir_raw:
        warnings.warn(
            "ablation_shift: cfg.ablation_compare_dir is empty; skipping the "
            "dataset-shift ablation (set ablation_compare_dir to the second "
            "run's results_dir).",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    compare_dir = Path(compare_dir_raw)

    current_manifest_path = results_dir / "run_manifest.json"
    compare_manifest_path = compare_dir / "run_manifest.json"
    if not current_manifest_path.is_file():
        warnings.warn(
            f"ablation_shift: current run manifest not found: {current_manifest_path}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if not compare_manifest_path.is_file():
        warnings.warn(
            f"ablation_shift: comparison run manifest not found: {compare_manifest_path}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    with open(current_manifest_path, encoding="utf-8") as f:
        current_manifest = json.load(f)
    with open(compare_manifest_path, encoding="utf-8") as f:
        compare_manifest = json.load(f)

    rows: list[dict[str, Any]] = []
    for manifest, run_dir in (
        (current_manifest, results_dir),
        (compare_manifest, compare_dir),
    ):
        balance = _manifest_balance(manifest, run_dir)
        for model, metrics in (manifest.get("metrics") or {}).items():
            rows.append(_ablation_row(model, balance, metrics, manifest, run_dir, cfg))

    columns = ["model", "balance", "val_auc", "test_auc", "auc_drop", "bal_acc_test"]
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "ablation_shift.csv"

    try:
        pd = _import_pandas()
        pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False)
    except RuntimeError:
        warnings.warn(
            "pandas is not installed; writing ablation_shift.csv with the "
            "stdlib csv module.",
            RuntimeWarning,
            stacklevel=2,
        )
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: ("" if v is None else v) for k, v in row.items()})

    return {
        "rows": rows,
        "current_run": str(results_dir),
        "compare_run": str(compare_dir),
    }


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


# ---------------------------------------------------------------------------
# Analysis stage orchestrator
# ---------------------------------------------------------------------------


def run_analysis(cfg: Any, dry_run: bool = False) -> None:
    """Run the full analysis stage: interpretability, DANN validation, ablation.

    Orchestrates the analysis stage of the pipeline: the interpretability
    artifacts (t-SNE domain invariance, calibration curve, SOTA table, 5-fold
    CV) followed by the quantitative DANN validation
    (:func:`run_dann_validation`) and, when ``cfg.ablation_compare_dir`` is
    set, the dataset-shift ablation (:func:`ablation_shift`). Every step is
    wrapped in its own try/except so a failure in one artifact never aborts
    the stage.

    Args:
        cfg: Configuration object. Recognised fields: ``artifacts_dir``,
            ``results_dir``, ``figures_dir``, ``ablation_compare_dir``.
        dry_run: When ``True``, print the planned action and return without
            executing anything.
    """
    if dry_run:
        print("  → analysis: would run interpretability + DANN validation + shift ablation")
        return

    feat_dir = Path(getattr(cfg, "artifacts_dir", "artifacts")) / "features"
    results_dir = Path(getattr(cfg, "results_dir", "results"))
    figures_dir = Path(getattr(cfg, "figures_dir", "figures"))

    # VAE features for the t-SNE / 5-fold CV steps.
    try:
        X_train = np.load(feat_dir / "vae_train.npy")
        y_train = np.load(feat_dir / "y_train.npy")
        X_test = np.load(feat_dir / "vae_test.npy")
        y_test = np.load(feat_dir / "y_test.npy")
    except OSError as exc:
        warnings.warn(
            f"analysis: could not load VAE features from {feat_dir}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        X_train = X_test = y_train = y_test = None

    if X_train is not None:
        try:
            plot_domain_invariance(
                X_train,
                X_test,
                figures_dir / "dann_domain_tsne.png",
                labels_train=y_train,
                labels_test=y_test,
            )
            print("  ✓ analysis: t-SNE saved")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ analysis: t-SNE failed: {exc}")

        try:
            vqc_test_path = results_dir / "vqc_test_probs.npy"
            if vqc_test_path.is_file():
                vqc_test_probs = np.load(vqc_test_path)
                brier_result = plot_calibration_curve(
                    y_test, vqc_test_probs, figures_dir / "calibration_curve.png"
                )
                print(f"  ✓ analysis: calibration Brier={brier_result['brier_score']:.4f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ analysis: calibration curve failed: {exc}")

        try:
            cv_result = run_5fold_cv(cfg, X_train, y_train)
            vqc_auc = cv_result.get("mean", {}).get("vqc", {}).get("roc_auc")
            mlp_auc = cv_result.get("mean", {}).get("mlp", {}).get("roc_auc")
            if vqc_auc is not None and mlp_auc is not None:
                print(f"  ✓ analysis: 5-fold CV — VQC AUC={vqc_auc:.4f}, "
                      f"MLP AUC={mlp_auc:.4f}")
            else:
                print("  ✓ analysis: 5-fold CV done")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ analysis: 5-fold CV failed: {exc}")

    try:
        sota_comparison_table()
        print("  ✓ analysis: SOTA table saved")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ analysis: SOTA table failed: {exc}")

    # Quantitative DANN validation (pre-DANN vs post-DANN discriminator).
    try:
        dann_result = run_dann_validation(cfg)
        if dann_result:
            pre = dann_result.get("pre_dann")
            post = dann_result.get("post_dann")
            if pre is not None and post is not None:
                print(f"  ✓ analysis: DANN discriminator — "
                      f"pre={pre['acc_mean']:.3f}±{pre['acc_std']:.3f}, "
                      f"post={post['acc_mean']:.3f}±{post['acc_std']:.3f}")
            elif post is not None:
                print(f"  ✓ analysis: DANN discriminator — "
                      f"post={post['acc_mean']:.3f}±{post['acc_std']:.3f} "
                      f"(raw features missing)")
            else:
                print("  ✓ analysis: DANN discriminator — sklearn unavailable")
        else:
            print("  ⚠ analysis: DANN validation skipped (no features)")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ analysis: DANN validation failed: {exc}")

    # Dataset-shift vs expressibility ablation (two-run manifest comparison).
    try:
        ablation = ablation_shift(cfg)
        if ablation is not None:
            print(f"  ✓ analysis: shift ablation — {len(ablation['rows'])} rows "
                  f"→ results/ablation_shift.csv")
        else:
            print("  ⚠ analysis: shift ablation skipped (no comparison run)")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ analysis: shift ablation failed: {exc}")
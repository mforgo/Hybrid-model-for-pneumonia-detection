"""Statistical evaluation: threshold selection, metrics, McNemar, bootstrap CI, figures.

Task 11 of the notebook → package rewrite. Consumes the outputs of
``pipeline.vqc.train_vqc`` / ``pipeline.mlp.train_mlp`` (validation/test
probabilities) and produces the results-chapter artifacts:

* ``results/main_results.csv`` — per-model metrics table.
* ``results/{model}_{split}_probs.npy`` — saved probability arrays.
* ``results/run_manifest.json`` — config hash + git sha + timestamp + metrics.
* ``figures/{roc_curve,confusion_matrices,confidence_distribution,training_curves}.png``.

Design decisions (LOCKED by the rewrite plan, T11):

* Threshold selection is **validation-only** — ``find_best_threshold`` scans
  ``tau_range`` and picks the tau maximising balanced accuracy on the set it
  is given. It never sees test labels.
* ``compute_all_metrics`` reports accuracy, balanced accuracy, precision,
  recall, specificity, F1 and ROC-AUC (computed on the raw probabilities,
  not on thresholded predictions).
* ``mcnemar_exact`` runs the exact (binomial) two-sided McNemar test on the
  discordant 2×2 table; ``statsmodels`` is imported lazily and a manual
  ``scipy.stats.binom`` fallback produces the identical p-value.
* ``bootstrap_ci`` resamples with replacement (seeded via
  ``np.random.default_rng(seed)``) and reports 2.5/97.5 percentiles for
  ROC-AUC and accuracy (at tau = 0.5).

Import policy
-------------
Only the standard library and NumPy are imported at module level.
``sklearn``, ``statsmodels``, ``scipy`` and ``matplotlib`` are imported
lazily inside the functions that need them (with a clear ``RuntimeError``
when missing), so ``import pipeline.evaluate`` succeeds even on a machine
without those packages (e.g. the local development box).
"""

from __future__ import annotations

import json
import subprocess
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from pipeline.seed import seed_everything

SEED = 6  # project-wide random seed (AGENTS.md global rule)

__all__ = [
    "SEED",
    "find_best_threshold",
    "compute_all_metrics",
    "mcnemar_exact",
    "bootstrap_ci",
    "plot_roc",
    "plot_confusion_matrices",
    "plot_confidence_distribution",
    "plot_training_curves",
    "save_results",
    "run_manifest",
]


# ---------------------------------------------------------------------------
# Lazy heavy imports (stdlib + numpy only at module level)
# ---------------------------------------------------------------------------


def _import_sklearn_metrics() -> Any:
    """Import ``sklearn.metrics`` lazily, raising a clear RuntimeError."""
    try:
        from sklearn import metrics

        return metrics
    except ImportError as exc:
        raise RuntimeError(
            "pipeline.evaluate requires scikit-learn at call time "
            "(pip install scikit-learn). It is imported lazily so the module "
            "imports with stdlib + numpy only."
        ) from exc


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
            "pipeline.evaluate requires matplotlib at call time "
            "(pip install matplotlib). It is imported lazily so the module "
            "imports with stdlib + numpy only."
        ) from exc


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _balanced_accuracy(labels: np.ndarray, preds: np.ndarray) -> float:
    """Balanced accuracy ``(recall + specificity) / 2`` (matches sklearn).

    Args:
        labels: Ground-truth binary labels (0/1).
        preds: Thresholded predictions (0/1).

    Returns:
        Balanced accuracy as a plain ``float``. Missing classes contribute 0
        recall/specificity (same convention as ``sklearn.metrics`` with
        ``zero_division=0``).
    """
    tp = float(np.sum((preds == 1) & (labels == 1)))
    tn = float(np.sum((preds == 0) & (labels == 0)))
    fp = float(np.sum((preds == 1) & (labels == 0)))
    fn = float(np.sum((preds == 0) & (labels == 1)))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.5 * (recall + specificity)


def _git_sha() -> str:
    """Short git HEAD sha, or ``"unknown"`` when git is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        sha = result.stdout.strip()
        return sha if sha else "unknown"
    except Exception:  # noqa: BLE001 - git may be missing entirely
        return "unknown"


def _write_main_results_csv(path: Path, metrics: dict[str, dict[str, float]]) -> None:
    """Write the per-model metrics table to *path* as CSV.

    Columns follow a canonical order (accuracy, balanced accuracy, precision,
    recall, specificity, F1, ROC-AUC, threshold); any extra metric keys found
    in the model dicts are appended alphabetically. Uses the stdlib ``csv``
    module — no pandas dependency.
    """
    import csv

    canonical = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "specificity",
        "f1",
        "roc_auc",
        "threshold",
    ]
    present = {key for m in metrics.values() for key in m.keys()}
    columns = [c for c in canonical if c in present]
    columns += sorted(present - set(canonical))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + columns)
        for name, m in metrics.items():
            writer.writerow([name] + [m.get(c, "") for c in columns])


# ---------------------------------------------------------------------------
# Threshold selection (VALIDATION ONLY — locked design)
# ---------------------------------------------------------------------------


def find_best_threshold(probs, labels, tau_range) -> float:
    """Pick the tau maximising balanced accuracy on the provided set.

    Scans ``tau_range`` and returns the threshold with the highest balanced
    accuracy on the given ``(probs, labels)`` pair. This function is
    validation-only by design: it must never be called with test labels.

    Tie-break (documented, deterministic): when several thresholds achieve
    the same balanced accuracy, the **largest** tau is returned.

    Args:
        probs: Predicted probabilities, shape ``(N,)``.
        labels: Ground-truth binary labels (0/1), shape ``(N,)``.
        tau_range: Candidate thresholds to scan (array-like of floats).

    Returns:
        The best threshold as a plain ``float``.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    tau_range = np.asarray(tau_range, dtype=np.float64)
    if tau_range.size == 0:
        raise ValueError("tau_range must contain at least one threshold.")

    best_tau = float(tau_range[0])
    best_score = -1.0
    for t in tau_range:
        preds = (probs > t).astype(int)
        score = _balanced_accuracy(labels, preds)
        # ``>=`` implements the largest-tau tie-break deterministically.
        if score >= best_score:
            best_score = score
            best_tau = float(t)
    return best_tau


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_all_metrics(probs, labels, tau) -> dict[str, float]:
    """Compute the full metric suite for one model on one split.

    Confusion-matrix metrics (accuracy, balanced accuracy, precision, recall,
    specificity, F1) are computed with plain NumPy — they match
    ``sklearn.metrics`` exactly (``zero_division=0`` convention). ROC-AUC is
    computed on the **raw probabilities** (not the thresholded predictions)
    via ``sklearn.metrics.roc_auc_score`` (lazy import).

    Args:
        probs: Predicted probabilities, shape ``(N,)``.
        labels: Ground-truth binary labels (0/1), shape ``(N,)``.
        tau: Decision threshold applied to *probs* for the hard predictions.

    Returns:
        Dict with keys ``accuracy``, ``balanced_accuracy``, ``precision``,
        ``recall``, ``f1``, ``roc_auc``, ``specificity``.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    preds = (probs > tau).astype(int)

    tp = float(np.sum((preds == 1) & (labels == 1)))
    tn = float(np.sum((preds == 0) & (labels == 0)))
    fp = float(np.sum((preds == 1) & (labels == 0)))
    fn = float(np.sum((preds == 0) & (labels == 1)))

    accuracy = float(np.mean(preds == labels))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2.0 * tp / (2.0 * tp + fp + fn) if (2.0 * tp + fp + fn) > 0 else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)

    metrics_mod = _import_sklearn_metrics()
    roc_auc = float(metrics_mod.roc_auc_score(labels, probs))

    return {
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "specificity": specificity,
    }


# ---------------------------------------------------------------------------
# McNemar's exact test
# ---------------------------------------------------------------------------


def _mcnemar_exact_binom(b: int, c: int) -> float:
    """Exact two-sided McNemar p-value via the binomial distribution.

    With ``n = b + c`` discordant pairs and ``k = min(b, c)``, the two-sided
    p-value is ``2 * min(P(X <= k), P(X >= k))`` for ``X ~ Binomial(n, 0.5)``
    (identical to ``statsmodels.stats.contingency_tables.mcnemar(exact=True)``).

    Args:
        b: Discordant pairs where A is right and B is wrong.
        c: Discordant pairs where B is right and A is wrong.

    Returns:
        The two-sided p-value in ``[0, 1]``.
    """
    n = int(b) + int(c)
    if n == 0:
        return 1.0
    k = min(int(b), int(c))
    try:
        from scipy.stats import binom
    except ImportError as exc:
        raise RuntimeError(
            "pipeline.evaluate requires statsmodels or scipy for mcnemar_exact "
            "(pip install statsmodels)."
        ) from exc
    p_le = float(binom.cdf(k, n, 0.5))  # P(X <= k)
    p_ge = float(binom.sf(k - 1, n, 0.5))  # P(X >= k) = 1 - P(X <= k - 1)
    return min(2.0 * min(p_le, p_ge), 1.0)


def mcnemar_exact(preds_a, preds_b, labels) -> float:
    """Exact (binomial) two-sided McNemar test between two classifiers.

    Builds the 2×2 discordant table from ``preds_a`` / ``preds_b`` against
    the ground-truth *labels* and runs the exact McNemar test on the
    off-diagonal counts. ``statsmodels`` is imported lazily; when it is
    missing the manual ``scipy.stats.binom`` fallback produces the identical
    p-value (no hard dependency).

    Args:
        preds_a: Hard predictions of classifier A, shape ``(N,)``.
        preds_b: Hard predictions of classifier B, shape ``(N,)``.
        labels: Ground-truth binary labels (0/1), shape ``(N,)``.

    Returns:
        The two-sided p-value as a plain ``float`` in ``[0, 1]``.
    """
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    labels = np.asarray(labels)

    a = float(np.sum((preds_a == labels) & (preds_b == labels)))
    b = float(np.sum((preds_a == labels) & (preds_b != labels)))  # A right, B wrong
    c = float(np.sum((preds_a != labels) & (preds_b == labels)))  # B right, A wrong
    d = float(np.sum((preds_a != labels) & (preds_b != labels)))

    try:
        from statsmodels.stats.contingency_tables import mcnemar
    except ImportError:
        return _mcnemar_exact_binom(int(b), int(c))

    table = np.array([[a, b], [c, d]])
    result = mcnemar(table, exact=True)
    return float(result.pvalue)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_ci(
    probs, labels, n_bootstrap: int = 1000, seed: int = 6
) -> dict[str, tuple[float, float]]:
    """Non-parametric bootstrap 95% CI for ROC-AUC and accuracy.

    Resamples the ``(probs, labels)`` pairs with replacement (seeded via
    ``np.random.default_rng(seed)``) and reports the 2.5/97.5 percentiles of
    the bootstrap distribution. Accuracy is evaluated at ``tau = 0.5``.
    Resamples where ROC-AUC is undefined (a single class drawn) are skipped.

    Args:
        probs: Predicted probabilities, shape ``(N,)``.
        labels: Ground-truth binary labels (0/1), shape ``(N,)``.
        n_bootstrap: Number of bootstrap resamples (default 1000).
        seed: Random seed for the resampling RNG (default 6).

    Returns:
        Dict mapping ``"roc_auc"`` and ``"accuracy"`` to ``(ci_low, ci_high)``
        tuples. When no resample produced a valid AUC the interval is
        ``(nan, nan)``.
    """
    seed_everything(seed)
    metrics_mod = _import_sklearn_metrics()

    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    rng = np.random.default_rng(seed)

    aucs: list[float] = []
    accs: list[float] = []
    for _ in range(int(n_bootstrap)):
        idx = rng.integers(0, n, size=n)
        y_b = labels[idx]
        p_b = probs[idx]
        if len(np.unique(y_b)) < 2:
            continue  # AUC undefined for a single-class resample
        try:
            aucs.append(float(metrics_mod.roc_auc_score(y_b, p_b)))
        except ValueError:
            continue
        preds = (p_b > 0.5).astype(int)
        accs.append(float(np.mean(preds == y_b)))

    if aucs:
        auc_ci = tuple(np.percentile(aucs, [2.5, 97.5]))
    else:
        auc_ci = (float("nan"), float("nan"))
    if accs:
        acc_ci = tuple(np.percentile(accs, [2.5, 97.5]))
    else:
        acc_ci = (float("nan"), float("nan"))

    return {
        "roc_auc": (float(auc_ci[0]), float(auc_ci[1])),
        "accuracy": (float(acc_ci[0]), float(acc_ci[1])),
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_roc(y_true, y_probs, path, label) -> None:
    """Plot and save an ROC curve with the AUC in the legend.

    Args:
        y_true: Ground-truth binary labels (0/1), shape ``(N,)``.
        y_probs: Predicted probabilities, shape ``(N,)``.
        path: Output PNG path (parent directories are created).
        label: Model name shown in the legend.
    """
    plt = _import_matplotlib()
    metrics_mod = _import_sklearn_metrics()

    y_true = np.asarray(y_true, dtype=np.int64)
    y_probs = np.asarray(y_probs, dtype=np.float64)

    fpr, tpr, _ = metrics_mod.roc_curve(y_true, y_probs)
    auc = float(metrics_mod.roc_auc_score(y_true, y_probs))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(cm_vqc, cm_mlp, path) -> None:
    """Plot side-by-side confusion-matrix heatmaps for VQC and MLP.

    Args:
        cm_vqc: 2×2 confusion matrix of the VQC (rows = true, cols = predicted).
        cm_mlp: 2×2 confusion matrix of the MLP.
        path: Output PNG path (parent directories are created).
    """
    plt = _import_matplotlib()

    cm_vqc = np.asarray(cm_vqc)
    cm_mlp = np.asarray(cm_mlp)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, cm, title in zip(axes, (cm_vqc, cm_mlp), ("VQC", "MLP")):
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Pneumonia"])
        ax.set_yticklabels(["Normal", "Pneumonia"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confidence_distribution(y_true, y_probs, path) -> None:
    """Plot a histogram of predicted probabilities split by class.

    Args:
        y_true: Ground-truth binary labels (0/1), shape ``(N,)``.
        y_probs: Predicted probabilities, shape ``(N,)``.
        path: Output PNG path (parent directories are created).
    """
    plt = _import_matplotlib()

    y_true = np.asarray(y_true, dtype=np.int64)
    y_probs = np.asarray(y_probs, dtype=np.float64)

    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(0.0, 1.0, 21)
    ax.hist(
        y_probs[y_true == 0],
        bins=bins,
        alpha=0.6,
        label="Normal (0)",
        color="tab:blue",
    )
    ax.hist(
        y_probs[y_true == 1],
        bins=bins,
        alpha=0.6,
        label="Pneumonia (1)",
        color="tab:red",
    )
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("Predicted probability distribution by class")
    ax.legend()
    ax.grid(alpha=0.3)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history_dict, path) -> None:
    """Plot train/val loss curves for one or more models.

    Args:
        history_dict: Maps model name → history dict with ``train_loss`` and
            ``val_loss`` lists (per-epoch).
        path: Output PNG path (parent directories are created).
    """
    plt = _import_matplotlib()

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, hist in history_dict.items():
        train_loss = hist.get("train_loss", [])
        val_loss = hist.get("val_loss", [])
        if train_loss:
            ax.plot(
                range(1, len(train_loss) + 1),
                train_loss,
                label=f"{name} train",
                marker="o",
                markersize=3,
            )
        if val_loss:
            ax.plot(
                range(1, len(val_loss) + 1),
                val_loss,
                label=f"{name} val",
                marker="s",
                markersize=3,
            )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(alpha=0.3)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Run manifest + result persistence
# ---------------------------------------------------------------------------


def run_manifest(cfg: Any, metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    """Build and save the run manifest JSON.

    The manifest records the config hash (``pipeline.config.config_hash``),
    the short git HEAD sha (``"unknown"`` when git is unavailable), an
    ISO8601 timestamp, all metrics and the artifact paths.

    Args:
        cfg: Configuration object. Recognised fields: ``results_dir``.
        metrics: Per-model metrics dict (model name → metric dict).

    Returns:
        The manifest dict with keys ``config_hash``, ``git_sha``,
        ``timestamp``, ``metrics`` and ``artifacts``.
    """
    from pipeline.config import config_hash  # lazy: config.py needs yaml

    manifest = {
        "config_hash": config_hash(cfg),
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "artifacts": {},
    }

    results_dir = Path(getattr(cfg, "results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def save_results(
    cfg: Any,
    metrics: dict[str, dict[str, float]],
    probs_dict: dict[str, Any],
    params_dict: dict[str, Any],
    history_dict: dict[str, dict[str, list[float]]],
) -> dict[str, Any]:
    """Persist all evaluation artifacts and the run manifest.

    Writes under ``cfg.results_dir`` (relative to the repo root — never
    ``/content/*``):

    * ``main_results.csv`` — per-model metrics table.
    * ``{model}_{split}_probs.npy`` — probability arrays from *probs_dict*.
      Each entry maps a model name to either a flat array (saved as
      ``{model}_probs.npy``) or a ``{split: array}`` dict (saved as
      ``{model}_{split}_probs.npy``).
    * ``{model}_best_params.npy`` / ``{model}_best.pt`` / ``{model}_params.json``
      — parameters from *params_dict* (numpy arrays, torch state dicts and
      JSON-serialisable values respectively).
    * ``{model}_history.json`` — per-epoch training histories.
    * ``run_manifest.json`` — config hash + git sha + timestamp + metrics +
      artifact paths.

    When *history_dict* is non-empty the training-curves figure is also
    written to ``cfg.figures_dir/training_curves.png`` (skipped with a
    warning if matplotlib is unavailable).

    Args:
        cfg: Configuration object. Recognised fields: ``results_dir``,
            ``figures_dir``.
        metrics: Per-model metrics dict (model name → metric dict).
        probs_dict: Model name → probability array or ``{split: array}`` dict.
        params_dict: Model name → parameters (ndarray, torch state dict, or
            JSON-serialisable value).
        history_dict: Model name → history dict with ``train_loss`` /
            ``val_loss`` lists.

    Returns:
        The final manifest dict (with populated ``artifacts``).
    """
    results_dir = Path(getattr(cfg, "results_dir", "results"))
    figures_dir = Path(getattr(cfg, "figures_dir", "figures"))
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, str] = {}

    # --- main_results.csv ---
    csv_path = results_dir / "main_results.csv"
    _write_main_results_csv(csv_path, metrics)
    artifacts["main_results.csv"] = str(csv_path)

    # --- probability arrays ---
    for name, probs in (probs_dict or {}).items():
        if isinstance(probs, dict):
            for split, arr in probs.items():
                p = results_dir / f"{name}_{split}_probs.npy"
                np.save(p, np.asarray(arr))
                artifacts[f"{name}_{split}_probs.npy"] = str(p)
        else:
            p = results_dir / f"{name}_probs.npy"
            np.save(p, np.asarray(probs))
            artifacts[f"{name}_probs.npy"] = str(p)

    # --- parameters ---
    for name, params in (params_dict or {}).items():
        if isinstance(params, np.ndarray):
            p = results_dir / f"{name}_best_params.npy"
            np.save(p, params)
            artifacts[f"{name}_best_params.npy"] = str(p)
        elif isinstance(params, dict):
            try:
                import torch  # lazy: torch is optional at import time

                p = results_dir / f"{name}_best.pt"
                torch.save(params, p)
                artifacts[f"{name}_best.pt"] = str(p)
            except ImportError:
                p = results_dir / f"{name}_params.json"
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(params, f, indent=2, default=str)
                artifacts[f"{name}_params.json"] = str(p)
        else:
            p = results_dir / f"{name}_params.json"
            with open(p, "w", encoding="utf-8") as f:
                json.dump(params, f, indent=2, default=str)
            artifacts[f"{name}_params.json"] = str(p)

    # --- training histories ---
    for name, hist in (history_dict or {}).items():
        p = results_dir / f"{name}_history.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2)
        artifacts[f"{name}_history.json"] = str(p)

    # --- training-curves figure (best effort) ---
    if history_dict:
        try:
            fig_path = figures_dir / "training_curves.png"
            plot_training_curves(history_dict, fig_path)
            artifacts["training_curves.png"] = str(fig_path)
        except RuntimeError as exc:
            warnings.warn(
                f"Could not plot training curves: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    # --- manifest (run_manifest saves the base dict; we enrich artifacts) ---
    manifest = run_manifest(cfg, metrics)
    manifest["artifacts"] = artifacts
    with open(results_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
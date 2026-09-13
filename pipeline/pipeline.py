"""CLI orchestrator dispatching pipeline stages.

Entry point for the hybrid QML pneumonia detection pipeline::

    python -m pipeline --stage {stage} [--config path] [--overrides key=value ...] [--dry-run]

Stages (execution order for ``--stage all``):

    data → features → vae/svae → ansatz → train → evaluate → qpu → analysis

Override prefix → flat field mapping (``--overrides``)::

    data.subset        → subset
    vae.epochs         → vae_epochs        vae.batch_size  → vae_batch_size
    vae.lr             → vae_lr            vae.beta        → vae_beta
    vqc.epochs         → epochs            vqc.batch_size  → batch_size
    vqc.n_layers       → n_layers
    mlp.epochs         → epochs            mlp.hidden      → mlp_hidden
    mlp.dropout        → mlp_dropout
    features.dann_epochs → dann_epochs      features.use_dann → use_dann

Import policy
-------------
Only the standard library and ``pipeline.config`` / ``pipeline.seed`` are
imported at module level — no numpy, torch, pennylane, sklearn, or matplotlib.
Every stage module is imported lazily inside its dispatch function so
``--dry-run`` never touches heavy libraries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.config import (
    Config,
    apply_cli_overrides,
    config_hash,
    config_to_dict,
    load_config,
    parse_cli_overrides,
)
from pipeline.seed import SEED, seed_everything

__all__ = [
    "STAGES",
    "build_cache_key",
    "select_device",
    "main",
]

#: Ordered tuple of all recognised stage names (``"all"`` included).
STAGES: tuple[str, ...] = (
    "data",
    "features",
    "vae",
    "svae",
    "ansatz",
    "train",
    "evaluate",
    "qpu",
    "analysis",
    "all",
)

#: Per-stage expected artifact filenames (relative to their output directories).
_STAGE_ARTIFACTS: dict[str, list[str]] = {
    "data": ["artifacts/dataset_meta.json"],
    "features": [
        "artifacts/features/convnext_tiny_train.npy",
        "artifacts/features/convnext_tiny_val.npy",
        "artifacts/features/convnext_tiny_test.npy",
        "artifacts/features/y_train.npy",
        "artifacts/features/y_val.npy",
        "artifacts/features/y_test.npy",
        "artifacts/features/feature_meta.json",
    ],
    "vae": [
        "artifacts/features/vae_train.npy",
        "artifacts/features/vae_val.npy",
        "artifacts/features/vae_test.npy",
        "artifacts/features/vae_meta.json",
    ],
    "ansatz": ["results/ansatz_meta.json"],
    "train": [
        "results/vqc_best_params.npy",
        "results/vqc_history.json",
        "results/vqc_val_probs.npy",
        "results/mlp_best.pt",
        "results/mlp_history.json",
        "results/mlp_val_probs.npy",
    ],
    "evaluate": [
        "results/main_results.csv",
        "results/run_manifest.json",
    ],
    "qpu": [
        "results/vqc_qpu_probs.npy",
        "results/vqc_fakekingston_probs.npy",
    ],
    "analysis": ["results/cv_results.json"],
}

#: Meta file per stage that stores ``config_hash`` for cache validation.
_STAGE_META: dict[str, str] = {
    "data": "artifacts/dataset_meta.json",       # NOTE: no config_hash inside
    "features": "artifacts/features/feature_meta.json",
    "vae": "artifacts/features/vae_meta.json",
    "svae": "artifacts/features/vae_meta.json",  # same file, different hash
    "ansatz": "results/ansatz_meta.json",
    "train": "results/run_manifest.json",
    "evaluate": "results/run_manifest.json",
    "qpu": "results/run_manifest.json",
    "analysis": "results/cv_results.json",
}


# ---------------------------------------------------------------------------
# Cache key builder  (test contract: tests/test_pipeline.py)
# ---------------------------------------------------------------------------


def _feature_hash(cfg: Config) -> str:
    """md5 hex digest of the first 100 rows of the training feature matrix.

    Returns an empty string when the file is unavailable.
    """
    feat_path = (
        Path(str(cfg.artifacts_dir or "artifacts")) / "features" / "convnext_tiny_train.npy"
    )
    if not feat_path.is_file():
        return ""
    try:
        import numpy as _np  # noqa: F811 — local import, not module-level

        arr = _np.load(str(feat_path))[:100]
        return hashlib.md5(_np.ascontiguousarray(arr).tobytes()).hexdigest()
    except Exception:  # noqa: BLE001 — file may be corrupt
        return ""


def _git_sha() -> str:
    """Short ``git rev-parse --short HEAD``, or ``"unknown"`` on failure."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        sha = result.stdout.strip()
        return sha if sha else "unknown"
    except Exception:  # noqa: BLE001 — git may be missing
        return "unknown"


def build_cache_key(cfg: Config) -> dict[str, str]:
    """Deterministic cache key for pipeline artifacts.

    Returns a dict with exactly four keys:

    ``feat_hash``
        md5 of the first 100 rows of the training feature matrix (``""``
        when the feature file does not yet exist).
    ``config_hash``
        sha-256 hex digest of the serialised ``Config`` — **must** equal
        ``config_hash(cfg)`` from ``pipeline.config`` (acceptance contract).
    ``git_sha``
        Short git HEAD sha via read-only ``git rev-parse``, or ``"unknown"``.
    ``timestamp``
        ISO-8601 UTC timestamp of when the key was built.

    Args:
        cfg: The current pipeline configuration.

    Returns:
        Dict with the four cache-key entries.
    """
    return {
        "feat_hash": _feature_hash(cfg),
        "config_hash": config_hash(cfg),
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# GPU / device selection
# ---------------------------------------------------------------------------


def select_device(cfg: Config) -> str:
    """Resolve the compute device and configure GPU environment.

    When ``cfg.device == "auto"``, the least-loaded CUDA GPU is selected by
    inspecting ``torch.cuda.mem_get_info()`` across available devices; the
    ``CUDA_VISIBLE_DEVICES`` environment variable is set accordingly (but
    ``cfg.device`` is **not** mutated — it stays ``"auto"`` so that the
    config hash remains stable across machines).

    When ``cfg.gpu_strategy == "manual"`` and ``cfg.cuda_device`` is set, that
    device is used directly.

    If ``torch`` is not installed, ``"cpu"`` is returned with a log line.

    Args:
        cfg: The pipeline configuration.

    Returns:
        The resolved device string (``"cuda"``, ``"cuda:N"``, or ``"cpu"``).
    """
    explicit = str(cfg.device or "auto").strip()
    if explicit != "auto":
        return explicit

    # Manual strategy: user pre-selected a specific CUDA device.
    if cfg.gpu_strategy == "manual" and cfg.cuda_device:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_device
        return cfg.cuda_device

    # Auto-select: pick the least-loaded GPU.
    try:
        import torch
    except ImportError:
        print("[device] torch not installed; using CPU")
        return "cpu"

    if not torch.cuda.is_available():
        print("[device] CUDA not available; using CPU")
        return "cpu"

    n_gpus = torch.cuda.device_count()
    if n_gpus == 0:
        print("[device] No CUDA devices found; using CPU")
        return "cpu"

    if n_gpus == 1:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print("[device] Single GPU detected: cuda:0")
        return "cuda"

    # Multiple GPUs: pick the one with the most free memory.
    best_idx, best_free = 0, 0
    for idx in range(n_gpus):
        free, _ = torch.cuda.mem_get_info(idx)
        if free > best_free:
            best_free = free
            best_idx = idx

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
    gpu_name = torch.cuda.get_device_name(best_idx)
    free_gb = best_free / (1024 ** 3)
    print(
        f"[device] Selected GPU {best_idx} ({gpu_name}, "
        f"{free_gb:.1f} GB free)"
    )
    return "cuda"


# ---------------------------------------------------------------------------
# Stage cache status
# ---------------------------------------------------------------------------


def _stage_cache_status(
    stage: str, cfg: Config, key: dict[str, str]
) -> tuple[bool, str]:
    """Check whether a stage can be skipped (all artifacts present + fresh).

    Args:
        stage: Stage name (must be a key in ``_STAGE_META``).
        cfg: Current configuration.
        key: Cache key from :func:`build_cache_key`.

    Returns:
        ``(cache_hit, reason)`` — ``True`` when every artifact exists and the
        meta file's ``config_hash`` matches *key*, else ``False`` with a
        human-readable reason string.
    """
    artifacts = _STAGE_ARTIFACTS.get(stage, [])
    meta_rel = _STAGE_META.get(stage)
    meta_path = Path(meta_rel) if meta_rel else None
    cfg_hash = key["config_hash"]

    # Check artifact files.
    missing = [a for a in artifacts if not Path(a).is_file()]
    if missing:
        return False, f"missing: {', '.join(missing)}"

    # Check meta file's config_hash.
    if meta_path is not None and meta_path.is_file():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            meta_hash = meta.get("config_hash", "")
            if meta_hash != cfg_hash:
                return False, (
                    f"stale: meta config_hash={meta_hash[:12]}… "
                    f"!= current {cfg_hash[:12]}…"
                )
            return True, "cache hit"
        except (json.JSONDecodeError, OSError):
            return False, "meta file unreadable"
    elif meta_path is not None:
        return False, f"meta file missing: {meta_path}"
    else:
        return False, "no meta defined for stage"


# ---------------------------------------------------------------------------
# Feature / VAE loading helpers
# ---------------------------------------------------------------------------


def _load_features_from_disk(cfg: Config) -> dict[str, Any]:
    """Load the six 768-dim feature/label arrays from ``artifacts/features/``.

    Returns:
        Dict with keys ``X_train``, ``y_train``, ``X_val``, ``y_val``,
        ``X_test``, ``y_test`` — each a numpy array.

    Raises:
        FileNotFoundError: If any required ``.npy`` file is missing.
    """
    import numpy as np

    feat_dir = Path(cfg.artifacts_dir) / "features"
    splits = ("train", "val", "test")
    result: dict[str, Any] = {}
    for split in splits:
        result[f"X_{split}"] = np.load(feat_dir / f"convnext_tiny_{split}.npy")
        result[f"y_{split}"] = np.load(feat_dir / f"y_{split}.npy")
    return result


def _load_vae_features(cfg: Config) -> dict[str, Any]:
    """Load the six 64-dim VAE feature/label arrays from ``artifacts/features/``.

    Returns:
        Dict with keys ``X_train``, ``y_train``, ``X_val``, ``y_val``,
        ``X_test``, ``y_test`` — each a numpy array.

    Raises:
        FileNotFoundError: If any required ``.npy`` file is missing.
    """
    import numpy as np

    feat_dir = Path(cfg.artifacts_dir) / "features"
    splits = ("train", "val", "test")
    result: dict[str, Any] = {}
    for split in splits:
        result[f"X_{split}"] = np.load(feat_dir / f"vae_{split}.npy")
        result[f"y_{split}"] = np.load(feat_dir / f"y_{split}.npy")
    return result


# ---------------------------------------------------------------------------
# Per-stage dispatch functions
# ---------------------------------------------------------------------------


def _run_data(cfg: Config, dry_run: bool = False) -> None:
    """Dispatch the ``data`` stage: download dataset and build loaders."""
    if dry_run:
        print(f"  → data: would download dataset and build loaders → {_STAGE_ARTIFACTS['data']}")
        return
    from pipeline.data import get_loaders
    loaders = get_loaders(cfg)
    print(f"  ✓ data: {loaders['meta'].train_count} train, "
          f"{loaders['meta'].val_count} val, {loaders['meta'].test_count} test")


def _run_features(cfg: Config, dry_run: bool = False) -> None:
    """Dispatch the ``features`` stage: extract or load cached 768-dim features."""
    if dry_run:
        print(f"  → features: would extract ConvNeXt-Tiny features → {_STAGE_ARTIFACTS['features']}")
        return
    from pipeline.data import get_loaders
    from pipeline.features import extract_or_load

    loaders = get_loaders(cfg)
    feat = extract_or_load(cfg, loaders)
    print(f"  ✓ features: X_train={feat['X_train'].shape}, "
          f"X_val={feat['X_val'].shape}, X_test={feat['X_test'].shape}")


def _run_vae(cfg: Config, dry_run: bool = False, svae: bool = False) -> None:
    """Dispatch the ``vae`` or ``svae`` stage: dimensionality reduction 768→64."""
    stage = "svae" if svae else "vae"
    if dry_run:
        print(f"  → {stage}: would fit {'SVAE' if svae else 'VAE'} reduction "
              f"768→64 → {_STAGE_ARTIFACTS['vae']}")
        return
    from pipeline.vae import fit_vae_pipeline

    feat = _load_features_from_disk(cfg)
    save_dir = Path(cfg.artifacts_dir) / "features"
    result = fit_vae_pipeline(
        X_train=feat["X_train"],
        X_val=feat["X_val"],
        X_test=feat["X_test"],
        cfg=cfg,
        save_dir=save_dir,
        prefix="",
        y_train=feat["y_train"],
        y_val=feat["y_val"],
        y_test=feat["y_test"],
    )
    print(f"  ✓ {stage}: latent_train={result['X_train_latent'].shape}, "
          f"latent_val={result['X_val_latent'].shape}")


def _run_ansatz(cfg: Config, dry_run: bool = False) -> None:
    """Dispatch the ``ansatz`` stage: expressibility + entanglement analysis."""
    if dry_run:
        print(f"  → ansatz: would compute expressibility/entanglement → {_STAGE_ARTIFACTS['ansatz']}")
        return
    from pipeline.ansatz import (
        count_params,
        entanglement_capability,
        expressibility_sweep,
    )

    n_params = count_params(cfg.n_qubits, cfg.n_layers,
                            cfg.use_learnable_scale, cfg.use_measurement_basis)
    print(f"  ansatz: {n_params} trainable params "
          f"({cfg.n_layers} layers × {cfg.n_qubits} qubits)")

    try:
        results = expressibility_sweep(
            n_qubits=cfg.n_qubits,
            layers=(cfg.n_layers,),
            n_samples=100,
            n_bins=50,
            seed=cfg.seed,
        )
        for r in results:
            print(f"  ansatz: L={r['n_layers']}, Expr(A)={r['Expr(A)']:.4f}, "
                  f"Ent(A)={r['Ent(A)']:.4f}")
    except Exception as exc:  # noqa: BLE001 — expressibility is best-effort
        print(f"  ⚠ ansatz: expressibility computation failed: {exc}")

    # Save ansatz metadata for cache.
    meta = {
        "config_hash": config_hash(cfg),
        "git_sha": _git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_qubits": cfg.n_qubits,
        "n_layers": cfg.n_layers,
        "n_params": n_params,
        "use_learnable_scale": cfg.use_learnable_scale,
        "use_measurement_basis": cfg.use_measurement_basis,
    }
    meta_path = Path("results") / "ansatz_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  ✓ ansatz: saved {meta_path}")


def _run_train(cfg: Config, dry_run: bool = False) -> None:
    """Dispatch the ``train`` stage: train VQC + MLP on 64-dim VAE features."""
    if dry_run:
        print(f"  → train: would train VQC+MLP → {_STAGE_ARTIFACTS['train']}")
        return
    from pipeline.mlp import train_mlp
    from pipeline.vqc import train_vqc

    vae = _load_vae_features(cfg)
    X_tr, y_tr = vae["X_train"], vae["y_train"]
    X_va, y_va = vae["X_val"], vae["y_val"]
    X_te, y_te = vae["X_test"], vae["y_test"]

    print("  ── VQC training ──")
    vqc_result = train_vqc(cfg, X_tr, y_tr, X_va, y_va, X_te, y_te)

    print("  ── MLP training ──")
    mlp_result = train_mlp(cfg, X_tr, y_tr, X_va, y_va, X_te, y_te)

    print(f"  ✓ train: VQC best_epoch={vqc_result['epochs_trained']}, "
          f"MLP best_epoch={mlp_result['epochs_trained']}")


def _run_evaluate(cfg: Config, dry_run: bool = False) -> None:
    """Dispatch the ``evaluate`` stage: metrics, threshold, figures, manifest."""
    if dry_run:
        print(f"  → evaluate: would compute metrics + figures → {_STAGE_ARTIFACTS['evaluate']}")
        return
    import numpy as np
    from pipeline.evaluate import (
        compute_all_metrics,
        find_best_threshold,
        plot_confidence_distribution,
        plot_confusion_matrices,
        plot_roc,
        plot_training_curves,
        save_results,
    )

    # Load ground-truth labels.
    feat = _load_features_from_disk(cfg)
    y_val = feat["y_val"]
    y_test = feat["y_test"]

    # Load model probabilities and histories.
    results_dir = Path(cfg.results_dir)
    vqc_val_probs = np.load(results_dir / "vqc_val_probs.npy")
    mlp_val_probs = np.load(results_dir / "mlp_val_probs.npy")

    with open(results_dir / "vqc_history.json", encoding="utf-8") as f:
        vqc_history = json.load(f)
    with open(results_dir / "mlp_history.json", encoding="utf-8") as f:
        mlp_history = json.load(f)

    # Threshold selection on validation set only.
    tau_range = np.arange(
        cfg.threshold_range_min,
        cfg.threshold_range_max + cfg.threshold_step / 2.0,
        cfg.threshold_step,
    )
    vqc_tau = find_best_threshold(vqc_val_probs, y_val, tau_range)
    mlp_tau = find_best_threshold(mlp_val_probs, y_val, tau_range)
    print(f"  evaluate: VQC tau={vqc_tau:.3f}, MLP tau={mlp_tau:.3f}")

    # Compute full metrics at the selected thresholds.
    vqc_metrics = compute_all_metrics(vqc_val_probs, y_val, vqc_tau)
    mlp_metrics = compute_all_metrics(mlp_val_probs, y_val, mlp_tau)

    # Load test probs if available (train stage writes them).
    vqc_test_path = results_dir / "vqc_test_probs.npy"
    mlp_test_path = results_dir / "mlp_test_probs.npy"
    vqc_test_probs = np.load(vqc_test_path) if vqc_test_path.is_file() else vqc_val_probs
    mlp_test_probs = np.load(mlp_test_path) if mlp_test_path.is_file() else mlp_val_probs

    # Assemble probs_dict and params_dict for save_results.
    probs_dict = {
        "vqc": {"val": vqc_val_probs, "test": vqc_test_probs},
        "mlp": {"val": mlp_val_probs, "test": mlp_test_probs},
    }

    params_dict: dict[str, Any] = {}
    vqc_params_path = results_dir / "vqc_best_params.npy"
    if vqc_params_path.is_file():
        params_dict["vqc"] = np.load(vqc_params_path)
    mlp_params_path = results_dir / "mlp_best.pt"
    if mlp_params_path.is_file():
        try:
            import torch
            params_dict["mlp"] = torch.load(mlp_params_path, map_location="cpu")
        except ImportError:
            pass  # save_results handles non-torch fallback

    history_dict = {"vqc": vqc_history, "mlp": mlp_history}

    metrics = {"vqc": vqc_metrics, "mlp": mlp_metrics}
    save_results(cfg, metrics, probs_dict, params_dict, history_dict)

    # Figures.
    figures_dir = Path(cfg.figures_dir)
    try:
        plot_roc(y_test, vqc_test_probs, figures_dir / "roc_vqc.png", "VQC")
        plot_roc(y_test, mlp_test_probs, figures_dir / "roc_mlp.png", "MLP")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ evaluate: ROC plot failed: {exc}")

    try:
        cm_vqc = np.array([
            [np.sum((vqc_test_probs <= vqc_tau) & (y_test == 0)),
             np.sum((vqc_test_probs > vqc_tau) & (y_test == 0))],
            [np.sum((vqc_test_probs <= vqc_tau) & (y_test == 1)),
             np.sum((vqc_test_probs > vqc_tau) & (y_test == 1))],
        ])
        cm_mlp = np.array([
            [np.sum((mlp_test_probs <= mlp_tau) & (y_test == 0)),
             np.sum((mlp_test_probs > mlp_tau) & (y_test == 0))],
            [np.sum((mlp_test_probs <= mlp_tau) & (y_test == 1)),
             np.sum((mlp_test_probs > mlp_tau) & (y_test == 1))],
        ])
        plot_confusion_matrices(cm_vqc, cm_mlp, figures_dir / "confusion_matrices.png")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ evaluate: confusion matrix plot failed: {exc}")

    try:
        plot_confidence_distribution(y_test, vqc_test_probs,
                                     figures_dir / "confidence_distribution.png")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ evaluate: confidence distribution plot failed: {exc}")

    try:
        plot_training_curves(history_dict, figures_dir / "training_curves.png")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ evaluate: training curves plot failed: {exc}")

    print(f"  ✓ evaluate: metrics saved to {results_dir}/")


def _run_qpu(cfg: Config, dry_run: bool = False) -> None:
    """Dispatch the ``qpu`` stage: hardware evaluation (best-effort)."""
    if dry_run:
        print(f"  → qpu: would run hardware eval → {_STAGE_ARTIFACTS['qpu']}")
        return
    import numpy as np
    from pipeline.qpu import qpu_evaluate

    vae = _load_vae_features(cfg)
    X_test, y_test = vae["X_test"], vae["y_test"]

    results_dir = Path(cfg.results_dir)
    vqc_params_path = results_dir / "vqc_best_params.npy"
    if not vqc_params_path.is_file():
        print("  ⚠ qpu: no vqc_best_params.npy — skipping (train stage not run?)")
        return

    params = np.load(vqc_params_path)

    # Build the QNode for inference.
    try:
        from pipeline.ansatz import build_vqc_circuit
        circuit_qnode = build_vqc_circuit(
            cfg.n_qubits, cfg.n_layers,
            use_scale=cfg.use_learnable_scale,
            use_meas_basis=cfg.use_measurement_basis,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ qpu: failed to build circuit: {exc}")
        return

    result = qpu_evaluate(cfg, circuit_qnode, params, X_test, y_test)
    if result.get("warnings"):
        for w in result["warnings"]:
            print(f"  ⚠ qpu: {w}")
    print(f"  ✓ qpu: backend={result.get('backend_name', 'N/A')}")


def _run_analysis(cfg: Config, dry_run: bool = False) -> None:
    """Dispatch the ``analysis`` stage: t-SNE, calibration, SOTA, 5-fold CV."""
    if dry_run:
        print(f"  → analysis: would run interpretability + CV → {_STAGE_ARTIFACTS['analysis']}")
        return
    import numpy as np
    from pipeline.analysis import (
        plot_domain_invariance,
        plot_calibration_curve,
        run_5fold_cv,
        sota_comparison_table,
    )

    vae = _load_vae_features(cfg)
    X_train, y_train = vae["X_train"], vae["y_train"]
    X_test, y_test = vae["X_test"], vae["y_test"]

    # t-SNE domain invariance.
    try:
        from pipeline.config import config_hash  # noqa: F811
        figures_dir = Path(cfg.figures_dir)
        plot_domain_invariance(
            X_train, X_test,
            figures_dir / "dann_domain_tsne.png",
            labels_train=y_train, labels_test=y_test,
        )
        print("  ✓ analysis: t-SNE saved")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ analysis: t-SNE failed: {exc}")

    # Calibration curve (using VQC test probs).
    results_dir = Path(cfg.results_dir)
    vqc_test_path = results_dir / "vqc_test_probs.npy"
    if vqc_test_path.is_file():
        try:
            vqc_test_probs = np.load(vqc_test_path)
            figures_dir = Path(cfg.figures_dir)
            brier_result = plot_calibration_curve(
                y_test, vqc_test_probs, figures_dir / "calibration_curve.png"
            )
            print(f"  ✓ analysis: calibration Brier={brier_result['brier_score']:.4f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  ⚠ analysis: calibration curve failed: {exc}")

    # SOTA comparison table.
    try:
        sota_comparison_table()
        print("  ✓ analysis: SOTA table saved")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ analysis: SOTA table failed: {exc}")

    # 5-fold CV (train+val features concatenated).
    try:
        cv_result = run_5fold_cv(cfg, X_train, y_train)
        vqc_mean = cv_result.get("mean", {}).get("vqc", {})
        mlp_mean = cv_result.get("mean", {}).get("mlp", {})
        print(f"  ✓ analysis: 5-fold CV — VQC AUC={vqc_mean.get('roc_auc', 'N/A'):.4f}, "
              f"MLP AUC={mlp_mean.get('roc_auc', 'N/A'):.4f}")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ analysis: 5-fold CV failed: {exc}")


# ---------------------------------------------------------------------------
# Unified dispatch
# ---------------------------------------------------------------------------

_STAGE_DISPATCH: dict[str, Any] = {
    "data": _run_data,
    "features": _run_features,
    "vae": _run_vae,
    "svae": lambda cfg, dry_run=False: _run_vae(cfg, dry_run, svae=True),
    "ansatz": _run_ansatz,
    "train": _run_train,
    "evaluate": _run_evaluate,
    "qpu": _run_qpu,
    "analysis": _run_analysis,
}


def _dispatch_stage(stage: str, cfg: Config, key: dict[str, str],
                    dry_run: bool = False) -> None:
    """Check cache and dispatch a single stage.

    If dry-run mode is active, print the planned action and return without
    executing the stage. Otherwise check cache status and skip the stage
    (with a log line) when a valid cache hit is detected.

    Args:
        stage: One of the recognised stage names (not ``"all"``).
        cfg: Current configuration.
        key: Cache key from :func:`build_cache_key`.
        dry_run: When ``True``, print only and do not execute.

    Raises:
        RuntimeError: When a stage module's heavy dependencies are missing.
    """
    hit, reason = _stage_cache_status(stage, cfg, key)
    if dry_run:
        dispatch_fn = _STAGE_DISPATCH.get(stage)
        if dispatch_fn is not None:
            dispatch_fn(cfg, dry_run=True)
        return

    if hit:
        print(f"  ↩ {stage}: {reason} — skipping")
        return

    dispatch_fn = _STAGE_DISPATCH.get(stage)
    if dispatch_fn is None:
        print(f"  ⚠ {stage}: unknown stage — skipping")
        return

    try:
        dispatch_fn(cfg, dry_run=False)
    except ImportError as exc:
        raise RuntimeError(
            f"Stage {stage!r} requires missing dependencies: {exc}. "
            f"Install the project dependencies (see requirements.txt)."
        ) from exc


def _run_all(cfg: Config, key: dict[str, str], dry_run: bool = False) -> None:
    """Execute all stages sequentially with per-stage error handling.

    ``qpu`` failures are tolerated (logged and skipped) because hardware
    access is best-effort. All other stage failures are re-raised.

    Args:
        cfg: Current configuration.
        key: Cache key from :func:`build_cache_key`.
        dry_run: When ``True``, print only and do not execute.
    """
    all_stages = [s for s in STAGES if s != "all"]

    for stage in all_stages:
        try:
            _dispatch_stage(stage, cfg, key, dry_run=dry_run)
        except Exception:
            if stage == "qpu":
                print(f"  ⚠ qpu: failed (non-fatal):")
                traceback.print_exc()
                continue
            raise


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Parse CLI arguments, resolve config, and dispatch pipeline stages.

    Args:
        argv: Command-line arguments (``None`` → ``sys.argv[1:]``).

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    parser = argparse.ArgumentParser(
        prog="python -m pipeline",
        description=(
            "Hybrid QML Pneumonia Detection Pipeline — stage orchestrator "
            "with config-hash-based caching."
        ),
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=list(STAGES),
        help="Pipeline stage to execute (or 'all' for the full pipeline).",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML configuration file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--overrides",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "CLI overrides in 'key=value' format. Supports dotted prefixes: "
            "data.subset, vae.epochs, vae.batch_size, vae.lr, vae.beta, "
            "vqc.epochs, vqc.batch_size, vqc.n_layers, mlp.epochs, "
            "mlp.hidden, mlp.dropout, features.dann_epochs, features.use_dann."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print planned stage actions without executing or importing heavy libs.",
    )
    args = parser.parse_args(argv)

    # ── Resolve configuration ────────────────────────────────────────────
    cfg = load_config(args.config)
    if args.overrides:
        overrides = parse_cli_overrides(args.overrides)
        cfg = apply_cli_overrides(cfg, overrides)

    # ── Seed ─────────────────────────────────────────────────────────────
    seed_everything(cfg.seed)

    # ── Print header ─────────────────────────────────────────────────────
    key = build_cache_key(cfg)
    print("=" * 70)
    print(f"  Hybrid QML Pneumonia Detection Pipeline")
    print(f"  Stage:   {args.stage}")
    print(f"  Config:  {args.config}")
    print(f"  Seed:    {cfg.seed}")
    print(f"  Hash:    {key['config_hash'][:16]}…")
    print(f"  Git:     {key['git_sha']}")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] No stages will be executed.\n")

    # ── Device selection (skipped in dry-run to avoid torch import) ──────
    if not args.dry_run:
        device = select_device(cfg)
        print(f"  Device:  {device}")
        print()

    # ── Dispatch ─────────────────────────────────────────────────────────
    try:
        if args.stage == "all":
            _run_all(cfg, key, dry_run=args.dry_run)
        else:
            _dispatch_stage(args.stage, cfg, key, dry_run=args.dry_run)
    except Exception:
        traceback.print_exc()
        return 1

    print("\n" + "=" * 70)
    print("  Pipeline finished successfully.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())

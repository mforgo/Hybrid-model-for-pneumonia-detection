"""CLI orchestrator dispatching pipeline stages.

Entry point for the hybrid QML pneumonia detection pipeline::

    python -m pipeline --stage {stage} [--config path] [--overrides key=value ...] [--dry-run]

Stages (execution order for ``--stage all``):

    data → features → vae/svae → ansatz → train → evaluate → qpu → analysis

``benchmark`` is a standalone stage (excluded from ``all``): it trains every
model in ``cfg.models`` via the model registry and writes the comparison
CSVs (``results/benchmark_comparison.csv``, ``results/benchmark_mcnemar.csv``)
plus the ``figures/benchmark_roc.png`` / ``figures/benchmark_auc_bar.png``
figures.

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
import warnings
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
    "benchmark",
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
    # NOTE: "train" is model-dynamic — see _train_stage_artifacts(cfg).
    "evaluate": [
        "results/main_results.csv",
        "results/run_manifest.json",
    ],
    "qpu": [
        "results/vqc_qpu_probs.npy",
        "results/vqc_fakekingston_probs.npy",
    ],
    "analysis": ["results/cv_results.json"],
    "benchmark": [
        "results/benchmark_comparison.csv",
        "results/benchmark_mcnemar.csv",
    ],
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
    "benchmark": "results/benchmark_meta.json",
}


def _train_stage_artifacts(cfg: Config) -> list[str]:
    """Per-model artifact list for the ``train`` stage (model-dynamic).

    Builds the expected ``results/{model}_*.npy`` / ``.json`` / ``.pt``
    artifact paths from ``cfg.models`` so the cache check covers exactly the
    models the current configuration trains. The params artifact depends on
    the model kind: ``{model}_best_params.npy`` for VQC variants,
    ``{model}_best.pt`` for torch models and ``{model}_params.json`` for
    sklearn / quantum-kernel models.

    Args:
        cfg: Current configuration (``models`` field).

    Returns:
        List of artifact paths relative to the repo root.
    """
    from pipeline import models  # lazy: models.py imports numpy

    artifacts: list[str] = []
    for name in getattr(cfg, "models", ["vqc", "mlp"]):
        artifacts.append(f"results/{name}_val_probs.npy")
        artifacts.append(f"results/{name}_history.json")
        try:
            spec = models.get_model(name)
        except KeyError:
            continue
        if spec.kind == "vqc":
            artifacts.append(f"results/{name}_best_params.npy")
        elif spec.kind == "torch":
            artifacts.append(f"results/{name}_best.pt")
        else:
            artifacts.append(f"results/{name}_params.json")
    return artifacts


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
    if stage == "train":
        artifacts = _train_stage_artifacts(cfg)
    else:
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


def _save_train_artifacts(results_dir: Path, name: str, result: dict[str, Any]) -> None:
    """Save per-model train artifacts with the ``{name}_...`` prefix.

    Mirrors the artifact naming that ``pipeline.vqc.train_vqc`` /
    ``pipeline.mlp.train_mlp`` already use (``vqc_*`` / ``mlp_*``) so the
    analysis / qpu stages and the run manifest keep working unchanged. The
    params artifact is chosen by what the result dict carries: ``best_params``
    (VQC variants) → ``{name}_best_params.npy``, ``best_state`` (torch) →
    ``{name}_best.pt``, else ``params`` (sklearn / quantum kernel) →
    ``{name}_params.json``.

    Args:
        results_dir: Output directory (``cfg.results_dir``).
        name: Model name (registry key).
        result: Train result dict from the ``ModelSpec.train`` contract.
    """
    import numpy as np

    results_dir.mkdir(parents=True, exist_ok=True)

    np.save(results_dir / f"{name}_val_probs.npy", np.asarray(result["val_probs"]))
    if result.get("test_probs") is not None:
        np.save(results_dir / f"{name}_test_probs.npy", np.asarray(result["test_probs"]))

    with open(results_dir / f"{name}_history.json", "w", encoding="utf-8") as f:
        json.dump(result.get("history", {}), f, indent=2)

    if result.get("best_params") is not None:
        np.save(results_dir / f"{name}_best_params.npy", np.asarray(result["best_params"]))
    elif result.get("best_state") is not None:
        try:
            import torch  # lazy: torch is optional at import time

            torch.save(result["best_state"], results_dir / f"{name}_best.pt")
        except ImportError:
            with open(results_dir / f"{name}_params.json", "w", encoding="utf-8") as f:
                json.dump(result["best_state"], f, indent=2, default=str)
    elif result.get("params") is not None:
        with open(results_dir / f"{name}_params.json", "w", encoding="utf-8") as f:
            json.dump(result["params"], f, indent=2, default=str)


def _run_train(cfg: Config, dry_run: bool = False) -> None:
    """Dispatch the ``train`` stage: train every model in ``cfg.models``."""
    if dry_run:
        print(f"  → train: would train {', '.join(cfg.models)} → {_train_stage_artifacts(cfg)}")
        return
    import time

    import numpy as np
    from pipeline import models

    vae = _load_vae_features(cfg)
    X_tr, y_tr = vae["X_train"], vae["y_train"]
    X_va, y_va = vae["X_val"], vae["y_val"]
    X_te, y_te = vae["X_test"], vae["y_test"]

    results_dir = Path(cfg.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    trained: dict[str, int] = {}
    for name in cfg.models:
        spec = models.get_model(name)
        print(f"  ── {name} training ──")
        t0 = time.perf_counter()
        try:
            result = spec.train(cfg, X_tr, y_tr, X_va, y_va, X_te, y_te)
        except RuntimeError as exc:
            if spec.kind == "quantum_kernel":
                warnings.warn(
                    f"train: {name} requires missing dependencies: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                print(f"  ⚠ {name}: skipped: missing deps ({exc})")
                continue
            raise
        elapsed = time.perf_counter() - t0
        _save_train_artifacts(results_dir, name, result)
        trained[name] = int(result["epochs_trained"])
        print(f"  ✓ {name}: trained in {elapsed:.2f}s, best_epoch={result['epochs_trained']}")

    if trained:
        summary = ", ".join(f"{n} best_epoch={e}" for n, e in trained.items())
        print(f"  ✓ train: {summary}")


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
        plot_confusion_grid,
        plot_roc,
        plot_training_curves,
        save_results,
    )

    # Load ground-truth labels.
    feat = _load_features_from_disk(cfg)
    y_val = feat["y_val"]
    y_test = feat["y_test"]

    results_dir = Path(cfg.results_dir)
    tau_range = np.arange(
        cfg.threshold_range_min,
        cfg.threshold_range_max + cfg.threshold_step / 2.0,
        cfg.threshold_step,
    )

    probs_dict: dict[str, Any] = {}
    params_dict: dict[str, Any] = {}
    history_dict: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    test_probs_by_model: dict[str, np.ndarray] = {}
    preds_by_model: dict[str, np.ndarray] = {}

    for name in cfg.models:
        val_path = results_dir / f"{name}_val_probs.npy"
        if not val_path.is_file():
            warnings.warn(
                f"Skipping {name}: no saved probabilities",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        val_probs = np.load(val_path)
        test_path = results_dir / f"{name}_test_probs.npy"
        test_probs = np.load(test_path) if test_path.is_file() else val_probs

        # Threshold selection on validation set only (locked protocol).
        tau = find_best_threshold(val_probs, y_val, tau_range)
        m = compute_all_metrics(val_probs, y_val, tau)
        print(f"  evaluate: {name} tau={tau:.3f}")

        probs_dict[name] = {"val": val_probs, "test": test_probs}
        metrics[name] = m
        test_probs_by_model[name] = test_probs
        preds_by_model[name] = (test_probs > tau).astype(int)

        hist_path = results_dir / f"{name}_history.json"
        if hist_path.is_file():
            with open(hist_path, encoding="utf-8") as f:
                history_dict[name] = json.load(f)

        params_path = results_dir / f"{name}_best_params.npy"
        if params_path.is_file():
            params_dict[name] = np.load(params_path)
        else:
            pt_path = results_dir / f"{name}_best.pt"
            if pt_path.is_file():
                try:
                    import torch  # lazy: torch is optional at import time

                    params_dict[name] = torch.load(pt_path, map_location="cpu")
                except ImportError:
                    pass  # save_results handles non-torch fallback
            else:
                json_path = results_dir / f"{name}_params.json"
                if json_path.is_file():
                    with open(json_path, encoding="utf-8") as f:
                        params_dict[name] = json.load(f)

    if not metrics:
        warnings.warn(
            "evaluate: no model probabilities found — nothing to evaluate",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    save_results(cfg, metrics, probs_dict, params_dict, history_dict)

    # Figures.
    figures_dir = Path(cfg.figures_dir)
    try:
        for name, test_probs in test_probs_by_model.items():
            plot_roc(y_test, test_probs, figures_dir / f"roc_{name}.png", name.upper())
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ evaluate: ROC plot failed: {exc}")

    try:
        plot_confusion_grid(y_test, preds_by_model, figures_dir / "confusion_matrices.png")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ evaluate: confusion matrix plot failed: {exc}")

    try:
        if "vqc" in test_probs_by_model:
            plot_confidence_distribution(
                y_test,
                test_probs_by_model["vqc"],
                figures_dir / "confidence_distribution.png",
            )
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
    """Dispatch the ``analysis`` stage: t-SNE, calibration, SOTA, 5-fold CV,
    DANN validation and dataset-shift ablation."""
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

    # DANN domain discriminator validation.
    try:
        from pipeline.analysis import run_dann_validation

        dann_result = run_dann_validation(cfg)
        if dann_result is not None:
            pre_dann = dann_result.get("pre_dann", {}).get("acc_mean", "N/A")
            post_dann = dann_result.get("post_dann", {}).get("acc_mean", "N/A")
            print(f"  ✓ analysis: DANN discriminator — pre={pre_dann:.3f}, post={post_dann:.3f}")
        else:
            print("  ⚠ analysis: DANN validation skipped (no raw features)")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ analysis: DANN validation failed: {exc}")

    # Dataset-shift ablation (requires cfg.ablation_compare_dir).
    try:
        from pipeline.analysis import ablation_shift

        ablation_result = ablation_shift(cfg)
        if ablation_result is not None:
            rows = ablation_result.get("rows", [])
            print(f"  ✓ analysis: shift ablation — {len(rows)} rows written to "
                  f"results/ablation_shift.csv")
        else:
            print("  ⚠ analysis: shift ablation skipped (ablation_compare_dir empty)")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ analysis: shift ablation failed: {exc}")

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
# Benchmark stage (standalone — excluded from ``--stage all``)
# ---------------------------------------------------------------------------


def _round4(value: Any) -> Any:
    """Round float values to 4 decimals; pass everything else through.

    ``np.float64`` is a subclass of Python ``float``, so a single
    ``isinstance`` check covers both. Ints (``n_params``, ``epochs_trained``)
    and strings (``error``) pass through unchanged.
    """
    if isinstance(value, float):
        return round(value, 4)
    return value


def _benchmark_mcnemar(preds_a, preds_b, labels) -> tuple[float, float | None, str]:
    """Exact McNemar p-value (statsmodels) with scipy chi2 fallback.

    Builds the 2×2 discordant table from ``preds_a`` / ``preds_b`` against
    the ground-truth *labels* and runs the exact McNemar test via
    ``statsmodels.stats.contingency_tables.mcnemar`` (lazy import). When
    statsmodels is missing, falls back to the chi2 approximation with
    continuity correction ``stat = (|b-c| - 1)^2 / (b+c)`` on 1 dof.

    Args:
        preds_a: Hard predictions of classifier A, shape ``(N,)``.
        preds_b: Hard predictions of classifier B, shape ``(N,)``.
        labels: Ground-truth binary labels (0/1), shape ``(N,)``.

    Returns:
        ``(p_value, stat, method)`` — ``stat`` is ``None`` for the exact
        test and the chi2 statistic for the approximation.
    """
    preds_a = np.asarray(preds_a)
    preds_b = np.asarray(preds_b)
    labels = np.asarray(labels)

    b = float(np.sum((preds_a == labels) & (preds_b != labels)))  # A right, B wrong
    c = float(np.sum((preds_a != labels) & (preds_b == labels)))  # B right, A wrong

    try:
        from statsmodels.stats.contingency_tables import mcnemar

        a = float(np.sum((preds_a == labels) & (preds_b == labels)))
        d = float(np.sum((preds_a != labels) & (preds_b != labels)))
        table = np.array([[a, b], [c, d]])
        result = mcnemar(table, exact=True)
        return float(result.pvalue), None, "statsmodels-exact"
    except ImportError:
        try:
            from scipy.stats import chi2
        except ImportError as exc:
            raise RuntimeError(
                "benchmark McNemar requires statsmodels or scipy "
                "(pip install statsmodels)."
            ) from exc

        n = b + c
        if n == 0:
            return 1.0, 0.0, "scipy-chi2"
        stat = (abs(b - c) - 1.0) ** 2 / n
        return float(chi2.sf(stat, 1)), stat, "scipy-chi2"


def _plot_benchmark_auc_bar(rows: list[dict[str, Any]], path: Path) -> None:
    """Bar chart of test AUC per model (lazy matplotlib, Agg backend)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "benchmark AUC bar chart requires matplotlib "
            "(pip install matplotlib)."
        ) from exc

    names = [r["model"] for r in rows if r.get("test_auc") not in (None, "")]
    aucs = [float(r["test_auc"]) for r in rows if r.get("test_auc") not in (None, "")]
    if not names:
        raise ValueError("benchmark: no test AUC values to plot")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, aucs, color="tab:blue", alpha=0.85)
    ax.set_ylabel("Test AUC")
    ax.set_title("Benchmark test AUC by model")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3, axis="y")
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.01, f"{v:.4f}", ha="center", va="bottom", fontsize=8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_benchmark(cfg: Config, dry_run: bool = False) -> None:
    """Dispatch the ``benchmark`` stage: train all models + comparison CSVs.

    Trains every model in ``cfg.models`` (registry order) on the same VAE
    features, saves per-model artifacts under ``results/benchmark/<model>/``,
    then produces ``results/benchmark_comparison.csv`` (per-model metrics at
    each model's own validation-selected threshold), 
    ``results/benchmark_mcnemar.csv`` (exact McNemar vs ``vqc``) and the
    ``figures/benchmark_roc.png`` / ``figures/benchmark_auc_bar.png`` figures.
    Per-model ``RuntimeError`` (missing deps, sklearn absent) is recorded in
    the CSV ``error`` column and the loop continues.
    """
    if dry_run:
        print(f"  → benchmark: would train {', '.join(cfg.models)} → "
              f"results/benchmark/<model>/ + benchmark_comparison.csv")
        return
    import csv
    import time

    import numpy as np
    from pipeline import models
    from pipeline.evaluate import (
        compute_all_metrics,
        find_best_threshold,
        plot_multi_roc,
    )

    if not cfg.models:
        warnings.warn(
            "benchmark: cfg.models is empty — nothing to train",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    vae = _load_vae_features(cfg)
    X_tr, y_tr = vae["X_train"], vae["y_train"]
    X_va, y_va = vae["X_val"], vae["y_val"]
    X_te, y_te = vae["X_test"], vae["y_test"]

    results_dir = Path(cfg.results_dir)
    benchmark_dir = results_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    tau_range = np.arange(
        cfg.threshold_range_min,
        cfg.threshold_range_max + cfg.threshold_step / 2.0,
        cfg.threshold_step,
    )

    cfg_hash = config_hash(cfg)
    timestamp = datetime.now(timezone.utc).isoformat()

    rows: list[dict[str, Any]] = []
    test_probs_by_model: dict[str, np.ndarray] = {}
    preds_by_model: dict[str, np.ndarray] = {}

    for name in cfg.models:
        spec = models.get_model(name)
        model_dir = benchmark_dir / name
        model_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        try:
            result = spec.train(cfg, X_tr, y_tr, X_va, y_va, X_te, y_te)
            elapsed = time.perf_counter() - t0

            np.save(model_dir / "val_probs.npy", np.asarray(result["val_probs"]))
            if result.get("test_probs") is not None:
                np.save(model_dir / "test_probs.npy", np.asarray(result["test_probs"]))
            with open(model_dir / "history.json", "w", encoding="utf-8") as f:
                json.dump(result.get("history", {}), f, indent=2)

            n_params = result.get("n_params")
            if n_params is None:
                n_params = spec.param_count(cfg)

            meta = {
                "model": name,
                "kind": spec.kind,
                "n_params": n_params,
                "epochs_trained": result.get("epochs_trained"),
                "config_hash": cfg_hash,
                "timestamp": timestamp,
            }
            with open(model_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, default=str)

            print(f"  [benchmark] {name}: trained in {elapsed:.2f}s")

            # Validation-only threshold, then test metrics at that tau (locked
            # anti-data-leakage protocol — the test set is never scanned).
            val_probs = np.asarray(result["val_probs"], dtype=np.float64)
            test_probs = result.get("test_probs")
            if test_probs is None:
                test_probs = val_probs
            test_probs = np.asarray(test_probs, dtype=np.float64)

            best_tau = find_best_threshold(val_probs, y_va, tau_range)
            val_metrics = compute_all_metrics(val_probs, y_va, best_tau)
            test_metrics = compute_all_metrics(test_probs, y_te, best_tau)

            test_probs_by_model[name] = test_probs
            preds_by_model[name] = (test_probs > best_tau).astype(int)

            rows.append({
                "model": name,
                "kind": spec.kind,
                "n_params": n_params,
                "val_auc": val_metrics["roc_auc"],
                "test_auc": test_metrics["roc_auc"],
                "best_tau": best_tau,
                "test_accuracy": test_metrics["accuracy"],
                "test_bal_acc": test_metrics["balanced_accuracy"],
                "test_sensitivity": test_metrics["recall"],
                "test_specificity": test_metrics["specificity"],
                "test_f1": test_metrics["f1"],
                "epochs_trained": result.get("epochs_trained"),
                "training_time_s": elapsed,
                "error": "",
            })
        except RuntimeError as exc:
            warnings.warn(
                f"benchmark: {name} failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            print(f"  ⚠ benchmark: {name}: error: {exc}")
            rows.append({
                "model": name,
                "kind": spec.kind,
                "n_params": "",
                "val_auc": "",
                "test_auc": "",
                "best_tau": "",
                "test_accuracy": "",
                "test_bal_acc": "",
                "test_sensitivity": "",
                "test_specificity": "",
                "test_f1": "",
                "epochs_trained": "",
                "training_time_s": "",
                "error": str(exc),
            })

    # --- benchmark_comparison.csv ---
    columns = [
        "model", "kind", "n_params", "val_auc", "test_auc", "best_tau",
        "test_accuracy", "test_bal_acc", "test_sensitivity", "test_specificity",
        "test_f1", "epochs_trained", "training_time_s", "error",
    ]
    csv_path = results_dir / "benchmark_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _round4(v) for k, v in row.items()})

    # --- benchmark_mcnemar.csv (each model vs vqc, own best_tau on test) ---
    mcnemar_columns = ["model_a", "model_b", "p_value", "stat", "method", "mcnemar_note"]
    mcnemar_path = results_dir / "benchmark_mcnemar.csv"
    note = ("predictions binarized at each model's own best_tau applied to "
            "the test set (validation-only threshold protocol)")
    try:
        with open(mcnemar_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=mcnemar_columns)
            writer.writeheader()
            if "vqc" in preds_by_model:
                for name in preds_by_model:
                    if name == "vqc":
                        continue
                    p_value, stat, method = _benchmark_mcnemar(
                        preds_by_model[name], preds_by_model["vqc"], y_te
                    )
                    writer.writerow({
                        "model_a": name,
                        "model_b": "vqc",
                        "p_value": round(p_value, 4),
                        "stat": "" if stat is None else round(stat, 4),
                        "method": method,
                        "mcnemar_note": note,
                    })
    except RuntimeError as exc:
        warnings.warn(
            f"benchmark: McNemar test failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        print(f"  ⚠ benchmark: McNemar test failed: {exc}")

    # --- benchmark meta (config_hash for cache validation) ---
    meta_path = results_dir / "benchmark_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "config_hash": cfg_hash,
            "git_sha": _git_sha(),
            "timestamp": timestamp,
            "models": list(cfg.models),
        }, f, indent=2)

    # --- figures ---
    figures_dir = Path(cfg.figures_dir)
    try:
        plot_multi_roc(
            y_te,
            test_probs_by_model,
            figures_dir / "benchmark_roc.png",
            "Benchmark ROC Curves",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ benchmark: ROC plot failed: {exc}")

    try:
        _plot_benchmark_auc_bar(rows, figures_dir / "benchmark_auc_bar.png")
    except Exception as exc:  # noqa: BLE001
        print(f"  ⚠ benchmark: AUC bar plot failed: {exc}")

    print(f"  ✓ benchmark: {len(rows)} models → {csv_path}")


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
    "benchmark": _run_benchmark,
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
    ``benchmark`` is a standalone stage and is excluded from ``all``.

    Args:
        cfg: Current configuration.
        key: Cache key from :func:`build_cache_key`.
        dry_run: When ``True``, print only and do not execute.
    """
    all_stages = [s for s in STAGES if s not in ("all", "benchmark")]

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

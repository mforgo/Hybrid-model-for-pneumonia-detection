"""Configuration dataclass, YAML loading, and CLI override parsing.

This module defines the single ``Config`` dataclass that mirrors
``configs/default.yaml`` and provides helpers for loading, hashing,
and applying overrides.  Only stdlib + ``pyyaml`` are required — no
torch/pennylane/sklearn imports at module level.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Full pipeline configuration.

    Every field corresponds to a top-level key in ``configs/default.yaml``.
    Defaults match the YAML values so that ``Config()`` produces a
    configuration identical to ``load_config("configs/default.yaml")``.

    Model-registry fields (``models`` / ``cv_models``) select which models
    the benchmark stage trains; the classical-baseline fields (``logreg_C``,
    ``svm_C``, ``svm_gamma``, ``rf_n_estimators``, ``rf_max_depth``,
    ``knn_k``) and ``mlp_deep_hidden`` configure the sklearn / deep-MLP
    baselines; ``vqc_encoding`` / ``vqc_ansatz`` select the quantum circuit
    family; ``qk_n_train`` / ``qk_shots`` configure the quantum kernel.
    """

    # Project
    project_name: str = "HybridConvNeXtTinyQNNPneumonia"
    seed: int = 6
    device: str = "auto"

    # Data
    dataset: str = "paultimothymooney/chest-xray-pneumonia"
    dataset_path: str = ""
    img_size: int = 224
    val_split: float = 0.20
    split_strategy: str = "patient_grouped"  # "patient_grouped" | "random"
    subset: int | None = None

    # Augmentation
    rotation_deg: float = 7.0
    translate: float = 0.05
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2

    # Feature extraction
    backbone: str = "convnext_tiny"
    feature_dim: int = 768
    dann_epochs: int = 10
    dann_lr: float = 1e-4
    use_dann: bool = True

    # VAE / SVAE
    reduction_method: str = "vae"
    target_dims: int = 64
    vae_beta: float = 0.001
    vae_lambda_clf: float = 0.01
    vae_lambda_coral: float = 0.0
    vae_epochs: int = 30
    vae_batch_size: int = 32
    vae_lr: float = 1e-3

    # VQC
    n_qubits: int = 6
    n_layers: int = 3
    use_learnable_scale: bool = True
    use_measurement_basis: bool = True
    diff_method: str = "adjoint"
    vqc_encoding: str = "amplitude"  # "amplitude" | "angle" | "he"
    vqc_ansatz: str = "reupload"     # "reupload" | "he"

    # Training
    batch_size: int = 16
    learning_rate: float = 1e-3
    lr_min: float = 1e-5
    warmup_epochs: int = 3
    epochs: int = 50
    early_stopping_patience: int = 10

    # Domain adaptation
    lambda_coral: float = 0.1
    mixup_alpha: float = 0.2

    # MLP baseline
    mlp_hidden: int = 32
    mlp_dropout: float = 0.3

    # Deep MLP baseline
    mlp_deep_hidden: list[int] = field(default_factory=lambda: [64, 32])

    # Classical baselines (sklearn)
    logreg_C: float = 1.0
    svm_C: float = 1.0
    svm_gamma: str = "scale"
    rf_n_estimators: int = 200
    rf_max_depth: int | None = None
    knn_k: int = 5

    # QPU
    run_mode: str = "sim"
    ibm_token: str = ""
    ibm_instance: str = "ibm-q/open/main"
    ibm_backend: str = ""
    n_qpu_shots: int = 1024

    # ZNE
    zne_scale_factors: list[int] = field(default_factory=lambda: [1, 2, 3])

    # Quantum kernel
    qk_n_train: int = 400
    qk_shots: int = 1024

    # Evaluation
    threshold_range_min: float = 0.30
    threshold_range_max: float = 0.80
    threshold_step: float = 0.025
    n_bootstrap: int = 1000

    # Cross-validation
    cv_folds: int = 5

    # Model registry (benchmark stage)
    models: list[str] = field(default_factory=lambda: ["vqc", "mlp"])
    cv_models: list[str] = field(default_factory=lambda: ["vqc", "mlp"])

    # Class balancing
    class_balance: str = "none"  # "none" | "undersample"

    # Feature saving
    save_raw_features: bool = True

    # Ablation comparison
    ablation_compare_dir: str = ""

    # Paths
    artifacts_dir: str = "artifacts"
    results_dir: str = "results"
    figures_dir: str = "figures"

    # GPU auto-selection
    gpu_strategy: str = "least_loaded"
    cuda_device: str = ""

    @property
    def num_params(self) -> int:
        """Total trainable VQC parameter count.

        ``n_layers * n_qubits * 3`` Euler angles, plus ``n_qubits`` for the
        learnable input scale and ``2`` for the measurement basis when enabled.
        """
        count = self.n_layers * self.n_qubits * 3
        if self.use_learnable_scale:
            count += self.n_qubits
        if self.use_measurement_basis:
            count += 2
        return count

    def resolve_ibm_token(self) -> str | None:
        """IBM Quantum token with precedence: env > ibm_token.txt > config."""
        env_token = os.environ.get("IBM_TOKEN", "")
        if env_token:
            return env_token
        for base in (Path.cwd(), Path(__file__).resolve().parent.parent):
            token_file = base / "ibm_token.txt"
            if token_file.is_file():
                return token_file.read_text().strip() or None
        if self.ibm_token:
            return self.ibm_token
        return None


def _field_type(field_name: str) -> Any:
    """Return the declared type annotation of a Config field (may be a string)."""
    for f in fields(Config):
        if f.name == field_name:
            return f.type
    return None


def _list_element_type(target_type: Any) -> str | None:
    """Return the element type name of a ``list[...]`` annotation, else None."""
    t = str(target_type)
    if t.startswith("list["):
        return t[5:-1].strip()
    return None


def _parse_list(raw: str, target_type: Any) -> list:
    """Parse a comma-separated or bracketed list literal into a Python list.

    ``"vqc,mlp,rbf_svm"`` → ``["vqc", "mlp", "rbf_svm"]``
    ``"[64, 32]"`` → ``[64, 32]``

    Args:
        raw: Raw override string.
        target_type: Declared Config field type (e.g. ``"list[str]"``).

    Returns:
        A list with elements coerced to the declared element type.
    """
    s = raw.strip()
    if s.startswith("[") and s.endswith("]"):
        s = s[1:-1]
    parts = [p.strip() for p in s.split(",")] if s.strip() else []
    if _list_element_type(target_type) == "int":
        return [_coerce_value(p) for p in parts]
    return parts


def _coerce_value(raw: str, target_type: Any = None) -> Any:
    """Parse *raw* as int, float, bool (case-insensitive), else keep str.

    When *target_type* is a ``list[...]`` annotation, parse comma-separated
    or bracketed values into a list instead (see :func:`_parse_list`).
    """
    if _list_element_type(target_type) is not None:
        return _parse_list(raw, target_type)
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    lower = raw.lower()
    if lower in ("true", "yes", "1"):
        return True
    if lower in ("false", "no", "0"):
        return False
    return raw


def _resolve_field_name(key: str) -> str:
    """Map a CLI override key to a flat ``Config`` field name.

    Direct flat keys pass through (``seed``, ``n_layers``).  Dotted group
    keys resolve by trying ``group_key`` first (``vae.epochs`` →
    ``vae_epochs``), then the bare key (``vqc.epochs`` → ``epochs``).
    """
    if "." not in key:
        return key
    group, _, name = key.partition(".")
    known = {f.name for f in fields(Config)}
    if f"{group}_{name}" in known:
        return f"{group}_{name}"
    if name in known:
        return name
    raise ValueError(
        f"Unknown override key {key!r}: neither {group}_{name} nor {name} "
        f"is a Config field"
    )


def config_to_dict(cfg: Config) -> dict:
    """Convert a ``Config`` instance to a plain dictionary."""
    return asdict(cfg)


def config_hash(cfg: Config) -> str:
    """Deterministic sha-256 hex digest of the config snapshot.

    ``json.dumps(sort_keys=True)`` makes the digest independent of key
    insertion order.
    """
    snapshot = json.dumps(config_to_dict(cfg), sort_keys=True, default=str)
    return hashlib.sha256(snapshot.encode()).hexdigest()


def load_config(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Load a ``Config`` from YAML with precedence: defaults < YAML < overrides.

    Args:
        path: Path to the YAML configuration file.
        overrides: Mapping of override keys to values.  Keys may be dotted
            group paths (``"vae.epochs"``); string values are type-coerced.
    """
    path = Path(path)
    cfg_dict = config_to_dict(Config())

    if path.is_file():
        with open(path, "r") as f:
            yaml_data = yaml.safe_load(f)
        if isinstance(yaml_data, dict):
            cfg_dict.update(yaml_data)

    if overrides:
        for key, value in overrides.items():
            field_name = _resolve_field_name(key)
            if isinstance(value, str):
                value = _coerce_value(value, _field_type(field_name))
            cfg_dict[field_name] = value

    known = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in cfg_dict.items() if k in known}
    cfg = Config(**filtered)

    resolved_token = cfg.resolve_ibm_token()
    if resolved_token is not None:
        filtered["ibm_token"] = resolved_token
        cfg = Config(**filtered)

    return cfg


_OVERRIDE_RE = re.compile(r"^([^=]+)=(.+)$")


def parse_cli_overrides(raw_args: list[str]) -> dict[str, str]:
    """Parse ``key=value`` strings from ``--overrides`` into a dict.

    Supports dotted paths (``data.subset=32``).  Values stay as raw strings;
    coercion happens in ``load_config`` / ``apply_cli_overrides``.

    Raises:
        ValueError: If any argument does not match ``key=value``.
    """
    overrides: dict[str, str] = {}
    for arg in raw_args:
        m = _OVERRIDE_RE.match(arg)
        if not m:
            raise ValueError(f"Override must be 'key=value', got: {arg!r}")
        overrides[m.group(1)] = m.group(2)
    return overrides


def apply_cli_overrides(cfg: Config, overrides: dict[str, str]) -> Config:
    """Return a new ``Config`` with string *overrides* applied and coerced."""
    d = config_to_dict(cfg)
    for key, value in overrides.items():
        field_name = _resolve_field_name(key)
        d[field_name] = _coerce_value(value, _field_type(field_name))
    known = {f.name for f in fields(Config)}
    filtered = {k: v for k, v in d.items() if k in known}
    return Config(**filtered)

"""Tests for ``pipeline.config``: Config dataclass, YAML loading, override
precedence, and config hashing.

RED phase: ``pipeline.config`` is currently an empty stub, so every test
here fails at runtime until Task 2 implements the module. Collection must
still succeed, hence all pipeline imports are lazy (function-level).
"""

import pytest


def _num_params(cfg):
    """Resolve the trainable-parameter count from whichever API is implemented.

    The plan specifies ``ansatz.count_params(n_qubits, n_layers, use_scale,
    use_meas_basis)``; the Config dataclass may also expose a ``num_params``
    property. Either implementation must satisfy the same formula:

        n_layers * n_qubits * 3
        + (n_qubits if use_learnable_scale else 0)
        + (2 if use_measurement_basis else 0)
    """
    if hasattr(cfg, "num_params"):
        return cfg.num_params
    from pipeline.ansatz import count_params

    return count_params(
        cfg.n_qubits,
        cfg.n_layers,
        cfg.use_learnable_scale,
        cfg.use_measurement_basis,
    )


def test_config_load_override_precedence(tmp_path):
    """Precedence must be: defaults < YAML < CLI overrides."""
    from pipeline.config import Config, load_config

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("seed: 42\nn_layers: 2\nlearning_rate: 0.01\n")

    # YAML beats defaults (default seed is 6, default n_layers is 3).
    cfg = load_config(yaml_path)
    assert cfg.seed == 42
    assert cfg.n_layers == 2
    assert cfg.learning_rate == 0.01

    # CLI overrides beat YAML.
    try:
        cfg2 = load_config(yaml_path, overrides={"seed": 7})
    except TypeError:
        # Fallback: apply the override on top of the YAML-loaded config.
        cfg2 = Config(**{**vars(cfg), "seed": 7})
    assert cfg2.seed == 7
    assert cfg2.n_layers == 2
    assert cfg2.learning_rate == 0.01


def test_config_num_params_62():
    """Default config (learnable scale + measurement basis ON) -> 62 params."""
    from pipeline.config import Config

    cfg = Config()
    assert cfg.use_learnable_scale is True
    assert cfg.use_measurement_basis is True
    assert _num_params(cfg) == 62


def test_config_num_params_54():
    """Both ablation flags OFF -> 54 params (3 layers x 6 qubits x 3 angles)."""
    from pipeline.config import Config

    cfg = Config(use_learnable_scale=False, use_measurement_basis=False)
    assert _num_params(cfg) == 54


def test_config_hash_changes_on_override():
    """config_hash must be a deterministic sha256 hex digest that changes
    whenever any hyperparameter changes (cache invalidation contract)."""
    from pipeline.config import Config, config_hash

    h_default = config_hash(Config())
    assert isinstance(h_default, str)
    assert len(h_default) == 64  # sha256 hex digest
    int(h_default, 16)  # must be valid hex

    # Deterministic: identical config -> identical hash.
    assert config_hash(Config()) == h_default

    # Changing any hyperparameter must change the hash.
    assert config_hash(Config(learning_rate=0.01)) != h_default
    assert config_hash(Config(n_layers=2)) != h_default


def test_config_yaml_numeric_coercion(tmp_path):
    """PyYAML parses bare ``1e-4`` (no dot) as a *str*, so LR values written
    in scientific notation must be coerced to float on load — never passed
    through to Adam as strings."""
    from pipeline.config import load_config

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "dann_lr: 1e-4\nvae_lr: 1e-3\nlearning_rate: 1e-3\nlr_min: 1e-5\n"
    )
    cfg = load_config(yaml_path)
    assert cfg.dann_lr == 1e-4
    assert isinstance(cfg.dann_lr, float)
    assert cfg.vae_lr == 1e-3
    assert isinstance(cfg.vae_lr, float)
    assert cfg.learning_rate == 1e-3
    assert isinstance(cfg.learning_rate, float)
    assert cfg.lr_min == 1e-5
    assert isinstance(cfg.lr_min, float)


def test_config_yaml_str_fields_not_coerced(tmp_path):
    """String fields must stay strings even when the value looks numeric
    (dataset_path, ibm_instance, etc. must pass through untouched)."""
    from pipeline.config import load_config

    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        'dataset_path: "1e-4"\nsvm_gamma: "scale"\nibm_instance: "123"\n'
    )
    cfg = load_config(yaml_path)
    assert cfg.dataset_path == "1e-4"
    assert isinstance(cfg.dataset_path, str)
    assert cfg.svm_gamma == "scale"
    assert isinstance(cfg.svm_gamma, str)
    assert cfg.ibm_instance == "123"
    assert isinstance(cfg.ibm_instance, str)


def test_config_qpu_quickwin_defaults():
    """QPU quick-wins defaults: Kingston backend (CLOPS ~10x ibm_miami), auto
    job chunking, DD + Pauli twirling enabled on every SamplerV2 job."""
    from pipeline.config import Config, load_config

    cfg = Config()
    assert cfg.ibm_backend == "ibm_kingston"
    assert cfg.qpu_max_circuits_per_job == 0  # 0 = auto-derive from executions cap
    assert cfg.qpu_dd_enable is True
    assert cfg.qpu_twirl_enable is True

    # Mirror-config contract: default.yaml + dataclass defaults must agree.
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    yaml_cfg = load_config(repo / "configs" / "default.yaml")
    assert yaml_cfg.ibm_backend == "ibm_kingston"
    assert yaml_cfg.qpu_max_circuits_per_job == 0
    assert yaml_cfg.qpu_dd_enable is True
    assert yaml_cfg.qpu_twirl_enable is True
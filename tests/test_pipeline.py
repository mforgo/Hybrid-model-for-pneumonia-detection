"""Tests for ``pipeline.pipeline``: cache key / config hash validity.

RED phase: ``pipeline.pipeline`` and ``pipeline.config`` are empty stubs,
so this test fails at runtime until Tasks 2/14 implement the modules.
Collection must still succeed, hence all pipeline imports are lazy
(function-level).
"""


def test_cache_hash_valid():
    """The cache key must embed a valid sha256 config hash that changes when
    any hyperparameter changes (cache invalidation contract)."""
    from pipeline.config import Config, config_hash
    from pipeline.pipeline import build_cache_key

    cfg = Config()
    h = config_hash(cfg)
    assert isinstance(h, str)
    assert len(h) == 64  # sha256 hex digest
    int(h, 16)  # must be valid hex

    # Deterministic: identical config -> identical hash.
    assert config_hash(Config()) == h
    # Changing any hyperparameter must change the hash.
    assert config_hash(Config(learning_rate=0.01)) != h

    # The cache key embeds the config hash.
    key = build_cache_key(cfg)
    assert key["config_hash"] == h
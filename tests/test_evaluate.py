"""Tests for ``pipeline.evaluate``: metrics vs sklearn, val-only threshold
selection, and exact McNemar.

RED phase: ``pipeline.evaluate`` is an empty stub and scikit-learn is not
installed in the development environment, so these tests fail at runtime
until Task 11 implements the module. Collection must still succeed, hence
all heavy imports are lazy (function-level).
"""

import numpy as np
import pytest


def test_metrics_match_sklearn():
    """compute_all_metrics must match sklearn.metrics on synthetic data."""
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    from pipeline.evaluate import compute_all_metrics

    rng = np.random.default_rng(6)
    y = rng.integers(0, 2, size=40)
    probs = rng.random(40)
    tau = 0.5
    preds = (probs > tau).astype(int)

    m = compute_all_metrics(probs, y, tau)

    assert m["accuracy"] == pytest.approx(accuracy_score(y, preds))
    assert m["balanced_accuracy"] == pytest.approx(balanced_accuracy_score(y, preds))
    assert m["precision"] == pytest.approx(precision_score(y, preds, zero_division=0))
    assert m["recall"] == pytest.approx(recall_score(y, preds, zero_division=0))
    assert m["f1"] == pytest.approx(f1_score(y, preds, zero_division=0))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y, probs))

    tn = np.sum((preds == 0) & (y == 0))
    fp = np.sum((preds == 1) & (y == 0))
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    assert m["specificity"] == pytest.approx(specificity)


def test_threshold_uses_val_only():
    """find_best_threshold must pick the tau that maximizes balanced accuracy
    on the provided (validation) set only -- it never sees test labels."""
    from sklearn.metrics import balanced_accuracy_score

    from pipeline.evaluate import find_best_threshold

    rng = np.random.default_rng(6)
    val_probs = rng.random(40)
    val_labels = rng.integers(0, 2, size=40)
    tau_range = np.arange(0.30, 0.80, 0.025)

    tau = find_best_threshold(val_probs, val_labels, tau_range)
    assert 0.30 <= tau <= 0.80

    # tau must achieve the best balanced accuracy achievable on this set.
    best_score = max(
        balanced_accuracy_score(val_labels, (val_probs > t).astype(int))
        for t in tau_range
    )
    assert balanced_accuracy_score(val_labels, (val_probs > tau).astype(int)) == pytest.approx(best_score)


def test_mcnemar_exact():
    """mcnemar_exact must return a valid p-value from the exact (binomial)
    McNemar test on the 2x2 discordant table."""
    from pipeline.evaluate import mcnemar_exact

    # 11 discordant pairs: 10x A right / B wrong, 1x B right / A wrong.
    # Exact two-sided McNemar p = 2 * P(Binomial(11, 0.5) <= 1) ~ 0.0117 < 0.05.
    labels = np.array([0] * 60 + [1] * 60)
    preds_a = np.array([0] * 60 + [1] * 60)  # all correct
    preds_b = preds_a.copy()
    preds_b[0:10] = 1  # A right, B wrong
    preds_a[60] = 0    # B right, A wrong

    p = mcnemar_exact(preds_a, preds_b, labels)
    assert 0.0 <= p <= 1.0
    assert p < 0.05
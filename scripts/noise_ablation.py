#!/usr/bin/env python
"""Noise ablation sweep: AUC-ROC vs depolarising noise level.

Uses ``default.mixed`` with per-gate ``DepolarizingChannel`` to model
NISQ noise.  Sweeps ``p_noise`` ∈ {0, 0.001, 0.005, 0.01, 0.02, 0.05}
and computes AUC-ROC at each level on the full test set.

Outputs:
    figures/noise_ablation.png   — AUC-ROC vs p_noise curve
    results/noise_ablation.csv   — tabular results (p_noise, AUC, Acc, …)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SEED = 6


def _l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation."""
    X = np.asarray(X, dtype=np.float64)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms > 0, norms, 1.0)


def _build_noisy_circuit(n_qubits, n_layers, p_noise, use_scale, use_meas_basis):
    """Build a ``default.mixed`` QNode matching the trained ansatz structure.

    Every single-qubit gate (RY re-upload, Rot, measurement-basis RY/RZ)
    and every 2-qubit gate (CNOT on both control and target) is followed by
    a ``DepolarizingChannel(p_noise)``.  ``AmplitudeEmbedding`` is noise-free
    (state preparation).
    """
    import pennylane as qml

    dev = qml.device("default.mixed", wires=n_qubits)

    n_rot = n_layers * n_qubits * 3
    n_scale = n_qubits if use_scale else 0
    n_amp = 2**n_qubits

    @qml.qnode(dev)
    def circuit(params, features):
        rot = params[:n_rot].reshape((n_layers, n_qubits, 3))
        scale = params[n_rot : n_rot + n_scale] if use_scale else None
        meas = params[n_rot + n_scale :] if use_meas_basis else None

        # AmplitudeEmbedding — state preparation, no noise.
        qml.AmplitudeEmbedding(
            features[:n_amp], wires=range(n_qubits), normalize=True, pad_with=0.0
        )

        for l in range(n_layers):
            # RY data re-upload + depolarising.
            for q in range(n_qubits):
                if use_scale:
                    qml.RY(scale[q] * features[q] * np.pi, wires=q)
                else:
                    qml.RY(features[q] * np.pi, wires=q)
                qml.DepolarizingChannel(p_noise, wires=q)
            # Trainable Rot + depolarising.
            for q in range(n_qubits):
                qml.Rot(rot[l, q, 0], rot[l, q, 1], rot[l, q, 2], wires=q)
                qml.DepolarizingChannel(p_noise, wires=q)
            # Ring CNOT + depolarising on both qubits.
            for q in range(n_qubits):
                qml.CNOT(wires=[q, (q + 1) % n_qubits])
                qml.DepolarizingChannel(p_noise, wires=q)
                qml.DepolarizingChannel(p_noise, wires=(q + 1) % n_qubits)

        # Measurement basis (trainable) + depolarising.
        if use_meas_basis:
            qml.RY(meas[0], wires=0)
            qml.DepolarizingChannel(p_noise, wires=0)
            qml.RZ(meas[1], wires=0)
            qml.DepolarizingChannel(p_noise, wires=0)

        return qml.expval(qml.PauliZ(0))

    return circuit


def evaluate_auc(circuit, params, X, y, threshold=0.5):
    """Evaluate AUC-ROC and other metrics on a noisy circuit.

    Args:
        circuit: QNode callable ``circuit(params, features) -> float``.
        params: Flat parameter array (``split_params`` layout).
        X: Feature matrix, shape ``(N, 64)`` (L2-normalised externally).
        y: Labels in {0, 1}, shape ``(N,)``.
        threshold: Threshold for accuracy (default 0.5).

    Returns:
        Dict with ``auc``, ``accuracy``, ``recall``, ``specificity``.
    """
    from sklearn.metrics import roc_auc_score, balanced_accuracy_score

    X = _l2_normalize(X)
    y = np.asarray(y, dtype=np.int64)

    z = np.array([float(circuit(params, row)) for row in X])
    probs = (1.0 + z) / 2.0

    preds = (probs > threshold).astype(int)

    auc = float(roc_auc_score(y, probs))
    bal_acc = float(balanced_accuracy_score(y, preds))
    acc = float(np.mean(preds == y))

    # Sensitivity (recall for pneumonia class=1) and specificity.
    tp = int(np.sum((preds == 1) & (y == 1)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    return {
        "auc": auc,
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "recall": recall,
        "specificity": specificity,
    }


def main() -> None:
    np.random.seed(SEED)

    repo = Path(__file__).resolve().parent.parent
    results_dir = repo / "results"
    figures_dir = repo / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # --- Config (matches default.yaml) ---
    n_qubits = 6
    n_layers = 3
    use_scale = True
    use_meas = True

    # --- Load trained parameters and test data ---
    params = np.load(results_dir / "vqc_best_params.npy")
    X_test = np.load(repo / "artifacts" / "features" / "vae_test.npy")
    y_test = np.load(repo / "artifacts" / "features" / "y_test.npy").astype(np.int64)

    print(f"Loaded {len(X_test)} test samples, {params.shape[0]} parameters")

    # --- Noise sweep ---
    p_values = [0.0, 0.001, 0.005, 0.01, 0.02, 0.05]
    rows = []

    for p_noise in p_values:
        print(f"  p_noise={p_noise:.4f} ... ", end="", flush=True)
        circuit = _build_noisy_circuit(n_qubits, n_layers, p_noise, use_scale, use_meas)
        metrics = evaluate_auc(circuit, params, X_test, y_test)
        row = {"p_noise": p_noise, **metrics}
        rows.append(row)
        print(
            f"AUC={metrics['auc']:.4f}  Acc={metrics['accuracy']:.4f}  "
            f"BalAcc={metrics['balanced_accuracy']:.4f}"
        )

    # --- Save CSV ---
    csv_path = results_dir / "noise_ablation.csv"
    import csv

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path}")

    # --- Plot ---
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_arr = np.array([r["p_noise"] for r in rows])
    auc_arr = np.array([r["auc"] for r in rows])
    acc_arr = np.array([r["balanced_accuracy"] for r in rows])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(p_arr * 100, auc_arr, "o-", color="#2563eb", linewidth=2, markersize=6, label="AUC-ROC")
    ax.plot(p_arr * 100, acc_arr, "s--", color="#dc2626", linewidth=1.5, markersize=5, label="Balanced Accuracy")

    ax.set_xlabel("Depolarising noise rate $p$ (%)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("VQC Performance vs. Depolarising Noise (6 qubits, $L$=3)", fontsize=12)
    ax.set_ylim(0.45, 0.95)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    fig_path = figures_dir / "noise_ablation.png"
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved {fig_path}")


if __name__ == "__main__":
    main()

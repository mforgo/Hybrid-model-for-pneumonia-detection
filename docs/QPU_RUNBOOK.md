# QPU Hardware Runbook

Running the trained VQC on real IBM Quantum hardware with ZNE error mitigation.

## Prerequisites

1. **IBM Quantum account**: [quantum.ibm.com](https://quantum.ibm.com) — requires an IBM Quantum account with access to Heron r2 processors (156 qubits, ECR native gate set).

2. **Token** (set one of):
   ```bash
   export IBM_TOKEN="your-ibm-quantum-token"  # environment variable (highest priority)
   # OR
   echo "your-ibm-quantum-token" > ibm_token.txt  # repo root
   # OR
   # edit configs/default.yaml  →  ibm_token: "..."
   ```

3. **Deps** (all lazy-imported; graceful degradation if missing):
   ```bash
   pip install qiskit qiskit-ibm-runtime>=0.47 pennylane-qiskit==0.45.0 mitiq==1.0.0
   ```

## Running

### Full pipeline (train → evaluate → QPU)

```bash
python -m pipeline.main --stage data,features,vae,train,evaluate,qpu
```

### QPU-only (if training already done)

```bash
python -m pipeline.main --stage qpu
```

### Config overrides (CLI)

```bash
python -m pipeline.main --stage qpu \
  --set run_mode=both \
  --set n_qpu_shots=4096 \
  --set zne_scale_factors="[1, 2, 3, 5]"
```

| Config field | Default | Meaning |
|---|---|---|
| `run_mode` | `"sim"` | `"sim"` (simulator), `"qpu"` (hardware), `"both"` |
| `n_qpu_shots` | `1024` | Shots per circuit execution |
| `zne_scale_factors` | `[1, 2, 3]` | Noise scaling factors for ZNE |
| `ibm_token` | `""` | IBM Quantum API token |
| `ibm_backend` | `""` | Force a specific backend (else `least_busy`) |
| `ibm_instance` | `"ibm-q/open/main"` | IBM Quantum instance |

## Backend selection

```python
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=6)
```

Prefers **Heron r2** processors (`ibm_kingston` 156 qubits). Retired backends (`ibm_brisbane`, `ibm_sherbrooke`) are automatically excluded.

## Execution flow

```
┌──────────────────────────────────────────────────────┐
│  qpu_evaluate(cfg, circuit_qnode, params, X, y)     │
├──────────────────────────────────────────────────────┤
│                                                      │
│  1. Ideal-sim probs        (mode="sim" or "both")   │
│     └─ vqc_predict(X, params)                       │
│                                                      │
│  2. Real-hardware probs    (mode="qpu" or "both")   │
│     ├─ backend = get_ibm_backend(cfg)                │
│     ├─ run_vqc_on_qpu(circuit, params, X, backend)  │
│     └─ ZNE on first min(10, N) samples:              │
│        for each sample:                              │
│          run_zne_mitigation(                         │
│            circuit, params, x_i, backend,            │
│            executor, scale_factors=[1,2,3]           │
│          )                                           │
│                                                      │
│  3. FakeKingston noisy-sim  (always attempted)       │
│     └─ run_fakekingston_baseline(circuit, params, X)│
│                                                      │
│  4. Metrics on primary probs (if y available)        │
│     └─ compute_all_metrics(primary, y, tau=0.5)     │
│                                                      │
│  5. Persist artifacts → results/                     │
└──────────────────────────────────────────────────────┘
```

**Hardware mode is job-mode only** — `SamplerV2(mode=backend)` (no `Session`; Session bills wall-clock time).

## Output artifacts

| File | Contents |
|---|---|
| `results/vqc_qpu_probs.npy` | QPU inference probabilities (full test subset) |
| `results/vqc_qpu_sim_probs.npy` | Ideal-simulator probabilities |
| `results/vqc_fakekingston_probs.npy` | FakeKingston noisy-simulator probabilities |
| `results/zne_comparison.csv` | Per-sample table: `sim_prob`, `fakekingston_prob`, `raw_qpu_prob`, `zne_prob` |

## ZNE mitigation

- **Method**: Zero-Noise Extrapolation via [Mitiq](https://mitiq.readthedocs.io/)
- **Scaling**: Gate-folding at factors `[1, 2, 3]` (configurable)
- **Extrapolation**: Richardson extrapolation (`RichardsonFactory`)
- **Coverage**: Applied to first `min(10, N)` test samples (queue-time constraint)

```python
# How it works internally:
mitigated = mitiq.zne.execute_with_zne(
    circuit,
    executor=ibm_executor,            # wraps SamplerV2(mode=backend)
    factory=RichardsonFactory(scale_factors=[1, 2, 3]),
)
```

## Interpreting results

The `zne_comparison.csv` allows direct comparison across noise regimes:

| Column | Source | Expected trend |
|---|---|---|
| `sim_prob` | Ideal simulation | Best-case accuracy |
| `fakekingston_prob` | Noisy simulation (FakeKingston) | Baseline degradation |
| `raw_qpu_prob` | Physical hardware (raw) | Worst-case (noisy) |
| `zne_prob` | ZNE-mitigated hardware | Intermediate (noise reduced) |

**Quality check**: `zne_prob` should lie between `raw_qpu_prob` and `sim_prob` for well-conditioned circuits. If `zne_prob` overshoots `sim_prob`, the Richardson extrapolation is unstable — consider increasing `zne_scale_factors`.

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `"No IBM backend available"` | Token missing or no backends online | Verify `IBM_TOKEN` env var; check [system status](https://quantum.ibm.com/known-issues) |
| `"QPU inference failed"` | Queue timeout or backend maintenance | Retry; use `--set ibm_backend=ibm_kingston` for a specific backend |
| `"ZNE mitigation failed"` | Insufficient queue budget | Reduce `zne_scale_factors` to `[1, 3]` or `n_qpu_shots` to 512 |
| Empty `zne_comparison.csv` | ZNE skipped (error or `run_mode=sim`) | Set `--set run_mode=qpu` or `--set run_mode=both` |
| `ModuleNotFoundError: qiskit_ibm_runtime` | Missing dep | `pip install qiskit-ibm-runtime>=0.47` |

## Quick verification (no hardware)

```bash
# Verify FakeKingston noisy-sim baseline without an IBM token:
python -m pipeline.main --stage qpu --set run_mode=sim
# → produces vqc_qpu_sim_probs.npy + vqc_fakekingston_probs.npy (no token needed)
```

## Post-QPU analysis

After QPU run, load and compare:

```python
import numpy as np
import pandas as pd

qpu_probs = np.load("results/vqc_qpu_probs.npy")
sim_probs = np.load("results/vqc_qpu_sim_probs.npy")
zne_df    = pd.read_csv("results/zne_comparison.csv")

print(f"Sim AUC:   {roc_auc_score(y_test, sim_probs):.3f}")
print(f"QPU AUC:   {roc_auc_score(y_test, qpu_probs):.3f}")
print(zne_df.describe())
```

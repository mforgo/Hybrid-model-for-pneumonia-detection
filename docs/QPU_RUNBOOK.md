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
python -m pipeline --stage data,features,vae,train,evaluate,qpu
```

### QPU-only (if training already done)

```bash
python -m pipeline --stage qpu
```

### Config overrides (CLI)

```bash
python -m pipeline --stage qpu \
  --overrides run_mode=both \
  --overrides n_qpu_shots=4096 \
  --overrides zne_scale_factors="[1, 2, 3, 5]"
```

Note: the CLI takes `--overrides KEY=VALUE` pairs, **not** `--set` (the
`--stage/--set` syntax in older versions of this runbook was outdated).

| Config field | Default | Meaning |
|---|---|---|
| `run_mode` | `"sim"` | `"sim"` (simulator), `"qpu"` (hardware), `"both"` |
| `n_qpu_shots` | `1024` | Shots per circuit execution |
| `zne_scale_factors` | `[1, 2, 3]` | Noise scaling factors for ZNE |
| `ibm_token` | `""` | IBM Quantum API token |
| `ibm_backend` | `"ibm_kingston"` | Heron r2 (CLOPS ~10× `ibm_miami`); `""` = least-busy auto-select |
| `ibm_instance` | `""` | IBM Quantum instance (service picks automatically when empty) |
| `qpu_max_circuits_per_job` | `0` | Circuits per SamplerV2 job; `0` = auto from the ~10M executions/job cap |
| `qpu_dd_enable` | `true` | Dynamical decoupling on every SamplerV2 job |
| `qpu_twirl_enable` | `true` | Pauli gate twirling (32 randomizations) on every SamplerV2 job |

## Backend selection

```python
backend = service.backend(cfg.ibm_backend)  # default "ibm_kingston"
```

The default backend is **`ibm_kingston`** (Heron r2, 156 qubits). `ibm_miami`
measured ~10× lower CLOPS in the 2026 literature than other Heron backends,
so it is no longer the pin target; set `ibm_backend: ""` to fall back to
`service.least_busy(operational=True, simulator=False, min_num_qubits=6)`.
Retired backends (`ibm_brisbane`, `ibm_sherbrooke`) are automatically excluded.

## NISQ state-preparation bottleneck

The amplitude encoding of a 64-dimensional real vector into 6 qubits dominates the hardware circuit cost — it is *not* the trainable ansatz:

| Circuit block | 2-qubit gates | Notes |
|---|---|---|
| `AmplitudeEmbedding` (state prep) | **62 CNOTs** | PennyLane's Möttönen decomposition for a *real-valued* state: $2^6 - 2 = 62$ (Y-cascade only; the Z-cascade for complex states adds another 62 → 124 total). The general complex-state bound of Shende–Bullock–Markov is $2^{n+1} - 2n = 116$ at $n=6$; the leading-order Iten et al. bound is $\approx 23/24 \cdot 2^n \approx 61$. |
| Trainable ansatz (3 layers) | 18 CNOTs | 3 layers × 6 ring-CNOTs (qubit $w \to w{+}1 \bmod 6$). |
| Measurement basis | 0 | local RY + RZ on qubit 0. |
| **Total** | **80 CNOTs** | encoding is **≈ 3.4× deeper than the classifier**. |

**Why this matters on hardware:**

- The 62 state-prep CNOTs are *fixed* (non-trainable) — noise in them corrupts every sample identically, and ZNE cannot "train them away" like ansatz angles.
- After transpilation to the IBM native gate set (`ECR`, `RZ`, `SX`, `X`) with connectivity routing, the actual 2-qubit-gate count is *at least* 80 (typically higher due to SWAP insertion on the 6-qubit coupling map).
- Expected hierarchy on a real device: state-preparation noise ≫ ansatz noise. The measured hardware accuracy drop vs. the ideal simulator is therefore primarily attributable to encoding, not to the learned classifier — an explicit limitation to report in the thesis (Section 6.x, "NISQ state-preparation bottleneck").

**Mitigation posture:**

- ZNE *does* fold through state-prep gates (gate folding is applied to the whole compiled circuit), so the `zne_prob` column of `zne_comparison.csv` captures encoding noise too. Caveat: at scale factor 3 the ~80-gate circuit becomes ~240 gates, stretching coherence budgets, and the 62 state-prep CNOTs alone become ~186.
- Mitiq has **no built-in "skip state prep" option** — excluding the encoding block (e.g., to isolate ansatz noise) requires a custom `scale_noise` function. Per-gate fidelity weights (`fidelities={"single": 1.0}`) only skip single-qubit gates, not CNOTs.
- Do **not** silently drop `run_mode=sim` comparison: the sim-vs-hardware gap IS the state-prep bottleneck quantification.
- For future work: consider structure-preserving approximate encoding (e.g., reduced-amplitude or variational state preparation) or re-uploading-style angle encoding to cut the 62-CNOT fixed overhead — both trade expressibility for shallower circuits.

**Differentiability caveat:** PennyLane skips the Z-cascade only when the state is real-typed *and* non-differentiable. Training with `requires_grad=True` features would emit 124 CNOTs instead of 62. Frozen `.npy` features (as in this pipeline) avoid this, but it matters if encoding is ever made end-to-end differentiable.

**References:** M. Möttönen et al., *Transformation of quantum states using uniformly controlled rotations*, Quantum Inf. Comput. 5(6):467 (2005), quant-ph/0407010; V. Shende, S. Bullock, I. Markov, *Synthesis of quantum logic circuits*, IEEE TCAD 25(6):1000 (2006), quant-ph/0406176; R. Iten et al., *Quantum circuits for isometries*, Phys. Rev. A 93, 032318 (2016), arXiv:1501.06911 (used by Qiskit `StatePreparation`).

## Execution flow

```
┌──────────────────────────────────────────────────────┐
│  _run_qpu(cfg)  →  _run_qpu_evaluate(cfg, ...)      │
├──────────────────────────────────────────────────────┤
│                                                      │
│  0. select_qpu_subset(X_test, y_test,               │
│                        n_qpu_samples, seed=SEED)     │
│     └─ stratified, class-balanced, seeded shuffle    │
│     └─ returns (X_sub, y_sub, idx) + persists idx    │
│        → results/qpu_sample_indices.npy              │
│                                                      │
│  then qpu_evaluate(cfg, circuit_qnode, params,       │
│                    X_sub, y_sub):                    │
│  1. Ideal-sim probs        (mode="sim" or "both")   │
│     └─ vqc_predict(X_sub, params)                    │
│                                                      │
│  2. Real-hardware probs    (mode="qpu" or "both")   │
│     ├─ backend = get_ibm_backend(cfg)                │
│     ├─ run_vqc_on_qpu(circuit, params, X_sub,        │
│     │                    backend)                    │
│     │   └─ ALL circuits in ONE SamplerV2 job         │
│     │      (chunked at qpu_max_circuits_per_job;     │
│     │       0 = auto from ~10M executions/job cap)   │
│     └─ ZNE on first min(10, N) samples, batched:    │
│        run_zne_batched(circuit, params, X[:10],      │
│                        backend, scale_factors)       │
│          └─ folds 10×3 circuits, ONE job             │
│             + Richardson extrapolation               │
│                                                      │
│  3. FakeKingston noisy-sim  (always attempted)       │
│     └─ run_fakekingston_baseline(circuit, params,    │
│                                   X_sub)             │
│                                                      │
│  4. Metrics on primary probs (if y available)        │
│     └─ compute_all_metrics(primary, y_sub, tau=cfg.qpu_tau) │
│                                                      │
│  5. Persist artifacts → results/                     │
└──────────────────────────────────────────────────────┘
```

**The evaluation subset is always class-balanced.** The test split is
label-sorted (all negatives precede all positives), so a naive `X_test[:n]`
slice evaluates a single class and makes AUC/sensitivity undefined — this
is what corrupted the 2026-09-15 `ibm_miami` run (79 jobs on an all-negative
slice). `select_qpu_subset` draws `ceil(n/2)` positives + `floor(n/2)`
negatives with the project seed and persists the original row indices so the
job → sample mapping can always be reconstructed.

**Hardware mode is job-mode only** — `SamplerV2(mode=backend)` (no `Session`; Session bills wall-clock time).

## Output artifacts

| File | Contents |
|---|---|
| `results/vqc_qpu_probs.npy` | QPU inference probabilities (class-balanced test subset) |
| `results/vqc_qpu_sim_probs.npy` | Ideal-simulator probabilities |
| `results/vqc_fakekingston_probs.npy` | FakeKingston noisy-simulator probabilities |
| `results/qpu_sample_indices.npy` | Original test-row indices of the balanced subset (job → sample mapping) |
| `results/zne_comparison.csv` | Per-sample table: `sim_prob`, `fakekingston_prob`, `raw_qpu_prob`, `zne_prob` |

## ZNE mitigation

- **Method**: Zero-Noise Extrapolation via [Mitiq](https://mitiq.readthedocs.io/)
- **Scaling**: Gate-folding at factors `[1, 2, 3]` (configurable)
- **Extrapolation**: Richardson extrapolation (`RichardsonFactory`)
- **Coverage**: Applied to first `min(10, N)` test samples; all 30 folded
  circuits (10 samples × 3 factors) run in **one** batched SamplerV2 job
  (`run_zne_batched`), then extrapolated per sample. No per-sample job loop.

```python
# How it works internally (batched, single job):
from mitiq.zne.inference import RichardsonFactory
from mitiq.zne.scaling import fold_gates_at_random

jobs = []  # (sample_idx, scale_idx, folded_circuit)
for i, x in enumerate(X_subset):
    qc = _qnode_to_qiskit(circuit_qnode, params, x)
    for k, sf in enumerate(scale_factors):
        jobs.append((i, k, fold_gates_at_random(qc, sf)))
# 1 SamplerV2 job over all folded circuits → z0[sample, scale]
# then per-sample: mitigated = RichardsonFactory.extrapolate(scale_factors, z0[i])
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
| `"QPU inference failed"` | Queue timeout or backend maintenance | Retry; use `--overrides ibm_backend=ibm_kingston` for the Heron r2 default |
| `"ZNE mitigation failed"` | Insufficient queue budget | Reduce `zne_scale_factors` to `[1, 3]` or `n_qpu_shots` to 512 |
| Empty `zne_comparison.csv` | ZNE skipped (error or `run_mode=sim`) | Set `--overrides run_mode=qpu` or `--overrides run_mode=both` |
| `ModuleNotFoundError: qiskit_ibm_runtime` | Missing dep | `pip install qiskit-ibm-runtime>=0.47` |

## Quick verification (no hardware)

```bash
# Verify FakeKingston noisy-sim baseline without an IBM token:
python -m pipeline --stage qpu --overrides run_mode=sim
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

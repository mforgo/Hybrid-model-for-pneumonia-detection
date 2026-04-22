# QML Pneumonia Detection - Complete Reference

> Consolidated documentation for SOČ 2026 hybrid quantum-classical ML project  
> Author: Michal Forgó

---

## Architecture

```
X-Ray (5856) → ConvNeXt-Tiny → Autoencoder → VQC (6q, L=3, 62p) vs MLP (2113p)
```

| Component | Specification |
|----------|----------------|
| Backbone | ConvNeXt-Tiny (frozen) |
| Feature dim | 768 → 64 |
| Qubits | 6 (2⁶ = 64) |
| Layers (L) | 3 data re-uploading |
| VQC params | 62 (54 rot + 6 scale + 2 meas) |
| Entanglement | Linear CNOT |
| Optimizer | Adam (cosine LR) |
| Loss | Weighted MSE |

---

## Results (Test Set, n=624)

| Metric | MLP | VQC | Δ |
|--------|-----|-----|-----|
| Accuracy | 82.53% | 81.25% | -1.28% |
| AUC-ROC | 0.940 | 0.860 | -0.080 |
| Params | 2,113 | **62** | 34× fewer |

---

## Enhancements Integrated (v5f9fef2)

| Enhancement | Params Added | Status |
|-------------|-------------|---------|
| Learnable scale w | +6 | ✅ Done |
| Trainable measurement | +2 | ✅ Done |
| Linear CNOT | 0 | ✅ Done |
| QNG optimizer | — | Tested |
| ZNE | — | Future work |

---

## Pipeline Phases

| Phase | Notebook | Output |
|-------|----------|--------|
| 1. Data | `01_*.ipynb` | Images → paths |
| 2. Features | `02_*.ipynb` | Features .npy |
| 3. VQC Training | `02_vqc_*.ipynb` | params.npy |
| 4. Evaluation | `03_*.ipynb` | Metrics + PDF |
| 5. Hardware | `04_ibm_*.ipynb` | QPU results |

---

## Key Papers

| Topic | Citation |
|-------|----------|
| Data re-uploading | Pérez-Salinas et al., *Quantum* 4:226 (2020) |
| Expressibility | Sim et al., *AQN* 2:1900070 (2019) |
| Local cost/Barren plateaus | Cerezo et al., *Nat Commun* 12:1791 (2021) |
| ZNE | Temme et al., *PRL* 119:180509 (2017) |
| VAE | Kingma & Welling, *ICLR* 2014 |

---

## Further Improvements (Priority)

| # | Improvement | Effort | Impact |
|---|-------------|--------|--------|
| 1 | VAE (replace MSE AE) | Medium | High |
| 2 | ZNE noise sweep | Low | High |
| 3 | Ensemble MLP+VQC | Low | Medium |
| 4 | Hardware eval (IBM) | High | Medium |

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `PLAN.md` | 466 | Implementation plan |
| `AGENTS.md` | 418 | Agent config |
| `soc.tex` | 1305 | Thesis |
| `*.ipynb` | — | Code notebooks |

---

## Build Thesis

```bash
cd docs/soc
pdflatex soc.tex
# Output: soc.pdf (64 pages, 3.35 MB)
```

---

*Last updated: 2026-04-22*
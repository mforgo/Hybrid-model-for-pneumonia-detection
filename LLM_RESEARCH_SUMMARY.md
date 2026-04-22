# LLM Research Summary: Further QML Improvements for SOČ Thesis

> This document provides a comprehensive overview for researching additional improvements to the hybrid QML pneumonia detection thesis.

---

## Current State (as of commit 5f9fef2)

### Architecture Summary
```
Chest X-Ray → ConvNeXt-Tiny (768D) → Autoencoder (64D) → VQC (6 qubits, 62 params)
                                                              ↓
                                              Classical MLP (2,113 params) as baseline
```

### Current VQC Specifications
- **Qubits**: 6 (matches 2^6 = 64 PCA dimensions)
- **Layers (L)**: 3 (data re-uploading)
- **Total Parameters**: 62
  - Rot layers: 54 (3 × 6 × 3)
  - Encoding scale: 6 (learnable w_i)
  - Measurement: 2 (θ_ry, θ_rz)
- **Entanglement**: Linear CNOT (i, i+1)
- **Measurement**: Pauli-Z with trainable basis rotations
- **Optimizer**: Adam (tested QNG but not used)
- **Loss**: Weighted MSE on ±1 targets

### Current Results (Simulator)
| Metric | MLP | VQC |
|--------|-----|-----|
| Accuracy | 82.53% | 81.25% |
| AUC-ROC (test) | 0.940 | 0.860 |
| Parameters | 2,113 | 62 |

---

## What's Implemented (Enhancement Integration)

### ✅ Already Integrated in soc.tex:
1. **Parameter count**: 62 (was 54)
2. **Enhanced encoding**: 6 learnable scale parameters w_i
3. **Trainable measurement**: 2 parameters (θ_ry, θ_rz)
4. **Linear entanglement**: CNOT(i, i+1) - no SWAP overhead
5. **QNG mention**: Tested but Adam performed better

### ❌ Not Yet Integrated:
1. **VAE** - Only mentioned in limitations (future work)
2. **ZNE** - Only mentioned in future work
3. **Hardware evaluation results** - No real QPU testing

---

## Opportunities for Further Improvement

### Category A: Architecture Improvements

#### A1. Deeper ansatz exploration
- **Current**: L = 3 layers
- **Possible**: Test L ∈ {2, 3, 4, 5} more rigorously
- **Trade-off**: Expressivity vs. noise sensitivity
- **Papers**: Sim et al. (2019) for expressibility metrics

#### A2. Different ansatz architectures
- **Current**: Data re-uploading (Pérez-Salinas 2020)
- **Alternatives**:
  - Hardware-efficient ansatz (HEA)
  - Strongly entangled ansatz
  - QAOA-inspired ansatz
- **Trade-off**: Barren plateaus vs. classification power

#### A3. Quantum convolutional layers (Quanvolutional)
- **Concept**: Quantum gates moving across input like CNN
- **Potential**: Better feature extraction
- **Papers**: Henderson et al. (2020)

#### A4. Quantum kernel methods
- **Current**: Variational circuit (parameterized)
- **Alternative**: Fixed quantum kernel + classical SVM
- **Papers**: Havlíček et al. (2019)

---

### Category B: Training Improvements

#### B1. Different optimizers
- **Current**: Adam with cosine LR schedule
- **Possible**:
  - RAdam (Rectified Adam)
  - LARS (Layer-wise Adaptive Rate)
  - QNG (tested but rejected)
  - SGD with momentum

#### B2. Loss function alternatives
- **Current**: Weighted MSE on ±1 targets
- **Alternatives**:
  - Cross-entropy with quantum labels
  - Hinge loss
  - Focal loss

#### B3. Batch normalization in quantum circuit
- **Concept**: Normalize quantum states between layers
- **Potential**: More stable training

---

### Category C: Error Mitigation (Hardware)

#### C1. Zero-Noise Extrapolation (ZNE) - PARTIAL
- **Status**: Mentioned in future work
- **Needed**: Actually run on IBM hardware
- **Methods**:
  - Richardson extrapolation
  - Zero-noise extrapolation
  - Virtual distillation

#### C2. Probabilistic Error Cancellation (PEC)
- **Concept**: Invert noise channel probabilities
- **Library**: Mitiq

#### C3. Readout Error Mitigation
- **Concept**: Calibrate measurement errors
- **Library**: QiskitIgnis

#### C4. Dynamical Decoupling
- **Concept**: Pulse sequences to suppress noise
- **Papers**: Viola et al. (1999)

---

### Category D: Dataset & Preprocessing

#### D1. VAE latent space
- **Current**: MSE autoencoder (deterministic)
- **Enhancement**: Variational Autoencoder (Kingma 2014)
- **Benefit**: Smooth latent space, better regularization
- **Implementation**:
  - Add KL divergence term: β = 1.0
  - Reparameterization trick

#### D2. Different feature extractors
- **Current**: ConvNeXt-Tiny
- **Tested (failed)**: ViT-B/16
- **Possible**:
  - ResNet-50
  - EfficientNet
  - DenseNet-121

#### D3. Data augmentation
- **Current**: RandAugment, ColorJitter, HorizontalFlip
- **Possible**:
  - CutMix
  - MixUp
  - AutoAugment

---

### Category E: Ensemble Methods

#### E1. Classical + Quantum ensemble
- **Concept**: Combine MLP and VQC predictions
- **Methods**:
  - Weighted average
  - Stacking
  - Voting

#### E2. Multiple quantum models
- **Concept**: Different ansatz + voting
- **Benefit**: Reduce variance

---

### Category F: Interpretability

#### F1. More comprehensive Grad-CAM
- **Current**: Single examples
- **Possible**: Full statistical analysis

#### F2. Quantum feature attribution
- **Concept**: Which qubits contribute most
- **Methods**: Saliency maps in QML

#### F3. Decision boundary visualization
- **Concept**: 2D projection of quantum space
- **Method**: t-SNE on quantum statevector

---

## Quick Wins (High Impact, Low Effort)

| Improvement | Effort | Impact | Priority |
|-------------|--------|--------|----------|
| VAE (replace MSE AE) | Medium | High | 1 |
| ZNE on simulator | Low | High | 2 |
| Hardware eval (small) | High | Medium | 3 |
| Ensemble MLP+VQC | Low | Medium | 4 |
| Deeper L search | Low | Medium | 5 |

---

## Key Papers to Research

| Topic | Citation | Status |
|-------|---------|--------|
| VAE | Kingma & Welling (2014) | Not implemented |
| Expressibility | Sim et al. (2019) | Theory done |
| ZNE | Temme et al. (2017) | Future work |
| Quanvolutional | Henderson et al. (2020) | Not researched |
| Quantum kernel | Havlíček et al. (2019) | Brief mention |
| PEC | Endo et al. (2018) | Not researched |
| Barren plateaus | McClean et al. (2015) | Theory done |

---

## Recommended Focus for Secondary School Project

Given the constraints of a SOČ project:

### Realistic Improvements:
1. **VAE** - Replace MSE autoencoder (~20 lines code, ~40 thesis lines)
2. **ZNE simulation** - Run noise sweep on simulator (~10 lines code)
3. **More rigorous L search** - Already have theory, need results

### Stretch Goals (if time permits):
1. **Small hardware eval** - 10-20 samples on IBM
2. **Ensemble** - Average MLP + VQC predictions

### Low Priority (for academic depth only):
1. New ansatz architectures (would require new training runs)
2. Multiple feature extractors (already tested ViT, failed)

---

## Code Locations for Implementation

| Component | Notebook | Lines |
|----------|---------|-------|
| Autoencoder | `02_preprocessing.ipynb` | ~100-200 |
| VQC | `02_vqc_training.ipynb` | ~200-400 |
| Evaluation | `03_evaluation.ipynb` | ~100-200 |
| Hardware | `04_ibm_hardware_eval.ipynb` | ~50-100 (template) |

---

## BibTeX Entries to Add

```bibtex
@article{kingma_vae,
  author = {Kingma, D.P. and Welling, M.},
  title = {Auto-Encoding Variational Bayes},
  journal = {ICLR},
  year = {2014}
}

@article{temme_zne,
  author = {Temme, K. and Bravyi, S. and Gambetta, J.M.},
  title = {Error Mitigation to Reach Quantum Computational Advantage},
  journal = {Phys. Rev. Lett.},
  year = {2017},
  volume = {119},
  pages = {180509}
}
```

---

*Last updated: 2026-04-22*
*Current thesis: soc.tex (1,305 lines)*
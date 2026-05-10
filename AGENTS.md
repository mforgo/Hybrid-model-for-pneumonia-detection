# Hybrid QML Pneumonia Detection - Complete Agent Configuration

**Project:** Středoškolská odborná činnost (SOČ) 2026  
**Author:** Michal Forgó

This document contains:
1. **Part A:** AI Agent Personas (strategic roles)
2. **Part B:** Pipeline Agents (technical components)

---

# Part A: AI Agent Personas

This document defines the specialized AI agent personas contributing to the research, development, and documentation of the hybrid quantum-classical neural network for medical diagnostics.

---

## 🔬 Agent 1: Deep Learning & Computer Vision Specialist

**Role:** PyTorch Architect & Domain Adaptation Expert  
**Goal:** Optimize the classical feature extraction pipeline, manage data imbalances, and implement advanced domain adaptation techniques.  
**Tech Stack:** PyTorch, Torchvision, Grad-CAM, PIL.

**Responsibilities & Context:**
* **Backbone Management:** Manage the `ConvNeXt-Tiny` backbone used to extract features from pediatric chest X-rays originating from the Guangzhou Women and Children's Medical Center.
* **Domain Adaptation:** Implement and maintain the Domain-Adversarial Neural Network (DANN) using a Gradient Reversal Layer (GRL) to mitigate the dataset shift between the training set (74.2% pneumonia) and testing set (62.5% pneumonia).
* **Interpretability:** Maintain the `pytorch-grad-cam` implementation to generate clinical heatmaps from the final convolutional layers, ensuring the model avoids black-box decision making.

---

## ⚛️ Agent 2: Quantum Machine Learning (QML) Engineer

**Role:** NISQ-Era Quantum Algorithm Designer  
**Goal:** Design, simulate, and execute the Variational Quantum Classifier (VQC) with maximum parameter efficiency.  
**Tech Stack:** PennyLane (`lightning.gpu`), Qiskit, IBM Quantum Runtime, Mitiq.

**Responsibilities & Context:**
* **Ansatz Design:** Implement the data re-uploading ansatz with L=3 layers to maximize expressibility using only 6 qubits.
* **Parameter Efficiency:** Ensure the VQC maintains exactly 54 trainable parameters to demonstrate extreme parameter efficiency compared to the classical MLP (39× fewer).
* **Hardware Execution:** Prepare the quantum circuits for physical execution on IBM Quantum processors, utilizing Zero-Noise Extrapolation (ZNE) to mitigate NISQ-era decoherence.
* **Gradient Optimization:** Use PennyLane's adjoint differentiation on classical simulators for fast training, switching to parameter-shift rules only when targeting physical hardware.

---

## 📊 Agent 3: Data Scientist & Statistician

**Role:** Evaluation & Metrics Lead  
**Goal:** Ensure rigorous, statistically sound evaluation of all models, completely avoiding misleading single-point metric claims.  
**Tech Stack:** Scikit-learn, SciPy, Pandas, NumPy.

**Responsibilities & Context:**
* **Dimensionality Reduction:** Manage the nonlinear autoencoder (768→256→64) compressing the `ConvNeXt-Tiny` features down to 64 dimensions, applying strict $L_2$-normalization for amplitude encoding.
* **Statistical Testing:** Execute McNemar's test with continuity correction to prove statistically significant differences between the classical MLP and the VQC.
* **Confidence Intervals:** Run non-parametric bootstrap resampling (B=1000 iterations) to generate 95% confidence intervals for AUC-ROC and Accuracy.
* **Thresholding:** Optimize the decision threshold on the validation set using Balanced Accuracy to handle the inherent class imbalance.

---

## ✍️ Agent 4: Academic Writer & Reviewer (SOČ)

**Role:** Thesis Formatting and Scientific Integrity Guardian  
**Goal:** Ensure the written thesis meets university-level academic standards in the Czech language, matching the formal requirements of the SOČ competition.

**Responsibilities & Context:**
* **Tone & Candor:** Maintain absolute scientific honesty. Explicitly state when hypotheses are not confirmed (e.g., acknowledging the classical model outperforms the VQC in pure accuracy).
* **Framing:** Frame the results around *parameter efficiency* and *NISQ limitations*, rather than claiming absolute superiority of quantum models.
* **Formatting:** Structure the text for scannability, utilizing proper citations, objective language, and rigorous discussion of limitations (e.g., computational limits, single-dataset reliance).

---

## ⚙️ Global Operating Rules for All Agents

1. **Reproducibility First:** Always respect the global random seed (`SEED = 6`).
2. **Resource Awareness:** Remember that training runs on a single Google Colab A100 GPU; optimize code to avoid Out-Of-Memory (OOM) errors and minimize training time.
3. **No Information Loss:** When editing the pipeline, ensure the output manifest (metrics, arrays, and PDF figures) remains intact and properly saved to the Colab `/content/results` directories.

---

# Part B: Pipeline Software Agents

> AI agents and automated tools used in the hybrid quantum-classical pneumonia detection project.
> Each entry documents the agent's role, inputs, outputs, and integration point in the pipeline.

---

## 1. ConvNeXt-Tiny Feature Extractor

**Type:** Pre-trained ConvNeXt-Tiny (classical)  
**Framework:** PyTorch / torchvision  
**Location:** `02_preprocessing.ipynb`

**Role:**
Acts as a frozen visual encoder. Transforms raw chest X-ray images into compact, high-level feature vectors. The classification head is removed; the network outputs the 768-dimensional pooled feature vector.

**Inputs:**
- JPEG chest X-ray images, resized to 224×224
- Normalised with ImageNet statistics (mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`)

**Outputs:**
- `convnext_tiny_pca_train.npy` — shape `(4185, 64)`
- `convnext_tiny_pca_val.npy`   — shape `(1047, 64)`
- `convnext_tiny_pca_test.npy`  — shape `(624,  64)`
- `convnext_tiny_scaler.pkl` — fitted `StandardScaler`
- `convnext_tiny_pca.pkl`    — fitted `PCA(n_components=64)`

**Key settings:**
```python
model = torchvision.models.convnext_tiny(weights="IMAGENET1K_V1")
model.classifier = torch.nn.Identity()   # remove classification head
model.eval()                     # frozen — no gradient updates
```

**Notes:**
Run once; outputs are cached to `artifacts/features/`. All subsequent experiments load `.npy` arrays directly, skipping this step.

---

## 2. VAE Dimensionality Reducer

**Type:** Variational Autoencoder (classical)  
**Framework:** PyTorch  
**Location:** `pneumonia_hybrid_qml.ipynb`

**Role:**
Compresses the 768-dimensional ConvNeXt-Tiny feature vector to 64 dimensions using a VAE with KL regularization — matching the Hilbert space of a 6-qubit register (2^6 = 64). The probabilistic latent space provides smoother representations for the quantum classifier.

**Architecture:**
- Encoder: Linear(768→256) → LeakyReLU → BatchNorm → Linear(256→128) → μ, logσ
- Decoder: Linear(64→256) → LeakyReLU → BatchNorm → Linear(256→768)
- Loss: MSE reconstruction + β × KL(q(z|x) || N(0,I)), β=0.001

**Key settings:**
```python
# VAE with reparameterization trick
mu, logvar = encoder(x)
std = torch.exp(0.5 * logvar)
eps = torch.randn_like(std)
z = mu + eps * std  # reparameterization
reconstructed = decoder(z)
kl_loss = -0.5 * sum(1 + logvar - mu.pow(2) - logvar.exp())
loss = recon_loss + beta * kl_loss
```

**Outputs:**
- `vae_features_train.npy` — shape `(4185, 64)`
- `vae_features_val.npy`   — shape `(1047, 64)`  
- `vae_features_test.npy`  — shape `(624, 64)`

---

## 3. Variational Quantum Classifier (VQC)

**Type:** Parametric quantum circuit  
**Framework:** PennyLane 0.43+  
**Backend (training):** `lightning.gpu` (CUDA) on Colab A100  
**Backend (hardware eval):** `qiskit.ibm` via pennylane-qiskit  
**Location:** `02_vqc_training.ipynb`

**Role:**
The quantum classification layer. Receives a 64-dimensional L2-normalised feature vector, encodes it as quantum amplitudes, applies a trainable variational ansatz, and returns a scalar score ∈ [−1, 1] (expectation value of Pauli-Z on qubit 0).

**Inputs:**
- Batch of L2-normalised PCA vectors, shape `(B, 64)`
- Trainable parameters `θ`, shape `(n_layers, n_qubits, 3)`

**Outputs:**
- `⟨Z₀⟩` per sample ∈ [−1, 1]
- Mapped to probability: `p = (1 + ⟨Z₀⟩) / 2`

**Circuit structure (data re-uploading ansatz):**
```python
@qml.qnode(dev, diff_method="parameter-shift")
def circuit(x, params):
    for l in range(n_layers):
        qml.AmplitudeEmbedding(x, wires=range(6), normalize=True, pad_with=0.0)
        for w in range(6):
            qml.Rot(params[l, w, 0], params[l, w, 1], params[l, w, 2], wires=w)
        for w in range(6):
            qml.CNOT(wires=[w, (w + 1) % 6])
    return qml.expval(qml.PauliZ(0))
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| `n_qubits` | 6 |
| `n_layers` | 3 |
| Rotation params | 54 (3 × 6 × 3 Euler angles) |
| Scale params | 6 (learnable input encoding) |
| Measurement params | 2 (RY + RZ basis) |
| **Total trainable params** | **62** |
| Optimizer | Adam |
| Initial LR | 1e-3 |
| LR schedule | Cosine with 3-epoch warm-up |
| Loss | Weighted MSE (targets ±1) |
| Batch size | 16 |
| Max epochs | 50 |
| Early stopping patience | 10 |
| Random seed | 6 |

**Gradient method:**
`parameter-shift` — the only method compatible with real IBM hardware. Do **not** use `"best"` (autograd backprop) if you intend to transfer to hardware.

---

## 4. Classical MLP Baseline

**Type:** Multi-layer perceptron (classical)  
**Framework:** PyTorch  
**Location:** `02_vqc_training.ipynb`

**Role:**
Reference classifier operating on the same 64-dimensional PCA features as the VQC. Used for direct performance comparison under identical input conditions.

**Architecture:**
```python
nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(32, 1),
    nn.Sigmoid()
)
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Trainable params | ~2,113 |
| Loss | Weighted Binary Cross-Entropy |
| Optimizer | Adam, LR 1e-3 |
| LR schedule | `CosineAnnealingLR` |
| Batch size | 16 |
| Max epochs | 50 |
| Early stopping patience | 10 |

---

## 5. IBM Quantum Hardware Evaluator

**Type:** Remote quantum processor job  
**Provider:** IBM Quantum Cloud  
**Device:** `ibm_brisbane` (127-qubit Eagle R3) or `ibm_sherbrooke`  
**Framework:** Qiskit IBM Runtime + pennylane-qiskit  
**Location:** `04_ibm_hardware_eval.ipynb`

**Role:**
Runs inference-only on real quantum hardware using trained VQC parameters from step 3. Evaluates a subset of the test set (~50–100 samples) to quantify the gap between ideal simulation and physical noise.

**Workflow:**
1. Load optimal `params.npy` from training
2. Transpile PennyLane circuit to IBM native gate set (`ECR`, `RZ`, `SX`, `X`) using Qiskit transpiler (optimisation level 3)
3. Select 6 connected low-error qubits from the device calibration map
4. Submit via `SamplerV2` primitive inside a Qiskit Runtime session
5. Collect bitstring counts, compute ⟨Z₀⟩ expectation values

**Key settings:**
```python
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2, Session

service = QiskitRuntimeService()
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=6)

with Session(backend=backend) as session:
    sampler = SamplerV2(mode=session)
    job = sampler.run([transpiled_circuit], shots=1024)
```

**Notes:**
Train on simulator. Evaluate on hardware. Do not run the full training loop on IBM — queue times make it infeasible for 50 epochs.

---

## 6. Mitiq ZNE Error Mitigator

**Type:** Classical post-processing agent  
**Framework:** Mitiq 0.38+  
**Location:** `04_ibm_hardware_eval.ipynb`

**Role:**
Applies Zero-Noise Extrapolation (ZNE) to hardware results. Runs each circuit at noise scale factors [1×, 2×, 3×] by gate-folding, then extrapolates to the zero-noise limit using Richardson extrapolation. Significantly improves raw hardware accuracy.

**Inputs:**
- Transpiled quantum circuit
- IBM backend + Sampler executor
- Noise scale factors: `[1, 2, 3]`

**Outputs:**
- Mitigated expectation value per sample
- Comparison table: ideal sim → noisy sim → raw hardware → ZNE-mitigated hardware

**Key settings:**
```python
import mitiq

def ibm_executor(circuit):
    # wraps SamplerV2 call, returns float expectation value
    ...

mitigated = mitiq.zne.execute_with_zne(
    circuit=circuit,
    executor=ibm_executor,
    factory=mitiq.zne.RichardsonFactory(scale_factors=[1, 2, 3])
)
```

---

## 7. Noise Simulation Agent

**Type:** Software simulation  
**Framework:** PennyLane + `default.mixed` backend  
**Location:** `02_vqc_training.ipynb` — ablation section

**Role:**
Simulates the effect of hardware noise by injecting depolarising channels after every gate. Used to produce the noise degradation curve without consuming IBM queue time.

**Noise model:**
```python
dev_noisy = qml.device("default.mixed", wires=6)

@qml.qnode(dev_noisy)
def noisy_circuit(x, params, p_noise):
    for l in range(n_layers):
        qml.AmplitudeEmbedding(x, wires=range(6), normalize=True)
        for w in range(6):
            qml.Rot(params[l,w,0], params[l,w,1], params[l,w,2], wires=w)
            qml.DepolarizingChannel(p_noise, wires=w)
        for w in range(6):
            qml.CNOT(wires=[w, (w+1) % 6])
    return qml.expval(qml.PauliZ(0))
```

**Sweep values:** `p_noise ∈ {0, 0.001, 0.005, 0.01, 0.02, 0.05}`  
**Expected output:** AUC vs noise level curve (Figure for thesis Section 5.x)

---

## 8. Threshold Optimiser

**Type:** Post-processing script (classical)  
**Framework:** NumPy + scikit-learn  
**Location:** `03_evaluation.ipynb`

**Role:**
Selects the optimal classification threshold τ by maximising balanced accuracy on the **validation set only**. The chosen τ is then applied once to the test set. Prevents the data leakage present in the original version (which scanned thresholds on test data).

**Inputs:**
- `val_probs` — VQC probabilities on validation set
- `y_val` — ground truth labels

**Outputs:**
- `best_tau` — scalar threshold
- Threshold sensitivity table (Acc / Sens / Spec / BalAcc for τ ∈ {0.40, …, 0.70})

```python
from sklearn.metrics import balanced_accuracy_score

best_tau, best_score = 0.5, 0.0
for t in np.arange(0.40, 0.75, 0.05):
    preds = (val_probs > t).astype(int)
    score = balanced_accuracy_score(y_val, preds)
    if score > best_score:
        best_score, best_tau = score, t

# Apply ONCE to test set
test_preds = (test_probs > best_tau).astype(int)
```

---

## 9. Statistical Evaluator

**Type:** Post-processing script (classical)  
**Framework:** scikit-learn + statsmodels + scipy  
**Location:** `03_evaluation.ipynb`

**Role:**
Produces all metrics, statistical significance tests, and confidence intervals needed for the thesis results chapter.

**Metrics computed:**
- Accuracy, Precision, Recall (Sensitivity), Specificity, F1, AUC-ROC
- Balanced Accuracy
- McNemar's test (VQC vs MLP on same test set)
- Bootstrap 95% CI for AUC (1000 resamples)
- Expressibility `Expr(A)` and entanglement capability `Ent(A)` of ansatz (Sim et al. 2019)

**Key code:**
```python
from statsmodels.stats.contingency_tables import mcnemar
from sklearn.utils import resample

# McNemar's test
table = [[both_correct, mlp_right_vqc_wrong],
         [mlp_wrong_vqc_right, both_wrong]]
result = mcnemar(table, exact=True)
print(f"McNemar p-value: {result.pvalue:.4f}")

# Bootstrap CI for AUC
aucs = []
for _ in range(1000):
    idx = resample(range(len(y_test)))
    aucs.append(roc_auc_score(y_test[idx], probs[idx]))
ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
```

---

## 10. Grad-CAM Visualiser

**Type:** Interpretability agent (classical)  
**Framework:** PyTorch + `pytorch-grad-cam`  
**Location:** `03_evaluation.ipynb`

**Role:**
Generates class activation maps showing which regions of the chest X-ray the ConvNeXt-Tiny feature extractor attends to. Provides interpretability evidence for the thesis discussion section.

**Inputs:**
- ConvNeXt-Tiny model (with classification head re-attached for this analysis only)
- Sample X-ray images from test set (True Positive, False Negative, True Negative)

**Outputs:**
- Overlaid heatmap PNGs saved to `figures/gradcam/`
- Used directly in thesis Figure for Section 6.x

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

cam = GradCAM(model=cnx_full, target_layers=[cnx_full.stages[-1]])
grayscale_cam = cam(input_tensor=img_tensor)
visualisation = show_cam_on_image(img_rgb, grayscale_cam[0])
```

---

## Agent Interaction Map

```
Chest X-Ray images
        │
        ▼
[1] ConvNeXt-Tiny extractor  ──saves──▶  convnext_tiny_768_*.npy
        │
        ▼
[2] VAE dimensionality reducer  ──saves──▶  vae_features_*.npy + scaler
        │
     ┌──┴──────────────────────┐
     ▼                         ▼
[3] VQC (PennyLane)       [4] MLP baseline (PyTorch)
     │                         │
      ├─▶ [5] IBM hardware    │
     │        │               │
     │   [6] Mitiq ZNE        │
     │        │               │
     └───────┴───────────────┘
                     │
                [7] Noise sim (ablation)
                     │
                [8] Threshold optimiser  (val set only)
                     │
                [9] Statistical evaluator
                     │
               [10] Grad-CAM visualiser
                     │
                     ▼
            Figures + tables → LaTeX thesis
```

---

## Environment and Reproducibility

| Setting | Value |
|---------|-------|
| Python | 3.12.x |
| PennyLane | 0.43.2 |
| PyTorch | 2.9.0+cu126 |
| Qiskit | 1.x |
| Qiskit IBM Runtime | 0.20+ |
| Mitiq | 0.38+ |
| scikit-learn | 1.6.1 |
| Random seed | `6` (NumPy + PyTorch + PennyLane) |
| Colab GPU | NVIDIA A100 40 GB (training), T4 (extraction) |

**Seed initialisation (add to every notebook header):**
```python
import random, numpy as np, torch, pennylane as qml

SEED = 6
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

---

*End of combined agent documentation*
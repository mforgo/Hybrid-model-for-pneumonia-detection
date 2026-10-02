# Hybrid Model for Pneumonia Detection

[![GitHub stars](https://img.shields.io/github/stars/mforgo/Hybrid-ZFNet-Quantum-Neural-Network-for-Pneumonia-Detection)](https://github.com/mforgo/Hybrid-ZFNet-Quantum-Neural-Network-for-Pneumonia-Detection/stargazers)

This repository contains the implementation of a hybrid classical–quantum model for pneumonia detection from chest X‑ray images, developed as part of a secondary school research project (Středoškolská odborná činnost, SOČ).
The model combines a ConvNeXt‑Tiny feature extractor with DANN, autoencoder, and a 6‑qubit variational quantum classifier (VQC) with data re‑uploading, implemented in PennyLane.

> ⚠️ **Disclaimer:** This code is a research and educational prototype and is **not** a medical device. It must not be used for clinical decision-making.

***

## 1. Overview

The goal of this project is to explore whether a small variational quantum circuit (VQC) can act as a compact classifier on top of deep CNN features for medical imaging, specifically pneumonia detection from chest X‑ray images.
A classical baseline using a fully connected neural network on the same features is used for comparison to assess whether the hybrid quantum–classical approach can reach comparable performance with far fewer trainable parameters.

**Key ideas:**

- Use a pre‑trained **ConvNeXt‑Tiny** as a feature extractor on chest X‑ray images.
- Apply **Domain‑Adversarial Neural Network (DANN)** to learn domain‑invariant features.
- Reduce the 768‑dimensional feature vector to 64 dimensions using a **nonlinear autoencoder**.
- Encode the 64‑dimensional vector into a **6‑qubit** quantum state using **amplitude embedding** and classify with a **data re‑uploading VQC**.
- Compare the hybrid model against a purely classical MLP baseline trained on the same features.

***

## 2. Method

### 2.1 Architecture

The full pipeline consists of five stages:

1. **Image preprocessing**  
   - Resize to 224×224, convert to tensor, and normalize with standard ImageNet statistics (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]).
   - Medical‑safe augmentations: RandomRotation (±7°), RandomAffine (±5% translate), ColorJitter (brightness ±0.2, contrast ±0.2). RandAugment and horizontal flip were removed as they are harmful for X‑rays.

2. **Classical backbone – ConvNeXt‑Tiny**  
   - A pre‑trained ConvNeXt‑Tiny with its final classification head replaced by an adaptive pooling + flatten layer is used as a feature extractor.
   - Each X‑ray is mapped to a 768‑dimensional feature vector.

3. **Domain‑Adversarial Neural Network (DANN)**  
   - A Gradient Reversal Layer (GRL) is inserted between the feature extractor and a domain classifier to learn domain‑invariant features.
   - The domain classifier tries to distinguish source (train) from target (test) distribution, while the feature extractor learns to fool it.
   - This mitigates the dataset shift between train (74.2% pneumonia) and test (62.5% pneumonia).
   - Formula: $\lambda(p) = \frac{2}{1 + e^{-10p}} - 1$ where $p$ is training progress.

4. **Autoencoder for dimensionality reduction**  
   - Features are reduced from 768 to **64 dimensions** using a nonlinear autoencoder. The stage is fitted on the training embeddings only and then frozen (0 trainable parameters at classification time).
   - Encoder: Linear(768→256) → LeakyReLU → BatchNorm → Linear(256→128) → μ, logσ; decoder mirrors the encoder back to 768. Implemented as a variational autoencoder (`pipeline/vae.py`, classes `VAE` / `SupervisedVAE`) trained with MSE reconstruction + β·KL, β = 0.001; after training only the encoder is kept.
   - Why autoencoder over PCA: among linear PCA, LDA and univariate SelectKBest, the nonlinear autoencoder gave the most stable training dynamics and the best validation performance at 64 latent dimensions (LDA was numerically unstable in the high-dimension/low-sample regime).
   - The 64‑dimensional vectors are L2‑normalized to satisfy amplitude encoding constraints.

5. **Quantum classifier (VQC) with data re‑uploading**  
   - The 64 autoencoder features are **amplitude‑encoded once** via `qml.AmplitudeEmbedding` into a **6‑qubit** state (all 64 components enter the circuit this way).
   - A **data re‑uploading ansatz** with $L=3$ layers re‑encodes part of the input in each layer: every qubit $q$ applies `RY(scale_q · x_q · π)`, re‑uploading latent component $x_q$ with a learnable scale. Only the **first 6 of the 64 components** take this per‑layer re‑uploading path; the remaining 58 components reach the circuit exclusively through the single amplitude embedding (an honest limitation of the 6‑qubit re‑uploading budget, see §8.1).
   - Each layer: RY data re‑upload → Rot($\phi,\theta,\omega$) → Ring CNOT entanglers.
   - Total **62 trainable parameters** — 54 rotation angles (3 layers × 6 qubits × 3 Euler angles) + 6 learnable scale parameters for the RY data re‑uploading + 2 measurement-basis parameters (RY + RZ).
   - Expressivity analysis: KL divergence vs. Haar measure confirms $L=3$ is optimal.
   - The model measures a single Pauli‑Z expectation value and maps it to a probability of pneumonia: `p = (1 + ⟨Z₀⟩) / 2`.
   - Gradient computation uses the **adjoint** differentiation method (~100× faster than parameter‑shift on `lightning.qubit`).

### 2.2 Classical baseline (MLP)

A classical baseline uses the same **64‑dimensional autoencoder features** but replaces the VQC with a small fully connected network:

- Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→1) → Sigmoid.
- **2,113 trainable parameters**.
- Trained with weighted binary cross‑entropy (class balancing handled by the WeightedRandomSampler during feature extraction).

***

## 3. Dataset

The project uses the public **Chest X‑Ray Images (Pneumonia)** dataset (Paul Mooney, Kaggle), containing pediatric chest X‑rays from Guangzhou Women and Children's Medical Center, labeled as **Normal** or **Pneumonia**.

- Total: **5856 images** (pediatric, 1–5 years).
- Original split: `train`, `val`, `test`, but the original validation set only had **16 images**.
- For stable validation, the original train+val sets were merged and re‑split 80:20 with class stratification using a **patient‑grouped** shuffle, so that no patient contributes images to both sides of the split. The original test partition was left untouched.
  - Train: **4,243** images (74.4 % pneumonia)  
  - Validation: **989** images (73.3 % pneumonia)  
  - Test: **624** images (62.5 % pneumonia, original split, unmodified)

### 3.1 Class distribution and imbalance

The dataset is **imbalanced**, and the test partition has a materially different class prevalence from the training and validation partitions:

- Training: 74.4 % pneumonia → 25.6 % normal
- Validation: 73.3 % pneumonia → 26.7 % normal
- Test: 62.5 % pneumonia → 37.5 % normal

This **roughly 11 percentage point shift** between the training prior and the test set is treated as an explicit, quantified dataset shift — a robustness stress test rather than a concealed confound — and motivates:
1. Softened WeightedRandomSampler (`weight = 1/√count`) during training
2. Domain‑Adversarial Neural Network (DANN) for domain adaptation
3. Balanced Accuracy as the primary metric

### 3.2 Pneumonia subtypes

The thesis identified two morphologically distinct subtypes in the dataset:

- **Bacterial pneumonia:** Focal lobar consolidation — well‑defined opacity affecting specific lung lobes.
- **Viral pneumonia:** Diffuse interstitial pattern (ground‑glass opacities) — bilateral, less localized.

This variability places high demands on the feature extractor, which must recognize both subtypes as the same class.

## 4. Research Hypothesis

The thesis evaluates the following hypothesis:

> **Hybrid quantum‑classical neural network achieves comparable classification accuracy to a classical MLP on pneumonia detection from chest X‑rays, while using orders of magnitude fewer trainable parameters in the decision (classification) stage.**

The thesis further investigates whether current NISQ‑era quantum machine learning methods are sufficiently robust for real biomedical imaging data, or whether their practical application is primarily limited by hardware constraints (noise and decoherence).

## 5. Installation

The code targets **Python 3.10** (`environment.yml` pins `cpython=3.10.18`). GPU execution was developed on a CUDA 12.x workstation with NVIDIA L40 GPUs; the pipeline auto-selects the least-loaded device (`gpu_strategy: least_loaded`).

The canonical way to install dependencies is via the pinned requirements file:

```bash
bash setup_env.sh            # creates .venv and runs: pip install -r requirements.txt
```

or manually:

```bash
pip install -r requirements.txt
```

Optional GPU extras for the PennyLane GPU backend (installed automatically by `setup_env.sh` only when `nvidia-smi` is present):

```bash
pip install pennylane-lightning-gpu==0.45.0 custatevec-cu12
```

Notes on the environment:
- `environment.yml` is a **CPU-only conda environment** (`pytorch=2.7.1=cpu_mkl`, `torchvision=0.22.0=cpu_py310`) and pins `pennylane=0.37.0`, which lags `requirements.txt` (`pennylane==0.45.1`). Prefer `requirements.txt`; update `environment.yml` if you use it.
- `pipeline/config.py` requires `pyyaml`, and `pipeline/evaluate.py` requires `statsmodels` (McNemar's test). `pipeline/models.py` imports `qiskit_machine_learning` for the quantum-kernel benchmark. `pytorch-grad-cam` and `Pillow` are required by the analysis stage. All are listed in `requirements.txt`.
- The test suite (`tests/`) additionally requires `pytest`.

The PennyLane device is resolved in the order `lightning.gpu` → `lightning.qubit` → `default.qubit` (`pipeline/vqc.py`). Gradient computation uses the **adjoint** method on the lightning backends; note that `default.qubit` forces parameter-shift.

***

## 6. Running the experiments

1. **Open the project**

   - The implementation is the `pipeline/` package; run it from the repository root.
   - Stages can be run individually or end-to-end:

     ```bash
     python -m pipeline --stage data        # download + build patient-grouped splits
     python -m pipeline --stage features    # frozen ConvNeXt-Tiny feature extraction
     python -m pipeline --stage vae         # autoencoder 768 -> 64
     python -m pipeline --stage train       # MLP baseline + VQC training
     python -m pipeline --stage evaluate    # thresholds, metrics, bootstrap CIs, McNemar
     python -m pipeline --stage qpu         # IBM Quantum hardware run + ZNE (optional)
     python -m pipeline --stage analysis    # figures, diagnostics, Grad-CAM
     ```

   - `scripts/run_all.sh` chains the same stages in order. The scripts under `notebooks_archive/` are **archived legacy** material and no longer reflect the current pipeline.

2. **Install dependencies**

   - Run `bash setup_env.sh` (or `pip install -r requirements.txt`). See §5.

3. **Configure experiment settings**

   - Configuration lives in `configs/default.yaml` and is loaded into the `Config` dataclass (`pipeline/config.py`). Key defaults:
   - `project_name: "HybridConvNeXtTinyQNNPneumonia"`  
   - `device: "auto"` (select the least-loaded GPU; falls back to CPU)  
   - `split_strategy: "patient_grouped"`, `reduction_method: "vae"`, `target_dims: 64`  
   - `n_qubits: 6`, `n_layers: 3`  
   - `vqc_encoding: "amplitude"`, `diff_method: "adjoint"`  
   - `batch_size: 16`, `learning_rate: 1e-3`, `epochs: 50`, `early_stopping_patience: 3`

   - Every stage can be overridden on the command line, e.g. `--set n_layers=2 device=cpu`.

4. **Step 1 – Feature extraction (ConvNeXt‑Tiny)**

   - The pipeline downloads the Kaggle dataset via `kagglehub`, builds PyTorch `DataLoader`s and runs frozen ConvNeXt‑Tiny to extract 768‑dimensional features for each image.
   - Features and metadata are written to `artifacts/features/` and cached for reuse.

5. **Step 2 – DANN domain adaptation**

   - Features are processed through a DANN with a Gradient Reversal Layer (GRL) to suppress the train/test distribution shift.
   - The domain classifier is trained to distinguish source (train) from target (test) distributions while the feature extractor learns to fool it.
   - GRL strength increases as λ(p) = 2/(1+exp(−10p)) − 1.
   - **Caveat:** the test partition is used as the unlabeled target domain here, and the effectiveness of this stage was not quantitatively evaluated (see §8.1).

6. **Step 3 – Autoencoder training**

   - Train the nonlinear autoencoder (768→256→128→64, mirrored decoder back to 768) fitted on the training embeddings.
   - Loss: MSE reconstruction + β·KL divergence, β = 0.001. Adam optimizer.
   - After training the decoder is discarded; the frozen encoder produces the 64‑dimensional features.

7. **Step 4 – Classical baseline training**

   - Train the classical MLP baseline (64→32→1) on the same 64‑dimensional features.
   - Uses weighted BCE loss, Adam optimizer, cosine learning‑rate schedule with warm‑up, early stopping on validation loss.

8. **Step 5 – Quantum model training**

   - Build a PennyLane QNode with the data re‑uploading ansatz (L=3).
   - Train with Adam optimizer, cosine LR schedule with warm‑up, using **class‑weighted MSE** loss in label space {−1, +1}.
   - Save best parameters (`results/vqc_best_params.npy`, 62 values) and training history to `results/`.

9. **Step 6 – Statistical evaluation**

   - Threshold scan (τ ∈ [0.30, 0.80], step 0.025) to find the optimal threshold on the **validation** set by balanced accuracy; apply it once to the test set.
   - Report metrics: Accuracy, Balanced Accuracy, Precision, Recall/Sensitivity, Specificity, F1‑score, AUC‑ROC.
   - Compute bootstrap 95% CI (B=1000) for AUC‑ROC.
   - Run McNemar's exact test to compare MLP vs. VQC statistical significance.

***

## 7. Results

### 7.1 Classical vs hybrid performance

On the 624‑image test set (62.5 % pneumonia, 37.5 % normal) with the **patient‑grouped split** (4,243 train / 989 val / 624 test — no patient leakage), the following metrics were obtained:

| Metric                  | Classical (ConvNeXt‑Tiny + MLP) | Hybrid (ConvNeXt‑Tiny + VQC) | Difference |
|-------------------------|--------------------------------|------------------------------|-----------|
| Accuracy                | 74.20 %                        | 74.84 %                      | +0.64 %   |
| Balanced accuracy       | 0.7295                         | 0.7466                       | +1.71 pp  |
| Precision               | 80.21 %                        | 82.82 %                      | +2.61 pp  |
| Recall (Sensitivity)    | 77.95 %                        | 75.38 %                      | −2.57 pp  |
| Specificity             | 67.95 %                        | 73.93 %                      | +5.98 pp  |
| F1‑score                | 0.7906                         | 0.7893                       | −0.0013   |
| AUC‑ROC (test)          | 0.8476                         | 0.8055                       | −0.0421   |
| Decision threshold τ\*   | 0.525                          | 0.800                        | —         |
| Trainable params (classifier) | 2,113                     | **62**                       | ≈34×      |

Both models consume identical features from an identical, patient‑disjoint split, so the comparison isolates the effect of the decision layer. Thresholds τ\* were selected on the validation set by balanced‑accuracy maximization and applied **once** to the test set.

**Training cost.** The VQC takes ~2.7 h to train (46 epochs on `lightning.qubit` with adjoint differentiation); the MLP takes ~11 s (50 epochs on GPU) — an ~860× difference reflecting quantum‑simulator overhead. The two layers cannot share a threshold: ⟨Z₀⟩ is remapped linearly onto [0,1] and is not calibrated against the sigmoid MLP, so each is calibrated separately on the validation set.

The hybrid model reaches **AUC 0.806 on the test set** (MLP: 0.848) with **≈34× fewer trainable decision‑layer parameters** (62 vs. 2,113). The VQC attains the higher thresholded accuracy, balanced accuracy, precision and specificity, at the cost of lower sensitivity, marginally lower F1, and a *lower* threshold‑independent ranking quality (AUC‑ROC). The two models are **statistically indistinguishable** (§7.2); the correct conclusion is **comparable performance, not superiority** — the MLP ranks the test set better, and the VQC's advantage is confined to the thresholded operating point selected by the balanced‑accuracy criterion. The notable positive result is that this operating point is reached with ≈34× fewer parameters.

### 7.2 Statistical evaluation

1. **McNemar's test (exact)** on the 624 paired test predictions: MLP vs VQC → **p = 0.71** — not significant at α = 0.05.
2. **Bootstrap 95% confidence intervals (B = 1000, SEED = 6)**, non‑parametric resampling of the held‑out test set, for AUC‑ROC: MLP **[0.818, 0.877]**, VQC **[0.766, 0.844]**. The intervals overlap.

The accuracy difference is therefore not significant, and we explicitly do **not** claim the hybrid model is superior. Both analyses operate on the thresholded operating point; AUC‑ROC is reported alongside because a single threshold can misrepresent two models whose output scales are not comparable.

**Single‑run caveat.** These figures come from one training run on one split, without cross‑validation. This is a deliberate trade‑off — VQC training cost makes k‑fold retraining expensive — and it is a genuine limitation on the strength of the comparison (§8.1).

### 7.3 Ansatz depth selection

The data re‑uploading ansatz depth L = 3 was selected empirically, balancing expressibility against NISQ feasibility. Expressibility `Expr(A)` (KL divergence vs. the Haar measure) and entanglement capability `Ent(A)` (Meyer–Wallach) were swept over L ∈ {1, 2, 3, 4}.

> **Note (provisional):** the sweep CSV (`results/expressibility_sweep.csv`) is not committed, so the figures below are carried over from the thesis tables and should be regenerated before publication. The qualitative conclusion — monotone improvement in `Expr(A)` with depth, with diminishing returns beyond L = 3 — is what motivates the choice.

| L | Rot params | Expr(A) ↓ | Ent(A) ↑ |
|---|-----------|-----------|----------|
| 1 | 18  | 2.842 | 0.312 |
| 2 | 36  | 1.456 | 0.478 |
| 3 | 54  | 0.923 | 0.621 |
| 4 | 72  | 0.847 | 0.689 |

- L = 1, 2 are under‑parameterized; `Expr(A)` remains far from the Haar limit.
- L ≥ 4 yields only marginal additional expressibility while increasing two‑qubit gate count and noise susceptibility.
- **Honest reading:** `Ent(A) = 0.621 < 1` means this ansatz *cannot* reach maximally entangled states — expressibility and entanglement capacity here are measured against the full unitary group on 2⁶ = 64 dimensions, which no shallow ansatz approaches. The claim is therefore **sufficiency at this depth**, not maximal expressibility; the evidence for sufficiency is the held‑out test performance in §7.1.

Adding the 6 learnable re‑upload scales and the 2 measurement‑basis parameters gives the final **62 trainable parameters** (54 rotation + 6 scale + 2 measurement).

### 7.4 Behaviour and interpretation

- The VQC achieves balanced **sensitivity (75.4 %) / specificity (73.9 %)** at τ=0.80; the MLP trades off differently: **sensitivity (78.0 %) / specificity (68.0 %)** at τ=0.525. The VQC is more conservative (fewer false positives, more missed cases).
- Test AUC: VQC **0.806**, MLP **0.848**, indicating the models generalise well to the balanced test set.
- A **dataset shift** (pneumonia: 74.2 % → 62.5 %) between train and test distributions exists. DANN partially mitigates but does not eliminate this shift.
- The VQC trains on CPU (46 epochs, 9,589 s on lightning.qubit with adjoint differentiation), while the MLP trains on GPU (50 epochs, 11 s) — an ~860× time cost reflecting the quantum simulator overhead.

***

## 8. Limitations & future work

### 8.1 Key limitations

The thesis acknowledges the following limitations:

1. **Training constraints**: VQC training on the ideal simulator takes ~2.7 h per run (46 epochs with patience = 10). Limited to 6 qubits (62 parameters), the re‑uploading path covers only 6 of 64 latent dimensions per layer — the remaining 58 components enter exclusively via amplitude embedding.
2. **Single dataset**: Only one public pediatric dataset from one institution (Guangzhou Women and Children's Medical Center). Generalization to other populations, hospitals, or acquisition protocols is unknown.
3. **Hardware evaluation limited by noise and a degenerate slice**: A real‑hardware run was performed on IBM Quantum (`ibm_miami`, 79 jobs, September 2026) but evaluated a label‑sorted all‑negative test slice (now fixed via `select_qpu_subset` with stratified, class‑balanced sampling). The raw hardware output was noise‑saturated: the sim‑to‑QPU probability span collapsed ~52%, yielding near‑coin‑flip accuracy. The run therefore provides no reliable on‑device accuracy estimate and is reported only as a supporting data point in the poster.
4. **Dataset shift**: The 11.7 percentage point shift (74.2% → 62.5% pneumonia) between train and test is significant. DANN partially mitigates but does not eliminate this.
5. **Autoencoder information loss**: While nonlinear autoencoder preserves more information than linear PCA, dimension reduction from 768→64 still discards some discriminative signal.
6. **ViT incompatibility**: Initial experiments showed ViT‑B/16 features were incompatible with the VQC (failed to learn, AUC ≈ 0.46). Only ConvNeXt‑Tiny features worked.
7. **Barren plateaus risk**: Deeper ansatzes (L>3) risk vanishing gradients; L=3 was carefully selected as optimal.

### 8.2 Future work directions

- Run the VQC on real IBM Quantum hardware with Zero‑Noise Extrapolation (ZNE) and Probabilistic Error Cancellation (PEC).
- Explore quantum kernels or quanvolutional layers as alternatives.
- Use multi‑institutional datasets (e.g., NIH ChestX‑ray14) for domain generalization studies.
- Investigate alternative encoding schemes (displacement, controlled‑displacement).
- Implement ensemble models combining classical MLP + VQC for hybrid decision‑making.

***

## 9. Citation

If you use this code, please cite:

```bibtex
@mastersthesis{forgo2026hybrid,
  author    = {Michal Forgó},
  title     = {Hybridní model pro detekci pneumonie / Hybrid Model for Pneumonia Detection},
  school    = {Středoškolská odborná činnost (SOČ)},
  year      = {2026},
  url       = {https://github.com/mforgo/Hybrid-ZFNet-Quantum-Neural-Network-for-Pneumonia-Detection},
  note      = {Czech secondary school research project}
}
```

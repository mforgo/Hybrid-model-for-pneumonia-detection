# Hybrid Model for Pneumonia Detection

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
   - Features are reduced from 768 to **64 dimensions** using a nonlinear autoencoder (vs. linear PCA).
   - Architecture: Linear(768→256) → LeakyReLU → BatchNorm → Dropout → Linear(256→64).
   - Why autoencoder over PCA: captures nonlinear correlations in CNN features, preserves discriminative signal better for classification.
   - The 64‑dimensional vectors are L2‑normalized to satisfy amplitude encoding constraints.

5. **Quantum classifier (VQC) with data re‑uploading**  
   - The 64 autoencoder features are **amplitude‑encoded** via `qml.AmplitudeEmbedding` into a **6‑qubit** state.
   - A **data re‑uploading ansatz** with $L=3$ layers re‑encodes input data in each layer for increased expressivity.
   - Each layer: AngleEmbedding → Rot($\phi,\theta,\omega$) → Ring CNOT entanglers.
   - Total **54 trainable parameters** (3 layers × 6 qubits × 3 Euler angles).
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
- For stable validation, the original train+val sets were merged and re‑split 80/20 with stratification:
  - Train: **4185** images  
  - Validation: **1047** images  
  - Test: **624** images (original test set, kept untouched)

### 3.1 Class distribution and imbalance

The dataset is **imbalanced**, with roughly **74.2 % pneumonia** in train/val and **62.5 % pneumonia** in the test set:

- Training/validation: 74.2% pneumonia → 25.8% normal (ratio ~2.9:1)
- Test: 62.5% pneumonia → 37.5% normal

This **11.7 percentage point shift** between train and test distributions is a significant domain shift that motivates:
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

The code is designed to run in **Python 3.12** on **Google Colab** with GPU acceleration (T4 or A100 were used in the experiments).

Install the main dependencies (on Colab this is done at the top of the notebook):

```bash
pip install \
  pennylane==0.44.1 pennylane-qiskit==0.44.1 pennylane-lightning-gpu==0.44.0 custatevec-cu12 \
  torch torchvision \
  qiskit qiskit-ibm-runtime qiskit-machine-learning==0.9.0 \
  mitiq \
  grad-cam \
  scikit-learn scikit-image \
  numpy scipy pandas matplotlib seaborn \
  kagglehub joblib optuna
```

The notebook uses PennyLane's `lightning.qubit` backend (falls back from `lightning.gpu` if cuQuantum is unavailable). Gradient computation uses the **adjoint** method.

***

## 6. Running the experiments

1. **Open the notebook**

   - Upload `pneumonia_hybrid_qml.ipynb` to Google Colab or open it directly from your GitHub repo.
   - Enable a **GPU runtime** in Colab (e.g. T4 or A100).

2. **Install dependencies**

   - Run the first cell that calls `pip install ...` to install all required packages.

3. **Configure experiment settings**

   - The configuration is defined via a `dataclass` `ExperimentConfig`, with defaults such as:
   - `project_name = "HybridConvNeXtTinyQNNPneumonia"`  
   - `device = "cuda"` (if available)  
   - `reduction_method = "autoencoder"`, `target_dims = 64`  
   - `n_qubits = 6`, `n_layers = 3`  
   - `encoding_method = "amplitude"`, `diff_method = "adjoint"`  
   - `batch_size = 16`, `learning_rate = 1e-3`, `epochs = 50`, `early_stopping_patience = 3`

4. **Step 1 – Feature extraction (ConvNeXt‑Tiny)**

   - The notebook downloads the Kaggle dataset via `kagglehub`, builds PyTorch `DataLoader`s and runs ConvNeXt‑Tiny to extract 768‑dimensional features for each image.
   - Features and metadata are saved into the `artifacts/features/` directory for reuse.

5. **Step 2 – DANN domain adaptation**

   - Features are processed through DANN to learn domain‑invariant representations.
   - The model trains with source (train+val) labeled data and unlabeled target (test) data.
   - GRL strength increases as λ(p) = 2/(1+exp(−10p)) − 1.

6. **Step 3 – Autoencoder training**

   - Train the nonlinear autoencoder (768→256→64→256→768) to reduce dimensionality.
   - Use MSE reconstruction loss, Adam optimizer, BatchNorm, Dropout(0.2).
   - After training, discard decoder — encoder produces 64‑dim features.

7. **Step 4 – Classical baseline training**

   - Train the classical MLP baseline (64→32→1) on the 64‑dimensional features.
   - Uses weighted BCE loss, Adam optimizer, cosine learning‑rate schedule with warm‑up, early stopping on validation loss.

8. **Step 5 – Quantum model training**

   - Build a PennyLane QNode with `lightning.qubit` device and data re‑uploading ansatz (L=3).
   - Train with Adam optimizer, cosine LR schedule with warm‑up, using **weighted MSE** loss in label space {−1, +1}.
   - Save best parameters and training history to `results/`.

9. **Step 6 – Statistical evaluation**

   - Threshold scan (0.30–0.80, step 0.025) to find optimal threshold on validation set.
   - Report metrics: Accuracy, Precision, Recall/Sensitivity, Specificity, F1‑score.
   - Compute bootstrap 95% CI (B=1000) for all metrics.
   - Run McNemar's test to compare MLP vs. VQC statistical significance.

***

## 7. Results

### 7.1 Classical vs hybrid performance

On the 624‑image test set (62.5 % pneumonia, 37.5 % normal), the following metrics were obtained:

| Metric             | Classical (ConvNeXt‑Tiny + MLP) | Hybrid (ConvNeXt‑Tiny + VQC) | Difference |
|--------------------|--------------------------------|------------------------------|-----------|
| Accuracy           | 82.53 %                        | 81.25 %                      | −1.28 %   |
| Precision          | 78.27 %                        | 77.03 %                      | −1.24 %   |
| Recall (Sensitivity) | 99.74 %                      | 99.74 %                      | 0.00 %    |
| Specificity        | 53.85 %                        | 50.43 %                      | −3.42 %   |
| F1‑score           | 0.8771                         | 0.8693                       | −0.0078   |
| AUC‑ROC (val)      | 0.991                          | **0.969**                    | —         |
| AUC‑ROC (test)     | 0.940                          | 0.860                        | −0.080    |
| Trainable params (classifier) | 2,113                | **54**                       | −2,059    |
| Training time      | ~7 epochs (early stop)        | 7 epochs (early stop)         | —         |

**Optimal thresholds** (selected on validation set using Balanced Accuracy):
- MLP: τ = 0.30
- VQC: τ = 0.35

The hybrid model achieves **96.9 % AUC on validation** and **86.0 % AUC on test**, with **39× fewer trainable parameters** (54 vs. 2,113), which demonstrates extreme parameter efficiency in the NISQ regime.

### 7.2 Statistical evaluation

To rigorously compare the models, the following statistical analyses were performed:

1. **Bootstrap 95% Confidence Intervals (B=1000)**: Non‑parametric resampling to estimate uncertainty in AUC and Accuracy.
2. **McNemar's Test**: Paired statistical test for comparing two classifiers on the same test set. Focuses on samples where predictions differ.
3. **5‑Fold Stratified Cross‑Validation**: Additional validation on train+val data to ensure robustness.

Results show the accuracy difference (1.28 percentage points) is **not statistically significant** at α=0.05 level via McNemar's test, validating the hypothesis that VQC achieves comparable performance.

### 7.3 Expressivity and entangling capability

The data re‑uploading ansatz with L=3 was selected based on:
- **Expressivity analysis**: KL divergence vs. Haar measure shows L=3 captures sufficient state diversity without barren plateaus.
- L=1,2: too "rigid" (under‑parameterized)
- L=4+: marginal expressivity gain, higher hardware noise susceptibility

This justifies the choice of **54 parameters** as the optimal balance between expressivity and NISQ feasibility.

### 7.4 Behaviour and interpretation

- The hybrid model shows very high **sensitivity (99.7 %)** with moderate specificity (50.4 %), meaning it correctly identifies nearly all pneumonia cases while having a higher false positive rate than the classical baseline.
- The gap between validation AUC (96.9 %) and test AUC (86.0 %) suggests some overfitting and domain shift.
- A noticeable **dataset shift** (pneumonia: 74.2% → 62.5%) between train and test distributions contributes to the performance gap.
- The VQC training stopped early at **epoch 7** (patience = 3), indicating the model converged quickly but did not generalize as well as the MLP.

***

## 8. Limitations & future work

### 8.1 Key limitations

The thesis acknowledges the following limitations:

1. **Training constraints**: VQC converged after only 7 epochs with early stopping (patience = 3), limiting achievable performance on the simulator.
2. **Single dataset**: Only one public pediatric dataset from one institution (Guangzhou Women and Children's Medical Center). Generalization to other populations, hospitals, or acquisition protocols is unknown.
3. **No hardware testing**: The quantum model was trained on an **ideal simulator** without realistic NISQ noise. Performance on real quantum hardware would likely be worse without explicit error‑mitigation (ZNE, PEC).
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

If you use this code or ideas from the project, please cite the associated SOČ thesis (Czech, English annotation in the front matter):

> M. Forgó, *Hybridní model pro detekci pneumonie / Hybrid Model for Pneumonia Detection*, Středoškolská odborná činnost (SOČ), 2026.

You can also link to this repository and mention that it implements a hybrid ConvNeXt‑Tiny + VQC pipeline for pneumonia detection using PennyLane and PyTorch.

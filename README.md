# Hybrid Model for Pneumonia Detection

This repository contains the implementation of a hybrid classical–quantum model for pneumonia detection from chest X‑ray images, developed as part of a secondary school research project (Středoškolská odborná činnost, SOČ).
The model combines a ConvNeXt‑Tiny feature extractor with PCA and a 6‑qubit variational quantum classifier implemented in PennyLane.

> ⚠️ **Disclaimer:** This code is a research and educational prototype and is **not** a medical device. It must not be used for clinical decision-making.

***

## 1. Overview

The goal of this project is to explore whether a small variational quantum circuit (VQC) can act as a compact classifier on top of deep CNN features for medical imaging, specifically pneumonia detection from chest X‑ray images.
A classical baseline using a fully connected neural network on the same features is used for comparison to assess whether the hybrid quantum–classical approach can reach comparable performance with far fewer trainable parameters.

**Key ideas:**

- Use a pre‑trained **ConvNeXt‑Tiny** as a feature extractor on chest X‑ray images.
- Reduce the 768‑dimensional feature vector to 64 dimensions using **PCA**.
- **Angle‑encode** the 64‑dimensional vector (first 6 components) into a **6‑qubit** quantum state and classify with a **variational quantum circuit**.
- Compare the hybrid model against a purely classical baseline trained on the same PCA features.

***

## 2. Method

### 2.1 Architecture

The full pipeline consists of four stages:

1. **Image preprocessing**  
   - Resize to 224×224, convert to tensor, and normalize with standard ImageNet statistics (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]).
   - Medical-safe augmentations: RandomRotation (±7°), RandomAffine (±5% translate), ColorJitter (brightness ±0.2, contrast ±0.2). RandAugment and horizontal flip were removed as they are harmful for X‑rays.

2. **Classical backbone – ConvNeXt‑Tiny**  
   - A pre‑trained ConvNeXt‑Tiny with its final classification head replaced by an adaptive pooling + flatten layer is used as a feature extractor.
   - Each X‑ray is mapped to a 768‑dimensional feature vector.

3. **Dimensionality reduction & normalization**  
   - Features are standardized and reduced to **64 principal components** using PCA (explained variance at 64 PCs: **0.749** for ConvNeXt‑Tiny).
   - The 64‑dimensional vectors are L2‑normalized to satisfy angle encoding constraints.

4. **Quantum classifier (VQC)**  
   - The first 6 PCA components are **angle‑encoded** via `qml.AngleEmbedding(rotation='Y')` into a **6‑qubit** state.
   - A variational circuit with 3 layers of single‑qubit rotations (`qml.Rot`) and ring‑style CNOT entanglers is applied.
   - The model measures a single Pauli‑Z expectation value and maps it to a probability of pneumonia: `p = (1 + ⟨Z₀⟩) / 2`.
   - Gradient computation uses the **adjoint** differentiation method (~100× faster than parameter-shift on `lightning.qubit`).

### 2.2 Classical baseline

A classical baseline uses the same **64‑dimensional PCA features** but replaces the VQC with a small fully connected network:

- Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→1) → Sigmoid.
- **2,113 trainable parameters**.
- Trained with unweighted binary cross‑entropy (class balancing handled by the WeightedRandomSampler during feature extraction).

***

## 3. Dataset

The project uses the public **Chest X‑Ray Images (Pneumonia)** dataset (Paul Mooney, Kaggle), containing pediatric chest X‑rays labeled as **Normal** or **Pneumonia**.

- Total: **5856 images**.
- Original split: `train`, `val`, `test`, but the original validation set only had **16 images**.
- For stable validation, the original train+val sets were merged and re‑split 80/20 with stratification:
  - Train: **4185** images  
  - Validation: **1047** images  
  - Test: **624** images (original test set, kept untouched)

The dataset is **imbalanced**, with roughly **74.2 % pneumonia** in train/val and **62.5 % pneumonia** in the test set, which motivates the use of a softened WeightedRandomSampler (`weight = 1/√count`) and balanced accuracy when evaluating.

The notebook automatically downloads the dataset via `kagglehub` when run in Google Colab.

## 4. Installation

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

## 5. Running the experiments

1. **Open the notebook**

   - Upload `pneumonia_hybrid_qml.ipynb` to Google Colab or open it directly from your GitHub repo.
   - Enable a **GPU runtime** in Colab (e.g. T4 or A100).

2. **Install dependencies**

   - Run the first cell that calls `pip install ...` to install all required packages.

3. **Configure experiment settings**

   - The configuration is defined via a `dataclass` `ExperimentConfig`, with defaults such as:
   - `project_name = "HybridConvNeXtTinyQNNPneumonia"`  
   - `device = "cuda"` (if available)  
   - `reduction_method = "pca"`, `target_dims = 64`  
   - `n_qubits = 6`, `n_layers = 3`  
   - `encoding_method = "angle"`, `diff_method = "adjoint"`  
   - `batch_size = 16`, `learning_rate = 1e-3`, `epochs = 50`, `early_stopping_patience = 3`

4. **Step 1 – Feature extraction (ConvNeXt‑Tiny)**

   - The notebook downloads the Kaggle dataset via `kagglehub`, builds PyTorch `DataLoader`s and runs ConvNeXt‑Tiny to extract 768‑dimensional features for each image.
   - Features and metadata are saved into the `artifacts/features/` directory for reuse.

5. **Step 2 – Classical preprocessing (PCA pipeline)**

   - The script merges the tiny original validation set with train, re‑splits 80/20 (stratified), and then fits a `StandardScaler + PCA(64)` pipeline.
   - Transformed `X_train`, `X_val`, `X_test` and labels are stored as `.npy` files and the fitted preprocessing pipeline is saved with `joblib`.

6. **Step 3 – Classical baseline training**

   - Train the classical MLP baseline on the 64‑dimensional features.
   - Uses unweighted BCE loss, Adam optimizer, cosine learning‑rate schedule and early stopping based on validation loss.

7. **Step 4 – Quantum model training**

   - Builds a PennyLane QNode with the `lightning.qubit` device and the custom ansatz (AngleEmbedding + Rot + ring CNOT layers).
   - Trains parameters with Adam and a cosine annealing learning‑rate schedule with warm‑up, using an **unweighted MSE** loss in the label space {−1, +1} (class balancing handled by the softened WeightedRandomSampler).
   - Saves the best parameters and training history to `results/`.

8. **Step 5 – Threshold scan & evaluation**

   - On the test set, the model outputs continuous probabilities; a threshold scan over values 0.30–0.80 (step 0.025) is used to study the trade‑off between sensitivity and specificity and to compute balanced accuracy.

***

## 6. Results

### 6.1 Classical vs hybrid performance

On the 624‑image test set (62.5 % pneumonia, 37.5 % normal), the following metrics were obtained:

| Metric             | Classical (ConvNeXt‑Tiny + MLP) | Hybrid (ConvNeXt‑Tiny + VQC) | Difference |
|--------------------|--------------------------------|------------------------------|-----------|
| Accuracy           | 86.54 %                        | 75.80 %                      | −10.74 %  |
| Precision          | 83.26 %                        | 77.35 %                      | −5.91 %   |
| Recall (Sensitivity) | 98.21 %                      | 86.67 %                      | −11.54 %  |
| Specificity        | 67.09 %                        | 57.69 %                      | −9.40 %   |
| F1‑score           | 0.9012                         | 0.8174                       | −0.0838   |
| AUC‑ROC (val)      | 0.9907                         | **0.9688**                   | —         |
| AUC‑ROC (test)     | 0.9530                         | 0.8600                       | −0.0930   |
| Trainable params (classifier) | 2,113                | 54                           | −2,059    |
| Training time      | ~16 epochs (early stop)        | 7 epochs (early stop)        | —         |

The hybrid model achieves a strong **96.88 % AUC on the validation set** and a respectable 86.00 % on the held‑out test set, with **39× fewer trainable parameters** in the decision stage (54 vs. 2,113), which is significant in the NISQ regime.

### 6.2 Behaviour and interpretation

- The hybrid model shows relatively high **sensitivity (86.7 %)** with moderate specificity (57.7 %), meaning it tends to correctly identify pneumonia cases while maintaining a higher false positive rate than the classical baseline.
- The gap between validation AUC (96.88 %) and test AUC (86.00 %) suggests some overfitting, which is expected given the small quantum model capacity and limited training data.
- A noticeable **domain shift** between train/validation and test label distributions (pneumonia share drops from ~74.2 % to 62.5 %) likely contributes to the performance gap.
- The VQC training stopped early at **epoch 7** (patience = 3), indicating the model converged quickly but did not generalize as well as the MLP.

***

## 7. Limitations & future work

Some key limitations identified in the thesis:

- Training the VQC on a classical GPU simulator is expensive; the model converged after only 7 epochs with early stopping (patience = 3), limiting the achievable performance.
- Only a single public pediatric dataset from one institution was used, so generalization to other populations, hospitals or acquisition protocols is unknown.
- The 768→64 PCA reduction retains only **74.9 %** of the explained variance (ConvNeXt‑Tiny), which may discard clinically relevant information and caps the achievable performance of the quantum classifier.
- The quantum model was trained on an **ideal simulator** without realistic noise; performance on real NISQ hardware would likely be worse without explicit error‑mitigation techniques.
- The gap between validation AUC (96.88 %) and test AUC (86.00 %) indicates overfitting, suggesting the need for stronger regularization or more training data.
- The VQC uses **angle encoding** (only 6 of 64 PCA components), which discards 90.6 % of the available PCA features. Amplitude encoding was explored but angle encoding was chosen for gradient efficiency.

Future directions suggested by the work include:

- Running the VQC on real hardware (e.g. IBM Quantum) with noise mitigation.  
- Exploring alternative quantum architectures (e.g. quantum kernels, quanvolutional layers, ensembles with classical models).  
- Using richer, multi‑institutional datasets and domain‑adaptation techniques to address dataset shift.
- Investigating amplitude encoding to utilize all 64 PCA components instead of just 6.

***

## 8. Citation

If you use this code or ideas from the project, please cite the associated SOČ thesis (Czech, English annotation in the front matter):

> M. Forg, *Hybridní model pro detekci pneumonie / Hybrid Model for Pneumonia Detection*, Středoškolská odborná činnost (SOČ), 2026.

You can also link to this repository and mention that it implements a hybrid ConvNeXt‑Tiny + VQC pipeline for pneumonia detection using PennyLane and PyTorch.

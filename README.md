# Hybrid Model for Pneumonia Detection

This repository contains the implementation of a hybrid classical–quantum model for pneumonia detection from chest X‑ray images, developed as part of a secondary school research project (Středoškolská odborná činnost, SOČ).
The model combines a ResNet‑50 feature extractor with PCA and a 6‑qubit variational quantum classifier implemented in PennyLane.

> ⚠️ **Disclaimer:** This code is a research and educational prototype and is **not** a medical device. It must not be used for clinical decision-making.

***

## 1. Overview

The goal of this project is to explore whether a small variational quantum circuit (VQC) can act as a compact classifier on top of deep CNN features for medical imaging, specifically pneumonia detection from chest X‑ray images.
A classical baseline using a fully connected neural network on the same features is used for comparison to assess whether the hybrid quantum–classical approach can reach comparable performance with far fewer trainable parameters.

**Key ideas:**

- Use a pre‑trained **ViT‑B/16** as a feature extractor on chest X‑ray images.
- Reduce the 2048‑dimensional feature vector to 64 dimensions using **PCA**.
- Amplitude‑encode the 64‑dimensional vector into a **6‑qubit** quantum state and classify with a **variational quantum circuit**.
- Compare the hybrid model against a purely classical baseline trained on the same PCA features.

***

## 2. Method

### 2.1 Architecture

The full pipeline consists of four stages:

1. **Image preprocessing**  
   - Resize to 224×224, convert to tensor, and normalize with standard ImageNet statistics (mean [0.485, 0.456, 0.406], std [0.229, 0.224, 0.225]).

2. **Classical backbone – ViT‑B/16**  
   - A pre‑trained ViT‑B/16 with its final classification head replaced by an identity layer is used as a feature extractor.
   - Each X‑ray is mapped to a 768‑dimensional feature vector.

3. **Dimensionality reduction & normalization**  
   - Features are standardized and reduced to **64 principal components** using PCA (chosen after comparing PCA, LDA, and SelectKBest).
   - The 64‑dimensional vectors are L2‑normalized to satisfy amplitude encoding constraints.

4. **Quantum classifier (VQC)**  
   - The 64‑dimensional vector is **amplitude‑encoded** into a 6‑qubit state (Hilbert space dimension 64).
   - A variational circuit with repeated layers of single‑qubit rotations (`qml.Rot`) and ring‑style CNOT entanglers is applied.
   - The model measures a single Pauli‑Z expectation value and maps it to a probability of pneumonia, followed by a threshold scan to pick an operating point.

### 2.2 Classical baseline

A classical baseline uses the same **64‑dimensional PCA features** but replaces the VQC with a small fully connected network:

- Linear → ReLU → Dropout → Linear → ReLU → Dropout → Linear → Sigmoid.
- Trained with weighted binary cross‑entropy to handle class imbalance.

***

## 3. Dataset

The project uses the public **Chest X‑Ray Images (Pneumonia)** dataset (Paul Mooney, Kaggle), containing pediatric chest X‑rays labeled as **Normal** or **Pneumonia**.

- Total: **5856 images**.
- Original split: `train`, `val`, `test`, but the original validation set only had **16 images**.
- For stable validation, the original train+val sets were merged and re‑split 80/20 with stratification:
  - Train: **4185** images  
  - Validation: **1047** images  
  - Test: **624** images (original test set, kept untouched)

The dataset is **imbalanced**, with roughly **74.2 % pneumonia** in train/val and **62.5 % pneumonia** in the test set, which motivates class‑weighted loss functions and using balanced accuracy when evaluating.

The notebook automatically downloads the dataset via `kagglehub` when run in Google Colab.

## 4. Installation

The code is designed to run in **Python 3.12** on **Google Colab** with GPU acceleration (T4 or A100 were used in the experiments).

Install the main dependencies (on Colab this is done at the top of the notebook):

```bash
pip install \
  pennylane pennylane-qiskit pennylane-lightning-gpu \
  torch torchvision \
  scikit-learn scikit-image \
  numpy scipy pandas matplotlib seaborn \
  opencv-python kagglehub optuna
```

The notebook uses PennyLane’s GPU‑accelerated backends (via `lightning.gpu` and NVIDIA cuQuantum) when available; otherwise it falls back to `default.qubit`.

***

## 5. Running the experiments

1. **Open the notebook**

   - Upload `pneumonia_qml.ipynb` to Google Colab or open it directly from your GitHub repo.
   - Enable a **GPU runtime** in Colab (e.g., T4 or A100).

2. **Install dependencies**

   - Run the first cell that calls `pip install ...` to install all required packages.

3. **Configure experiment settings**

   - The configuration is defined via a `dataclass` `ExperimentConfig`, with defaults such as:
     - `project_name = "HybridResNet50QNNPneumonia"`  
     - `device = "cuda"` (if available)  
     - `reduction_method = "pca"`, `target_dims = 64`  
     - `n_qubits = ceil(log2(64)) = 6`, `n_layers = 6`  
     - `encoding_method = "amplitude"`  
     - `batch_size = 16`, `learning_rate = 1e-3`, `epochs = 50`  

4. **Step 1 – Feature extraction (ViT‑B/16)**

   - The notebook downloads the Kaggle dataset via `kagglehub`, builds PyTorch `DataLoader`s and runs ViT‑B/16 to extract 768‑dimensional features for each image.
   - Features and metadata are saved into the `.results/` directory for reuse.

5. **Step 2 – Classical preprocessing (PCA pipeline)**

   - The script merges the tiny original validation set with train, re‑splits 80/20 (stratified), and then fits a `StandardScaler + PCA(64)` pipeline.
   - Transformed `X_train`, `X_val`, `X_test` and labels are stored as `.npy` files and the fitted preprocessing pipeline is saved with `joblib`.

6. **Step 3 – Classical baseline training**

   - Train the classical MLP baseline on the 64‑dimensional features.
   - Uses weighted BCE loss, Adam optimizer, cosine learning‑rate schedule and early stopping based on validation loss.

7. **Step 4 – Quantum model training**

   - Builds a PennyLane QNode with the chosen device (`lightning.gpu` where possible) and the custom ansatz (Rot + ring CNOT layers).
   - Trains parameters with Adam and a cosine annealing learning‑rate schedule with warm‑up, using a **weighted MSE** loss in the label space \{-1, +1\} to handle class imbalance.
   - Saves the best parameters, training history and test metrics to `.results/quantumresults.json`.

8. **Step 5 – Threshold scan & evaluation**

   - On the test set, the model outputs continuous probabilities; a threshold scan over values 0.50–0.90 is used to study the trade‑off between sensitivity and specificity and to compute balanced accuracy.

***

## 6. Results

### 6.1 Classical vs hybrid performance

On the 624‑image test set (62.5 % pneumonia, 37.5 % normal), the following metrics were obtained:

| Metric             | Classical (ResNet‑50 + MLP) | Hybrid (ResNet‑50 + VQC) | Difference |
|--------------------|-----------------------------|---------------------------|-----------|
| Accuracy           | 85.90 %                     | 71.63 %                   | −14.27 %  |
| Precision          | 83.41 %                     | 75.91 %                   | −7.50 %   |
| Recall (Sensitivity) | 97.95 %                  | 80.00 %                   | −17.95 %  |
| Specificity        | 70.51 %                     | 57.69 %                   | −12.82 %  |
| F1‑score           | 0.9022                      | 0.6915                    | −0.2107   |
| AUC‑ROC            | 0.912                       | 0.7445                    | −0.1675   |
| Trainable params (classifier) | 164,865        | 108                       | −152,757  |
| Training time      | ~9 s                        | ~135 min                  | —         |

The hybrid model does **not** yet match the classical baseline in raw predictive performance, but it achieves a non‑trivial AUC (~0.74) with a **dramatically smaller number of trainable parameters** in the decision stage, which is important in the NISQ regime.

### 6.2 Behaviour and interpretation

- The hybrid model shows relatively high **sensitivity (80 %)** but lower specificity, meaning it tends to over‑call pneumonia rather than miss it.
- This behaviour can be acceptable in a screening setting (better to flag suspicious cases than to miss disease), but it increases the number of false positives compared to the classical baseline.
- A noticeable **domain shift** between train/validation and test label distributions (pneumonia share drops from ~74.2 % to 62.5 %) likely contributes to the performance gap and increased false positives on the test set.

***

## 7. Limitations & future work

Some key limitations identified in the thesis:

- Training the VQC on a classical GPU simulator is expensive; a 50‑epoch run with 6 qubits and 6 layers took about **135 minutes**, limiting hyperparameter exploration.
- Only a single public pediatric dataset from one institution was used, so generalization to other populations, hospitals or acquisition protocols is unknown.
- The 2048→64 PCA reduction may discard clinically relevant information, which caps the achievable performance of the quantum classifier.
- The quantum model was trained on an **ideal simulator** without realistic noise; performance on real NISQ hardware would likely be worse without explicit error‑mitigation techniques.

Future directions suggested by the work include:

- Running the VQC on real hardware (e.g. IBM Quantum) with noise mitigation.  
- Exploring alternative quantum architectures (e.g. quantum kernels, quanvolutional layers, ensembles with classical models).  
- Using richer, multi‑institutional datasets and domain‑adaptation techniques to address dataset shift.

***

## 8. Citation

If you use this code or ideas from the project, please cite the associated SOČ thesis (Czech, English annotation in the front matter):

> M. Forg, *Hybridní model pro detekci pneumonie / Hybrid Model for Pneumonia Detection*, Středoškolská odborná činnost (SOČ), 2026.

You can also link to this repository and mention that it implements a hybrid ResNet‑50 + VQC pipeline for pneumonia detection using PennyLane and PyTorch.
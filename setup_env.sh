#!/usr/bin/env bash
#
# setup_env.sh — create the Python virtual environment and install dependencies
# for the hybrid QML pneumonia-detection pipeline.
#
# Target deployment: Ubuntu server with CUDA 12.x and 2x NVIDIA L40 (48 GB).
# GPU auto-selection (least-loaded device) is handled by the pipeline code
# (config gpu_strategy=least_loaded), NOT by this script.
#
# The script is also safe on CPU-only machines (e.g. local dev boxes):
# the optional GPU packages are only attempted when nvidia-smi is present,
# and a failure there is non-fatal (CPU fallback via lightning.qubit).
#
set -e

echo "==> Creating virtual environment: .venv"
python -m venv .venv
source .venv/bin/activate

echo "==> Installing core dependencies from requirements.txt"
pip install --upgrade pip
pip install -r requirements.txt

# --- Optional GPU acceleration (PennyLane lightning.gpu via cuQuantum) ---
# Only attempt when a CUDA-capable GPU is actually present.
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "==> CUDA detected — installing GPU acceleration packages"
    echo "    (pennylane-lightning-gpu==0.45.0, custatevec-cu12)"
    if ! pip install "pennylane-lightning-gpu==0.45.0" "custatevec-cu12"; then
        echo "!! GPU package installation failed — skipping."
        echo "!! The pipeline will fall back to the CPU simulator (lightning.qubit)."
    fi
else
    echo "==> nvidia-smi not found — no CUDA GPU detected."
    echo "    Skipping GPU acceleration; the pipeline will use lightning.qubit (CPU)."
fi

echo ""
echo "==> Environment ready."
echo "    Python: $(python --version)"
echo "    Activate with: source .venv/bin/activate"
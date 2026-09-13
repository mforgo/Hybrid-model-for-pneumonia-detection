#!/usr/bin/env bash
# smoke_test.sh — quick end-to-end verification of the full pipeline.
#
# Runs every stage (data → features → vae → ansatz → train → evaluate →
# analysis) on a 32-sample subset with minimal epochs so that the entire
# pipeline executes in a few minutes on GPU (slightly longer on CPU).
#
# Expected runtime: ~2–5 min on GPU, ~5–10 min on CPU.
# Intended for CI / pre-commit checks / post-merge verification.

set -euo pipefail

# ---------------------------------------------------------------------------
# Activate project venv if present
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_ROOT/.venv/bin/activate"
fi

# ---------------------------------------------------------------------------
# Build the command
# ---------------------------------------------------------------------------
OVERRIDES=(
    data.subset=32
    vae.epochs=2
    vae.batch_size=8
    vqc.epochs=1
    vqc.batch_size=8
    mlp.epochs=1
    features.dann_epochs=1
)

CMD=(python -m pipeline --stage all --overrides "${OVERRIDES[@]}")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
echo "============================================================"
echo " SMOKE TEST — full pipeline on tiny subset"
echo " Command: ${CMD[*]}"
echo "============================================================"

if "${CMD[@]}"; then
    echo ""
    echo "============================================================"
    echo " SMOKE TEST PASSED"
    echo "============================================================"
    exit 0
else
    echo ""
    echo "============================================================"
    echo " SMOKE TEST FAILED — pipeline exited with non-zero status"
    echo "============================================================"
    exit 1
fi

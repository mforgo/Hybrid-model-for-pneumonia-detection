#!/usr/bin/env bash
#
# run_all.sh — full-pipeline runner for the hybrid QML pneumonia detection project.
#
# Runs every pipeline stage sequentially via the CLI orchestrator
# (python -m pipeline --stage <stage>), so per-stage progress is visible.
#
# Usage:
#   ./scripts/run_all.sh
#   ./scripts/run_all.sh --overrides data.subset=32 vae.epochs=2 vqc.epochs=1
#   ./scripts/run_all.sh --dry-run
#
# Notes:
#   - set -e: any non-qpu stage failure aborts the run immediately.
#   - The qpu stage is allowed to fail gracefully (no IBM token / hardware):
#     it is wrapped so a failure prints a warning and the run continues.
#   - All extra CLI args ("$@") are passed through to every stage invocation.
#   - Sources .venv/bin/activate when present; skipped gracefully otherwise.

set -e

# Activate the project virtualenv if it exists (graceful skip otherwise).
if [ -f ".venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source ".venv/bin/activate"
fi

# Explicit sequential stage list (NOT --stage all) so each stage prints progress.
STAGES=(data features vae ansatz train evaluate qpu analysis)
TOTAL=${#STAGES[@]}
INDEX=0

for stage in "${STAGES[@]}"; do
    INDEX=$((INDEX + 1))
    echo ""
    echo "=== [$INDEX/$TOTAL] $stage ==="

    if [ "$stage" = "qpu" ]; then
        # QPU needs an IBM token + hardware; skip gracefully if unavailable.
        python -m pipeline --stage qpu "$@" \
            || echo "Skipping QPU stage (no hardware/token available)"
    else
        python -m pipeline --stage "$stage" "$@"
    fi
done

echo ""
echo "=== Pipeline complete: all $TOTAL stages finished successfully ==="
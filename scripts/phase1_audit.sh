#!/bin/bash
# Phase 1: tokenizer audit (CPU; runs on Babel login node, no SLURM needed).
# Usage:  ./scripts/phase1_audit.sh

set -e
cd "$(dirname "$0")/.."

# Activate conda env
if [ -z "$CONDA_DEFAULT_ENV" ] || [ "$CONDA_DEFAULT_ENV" != "blackbox" ]; then
    if [ -f "$CONDA_PREFIX/etc/profile.d/conda.sh" ]; then
        source "$CONDA_PREFIX/etc/profile.d/conda.sh"
    fi
    conda activate blackbox
fi

# BLACKBOX_DATA defaults to /data/user_data/$USER/blackbox if unset
: "${BLACKBOX_DATA:=/data/user_data/$USER/blackbox}"
mkdir -p "$BLACKBOX_DATA/tokenizer_audit"
mkdir -p "$BLACKBOX_DATA/logs"

LOG="$BLACKBOX_DATA/logs/phase1_audit_$(date +%Y%m%d_%H%M%S).log"
echo "Logging to $LOG"

python code/run_phase1_audit.py --data-dir "$BLACKBOX_DATA" 2>&1 | tee "$LOG"

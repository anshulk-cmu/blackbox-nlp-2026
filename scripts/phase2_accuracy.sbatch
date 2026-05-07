#!/bin/bash
#SBATCH --job-name=blackbox_phase2_acc
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/data/user_data/%u/blackbox/logs/%x_%j.out
#SBATCH --error=/data/user_data/%u/blackbox/logs/%x_%j.err
#
# Phase 2: accuracy reproduction. Per-model.
# Usage:  sbatch scripts/phase2_accuracy.sbatch <model_key>
#         where <model_key> in {gpt-j-6b, pythia-6.9b, llama-3.1-8b}
#
# A6000 (48GB) fits Llama 3.1 8B bf16 (~16GB weights + KV cache + activations)
# comfortably with batch 32. Switch to A100_80GB:1 if A6000 is contended:
#   sbatch --gres=gpu:A100_80GB:1 scripts/phase2_accuracy.sbatch <model_key>

set -e

MODEL_KEY=$1
if [ -z "$MODEL_KEY" ]; then
    echo "ERROR: model key required as first argument"
    echo "Usage: sbatch scripts/phase2_accuracy.sbatch <gpt-j-6b|pythia-6.9b|llama-3.1-8b>"
    exit 1
fi

cd ~/blackbox-nlp-2026
git pull --ff-only

# Conda env
module load anaconda3 || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate blackbox

: "${BLACKBOX_DATA:=/data/user_data/$USER/blackbox}"

echo "=========================================="
echo "Phase 2: $MODEL_KEY"
echo "Job: $SLURM_JOB_ID on $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "=========================================="

python code/run_phase2_accuracy.py --model "$MODEL_KEY" --data-dir "$BLACKBOX_DATA"

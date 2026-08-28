#!/bin/bash
# ==============================================================================
# run_all.sh — run every config serially on ONE reserved GPU node.
#
#   condor_submit adp_batch.sub
#
# Use this instead of adp.sub when the account is limited to one concurrent
# job: the whole sweep holds a single node and iterates over config indexes
# 0..N-1 back-to-back, so runs never wait on the scheduler between configs.
#
# The skip-guard in run_experiments.py makes reruns safe: (config, seed)
# pairs already recorded as OK in run_manifest.jsonl are skipped, so a
# partial batch can be resubmitted.
# ==============================================================================

set -euo pipefail

# HTCondor's vanilla job sandbox does NOT guarantee a USER env var; derive it.
export USER="${USER:-$(id -un)}"
echo "=== run_all.sh: user=${USER}, job=$(hostname) ==="

CONFIG_FILE=${1:-configs/experiments.jsonl}
OUTPUT_DIR=${2:-/scratch/$USER/adp_final}
SEEDS=${3:-42}
DEVICE=${4:-0}

# HTCondor does not shell-expand $USER inside `arguments`; expand it here
# (same as run_adp.sh) so /scratch/$USER/adp_v2 becomes /scratch/hyonghua/adp_v2.
OUTPUT_DIR=$(eval echo "$OUTPUT_DIR")

mkdir -p condor_logs "$OUTPUT_DIR"

# --- Deterministic environment (same as run_adp.sh) ---
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=42
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export WANDB_MODE=offline

# --- HuggingFace cache (reuse whatever the cluster already downloaded) ---
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
mkdir -p "$HF_HOME"

# --- Python environment (path-based conda env on scratch) ---
ADP_ENV="/scratch/${USER}/conda_envs/adp"
if [ ! -x "${ADP_ENV}/bin/python" ]; then
    echo "ERROR: conda env not found at ${ADP_ENV}" >&2
    exit 1
fi
export PATH="${ADP_ENV}/bin:${PATH}"
export CONDA_PREFIX="${ADP_ENV}"

# --- GPU guard: fail fast instead of silently falling back to CPU ---
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA not available - refusing to run on CPU")
print("GPU:", torch.cuda.get_device_name(0))
PY

# --- Run every config x seed serially; a failing run does NOT stop the sweep ---
echo "=== configs x seeds [${SEEDS}] serially on $(hostname) ==="

python src/run_experiments.py \
    --config "$CONFIG_FILE" --seeds "$SEEDS" --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR"

echo
echo "=== sweep finished on $(hostname) ==="

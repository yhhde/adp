#!/bin/bash
# ==============================================================================
# run_adp.sh — HTCondor worker: one (config index, seed) pair per job
#
# Submit via:  condor_submit adp.sub
#
# HTCondor maps $(Process) -> (config index, seed):
#   index = Process % N_CONFIGS
#   seed  = SEEDS[ Process / N_CONFIGS ]      (SEEDS default "42")
#   e.g. 8 configs x 1 seed = 8 jobs (Process 0-7)
#
# Params live in adp.sub; the (config index, seed) grid is derived from
# $(Process), so the submit file only needs `queue N_CONFIGS x N_SEEDS`.
# ==============================================================================

set -euo pipefail

# HTCondor's vanilla job sandbox does NOT guarantee a USER env var, but this
# script references ${USER} in default paths. Derive it from the system so
# `set -u` never trips on an unbound variable.
export USER="${USER:-$(id -un)}"
echo "=== run_adp.sh: user=${USER}, job=$(hostname) ==="

CONFIG_FILE=${1:-configs/experiments.jsonl}
OUTPUT_DIR=${2:-/scratch/$USER/adp_final}
SEEDS=${SEEDS:-"42"}
PROCESS=${4:-0}   # $(Process) is forwarded as the 4th argument by adp.sub

# Expand shell variables inside the argument (e.g. $USER coming from adp.sub)
OUTPUT_DIR=$(eval echo "$OUTPUT_DIR")

mkdir -p condor_logs "$OUTPUT_DIR"

# --- Deterministic environment ---
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
# No `conda activate` on purpose: the HTCondor job sandbox has neither conda on
# PATH nor a reliable HOME, so the shell-hook/source route dies. A path-based
# env is self-contained - prepending its bin/ to PATH is all it needs.
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

# --- Unpack the 2D job grid ---
readarray -t SEED_ARR <<< "$(echo "$SEEDS" | tr ' ' '\n')"
N_CONFIGS=$(grep -cve '^\s*$' "$CONFIG_FILE")
IDX=$(( PROCESS % N_CONFIGS ))
SEED=${SEED_ARR[$(( PROCESS / N_CONFIGS ))]}

echo "=== job ${PROCESS}: config index ${IDX}, seed ${SEED}, output ${OUTPUT_DIR} ==="

python src/run_experiments.py \
    --config "$CONFIG_FILE" \
    --index "$IDX" \
    --seed "$SEED" \
    --device 0 \
    --output-dir "$OUTPUT_DIR"

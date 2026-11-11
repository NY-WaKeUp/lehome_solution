#!/usr/bin/env bash
# Ultra-short Aserver 2x4090 smoke: prove overlay config trains without OOM.
# Does NOT run the full 1500x2 ablation.
set -euo pipefail
ROOT=/home/ningyu/lehome_solution
export PATH="$HOME/.local/bin:$PATH"
cd "$ROOT"

# Prefer Mac proxy tunnel (17897 -> Mac 7897) for any Hub/GCS fallback.
if [[ -f scripts/autodl/mac_proxy.sh ]]; then
  # shellcheck disable=SC1091
  source scripts/autodl/mac_proxy.sh on
fi

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
# Use ~/.cache/openpi where pi05_base already lives. A separate OPENPI_DATA_HOME
# tree with symlinks breaks openpi.shared.download.maybe_download path checks.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$HOME/.cache/openpi}"

BATCH="${BATCH:-16}"
STEPS="${STEPS:-3}"
CFG="${CFG:-configs/rl_pipeline_sim_aserver_4090.yaml}"
EXP="${EXP:-sim_rl_aserver_4090_smoke_b${BATCH}}"

mkdir -p logs/aserver_ablation
LOG="logs/aserver_ablation/smoke_${EXP}_s${STEPS}.log"

echo "smoke start cfg=$CFG batch=$BATCH steps=$STEPS exp=$EXP $(date -Is)" | tee "$LOG"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv | tee -a "$LOG"

.venv/bin/python -u scripts/run_rl_pipeline.py \
  --config="$CFG" \
  --train_only \
  --exp_name "$EXP" \
  --steps_per_iteration "$STEPS" \
  --batch_size "$BATCH" \
  2>&1 | tee -a "$LOG"

echo "smoke done $(date -Is)" | tee -a "$LOG"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv | tee -a "$LOG"
grep -E "Step |Training complete|OOM|Out of memory|RESOURCE_EXHAUSTED|Traceback" "$LOG" | tail -40 | tee -a "$LOG"

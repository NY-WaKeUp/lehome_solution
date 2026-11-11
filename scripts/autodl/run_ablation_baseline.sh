#!/bin/bash
set -uo pipefail
exec 9>/root/autodl-tmp/gpu_exclusive.lock
flock -n 9 || { echo lock_held; exit 1; }
export PATH="$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi-cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-home
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
cd /root/autodl-tmp/lehome_solution
LOG=/root/autodl-tmp/ablation_sampling_b128.log
RESULT=/root/autodl-tmp/ablation_results.md
echo "# AWR ablation start $(date -Is) batch=128" > "$RESULT"
echo "## sampling (awr_loss_weighted=false)" >> "$RESULT"
.venv/bin/python -u scripts/run_rl_pipeline.py \
  --config=configs/rl_pipeline_sim.yaml \
  --train_only --steps_per_iteration 30 --batch_size 128 \
  > "$LOG" 2>&1
EC=$?
echo "sampling exit=$EC $(date -Is)" | tee -a "$RESULT"
grep -E "Step |Progress on:|Training complete" "$LOG" | tail -40 | tee -a "$RESULT"
# cleanup ckpt to save disk
find outputs/checkpoints/pi_modified_bc_rl/sim_rl -maxdepth 1 -type d -regex ".*/[0-9]+" -exec rm -rf {} + 2>/dev/null || true
df -h /root/autodl-tmp | tail -1 | tee -a "$RESULT"
echo DONE_ABLATION_SAMPLING $(date -Is) | tee -a "$RESULT"

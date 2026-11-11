#!/bin/bash
set -uo pipefail
exec 9>/root/autodl-tmp/gpu_exclusive.lock
flock -n 9 || { echo "GPU lock held; abort"; exit 1; }
EXIST=$(ps -eo pid,cmd | grep -E "scripts/train.py|run_rl_pipeline" | grep -v grep | wc -l)
if [ "$EXIST" -gt 0 ]; then echo ABORT_BUSY; exit 2; fi

export PATH="$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi-cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-home
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
cd /root/autodl-tmp/lehome_solution

BATCH=128
STATUS=/root/autodl-tmp/ablation/STATUS.md
ASSET_SRC=outputs/assets/pi_modified_bc_rl/sim_rl
mkdir -p /root/autodl-tmp/ablation logs

# ensure disk_guard running (single)
if ! pgrep -f "/root/autodl-tmp/disk_guard.sh" >/dev/null 2>&1; then
  setsid bash /root/autodl-tmp/disk_guard.sh </dev/null >/dev/null 2>&1 &
  echo "started disk_guard pid=$!" | tee -a "$STATUS"
fi

{
  echo ""
  echo "## resume_orchestrator $(date -Is)"
  echo "disk_policy: save_interval=1000 keep_period=null max_to_keep=1(orbax) disk_guard thresh=15G"
  echo "openpi-cache: symlink -> /root/openpi-cache (system disk)"
  df -h /root/autodl-tmp | tail -1
} | tee -a "$STATUS"

seed_assets() {
  local EXP="$1"
  local DEST="outputs/checkpoints/pi_modified_bc_rl/${EXP}/assets/sim_rl"
  if [ -f "${DEST}/norm_stats.json" ]; then
    echo "assets ok: $DEST" | tee -a "$STATUS"
    return 0
  fi
  mkdir -p "outputs/checkpoints/pi_modified_bc_rl/${EXP}/assets"
  rm -rf "$DEST"
  cp -a "$ASSET_SRC" "$DEST"
  echo "seeded assets -> $DEST" | tee -a "$STATUS"
}

free_exp_numbered() {
  local EXP="$1"
  local DIR="outputs/checkpoints/pi_modified_bc_rl/${EXP}"
  find "$DIR" -maxdepth 1 -type d -regex ".*/[0-9]+" -exec rm -rf {} + 2>/dev/null || true
  find "$DIR" -maxdepth 2 \( -name "*.tmp*" -o -name "*orbax-checkpoint-tmp*" \) -exec rm -rf {} + 2>/dev/null || true
}

# --- baseline resume from latest complete numbered ckpt ---
BASE_EXP=sim_rl_awr_sample_b128
BASE_DIR=outputs/checkpoints/pi_modified_bc_rl/${BASE_EXP}
LATEST=$(find "$BASE_DIR" -maxdepth 1 -type d -regex ".*/[0-9]+" | while read -r d; do
  [ -d "$d/params" ] && basename "$d"
done | sort -n | tail -1)

if [ -z "${LATEST:-}" ]; then
  echo "NO_COMPLETE_CKPT — cannot resume; abort" | tee -a "$STATUS"
  exit 3
fi

# remaining steps to hit ~10001 total: target = latest + steps + 1
TARGET=10001
STEPS=$((TARGET - LATEST - 1))
if [ "$STEPS" -lt 1 ]; then
  echo "baseline already at/past target latest=$LATEST target=$TARGET" | tee -a "$STATUS"
  BASE_EC=0
else
  {
    echo ""
    echo "## arm=baseline_RESUME $(date -Is)"
    echo "resume_from_step=$LATEST steps_per_iteration=$STEPS target_num_train_steps=$((LATEST+STEPS+1))"
    echo "config=configs/rl_pipeline_sim.yaml exp_name=$BASE_EXP"
  } | tee -a "$STATUS"

  # drop truncated temps only; KEEP latest complete ckpt
  find "$BASE_DIR" -maxdepth 2 \( -name "*.tmp*" -o -name "*orbax-checkpoint-tmp*" \) -exec rm -rf {} + 2>/dev/null || true
  seed_assets "$BASE_EXP" || exit 4

  LOG=logs/awr_baseline_b128_resume.log
  : > "$LOG"
  set +e
  .venv/bin/python -u scripts/run_rl_pipeline.py \
    --config=configs/rl_pipeline_sim.yaml \
    --train_only \
    --steps_per_iteration "$STEPS" \
    --batch_size "$BATCH" \
    --exp_name "$BASE_EXP" \
    --num_workers 2 \
    > "$LOG" 2>&1
  BASE_EC=$?
  set -e
  echo "baseline_resume exit=$BASE_EC finished=$(date -Is)" | tee -a "$STATUS"
  grep -E "Step |Training complete|ERROR|Resuming|restor|No space" "$LOG" | head -20 | tee -a "$STATUS"
  grep -E "Step |Training complete|ERROR|No space" "$LOG" | tail -15 | tee -a "$STATUS"
  df -h /root/autodl-tmp | tail -1 | tee -a "$STATUS"
fi

# --- loss_weighted only after baseline OK ---
if [ "$BASE_EC" -ne 0 ]; then
  echo "SKIP loss_weighted because baseline_ec=$BASE_EC" | tee -a "$STATUS"
  exit "$BASE_EC"
fi

LOSS_EXP=sim_rl_awr_loss_b128
{
  echo ""
  echo "## arm=loss_weighted $(date -Is)"
  echo "config=configs/rl_pipeline_sim_loss_weighted.yaml exp_name=$LOSS_EXP"
} | tee -a "$STATUS"

# free baseline numbered ckpts to protect 50G (keep assets/logs); fresh loss arm
free_exp_numbered "$BASE_EXP"
free_exp_numbered "$LOSS_EXP"
seed_assets "$LOSS_EXP" || exit 5
df -h /root/autodl-tmp | tee -a "$STATUS"

LOG=logs/awr_loss_weighted_b128.log
: > "$LOG"
set +e
.venv/bin/python -u scripts/run_rl_pipeline.py \
  --config=configs/rl_pipeline_sim_loss_weighted.yaml \
  --train_only \
  --steps_per_iteration 10000 \
  --batch_size "$BATCH" \
  --exp_name "$LOSS_EXP" \
  --num_workers 2 \
  > "$LOG" 2>&1
LOSS_EC=$?
set -e
echo "loss_weighted exit=$LOSS_EC finished=$(date -Is)" | tee -a "$STATUS"
grep -E "Step 0:|Training complete|ERROR|No space" "$LOG" | head -8 | tee -a "$STATUS"
grep -E "Step |Training complete|ERROR|No space" "$LOG" | tail -10 | tee -a "$STATUS"
df -h /root/autodl-tmp | tail -1 | tee -a "$STATUS"

echo "" | tee -a "$STATUS"
echo "DONE_ABLATION base_ec=$BASE_EC loss_ec=$LOSS_EC $(date -Is)" | tee -a "$STATUS"

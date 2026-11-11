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

# sustainable disk policy
if ! pgrep -f "/root/autodl-tmp/disk_guard.sh" >/dev/null 2>&1; then
  setsid bash /root/autodl-tmp/disk_guard.sh </dev/null >/dev/null 2>&1 &
fi

BATCH=192
STEPS=10000
STATUS=/root/autodl-tmp/ablation/STATUS.md
ASSET_SRC=outputs/assets/pi_modified_bc_rl/sim_rl
mkdir -p /root/autodl-tmp/ablation logs

{
  echo "# AWR ablation on A800 (batch=192, Ilia-aligned)"
  echo "started: $(date -Is)"
  echo "batch_size=$BATCH steps_per_iteration=$STEPS fsdp=1 flow_samples=5 num_workers_dl=8 ema=0.99 use_xsa=true"
  echo "MAX_OK=192 (probe 160/176/192 all OK; peak~70GB VRAM / ~83GiB cgroup)"
  echo "fresh_start: new exp names; do NOT hard-resume b128/ckpt500 into different batch"
  echo "history_kept: sim_rl_awr_sample_b128/500 (~12G) retained as archive"
  echo "disk_policy: save_interval=500 keep_period=5000 max_to_keep=1 sync_orbax=1 disk_guard thresh=15G (never purge in-flight orbax-tmp while train.py runs)"
  echo "fix: seed checkpoint assets (norm_stats) for fresh --exp_name"
} > "$STATUS"

seed_assets() {
  local EXP="$1"
  local DEST="outputs/checkpoints/pi_modified_bc_rl/${EXP}/assets/sim_rl"
  if [ -f "${DEST}/norm_stats.json" ]; then
    echo "assets ok: $DEST" | tee -a "$STATUS"
    return 0
  fi
  if [ ! -f "${ASSET_SRC}/norm_stats.json" ]; then
    echo "MISSING_ASSET_SRC $ASSET_SRC" | tee -a "$STATUS"
    return 1
  fi
  mkdir -p "outputs/checkpoints/pi_modified_bc_rl/${EXP}/assets"
  rm -rf "$DEST"
  cp -a "$ASSET_SRC" "$DEST"
  echo "seeded assets -> $DEST" | tee -a "$STATUS"
}

# Only wipe numbered ckpts under b192 exps (never touch *_b128 history)
clean_b192_numbered() {
  local EXP="$1"
  local BASE="outputs/checkpoints/pi_modified_bc_rl"
  # current exp numbered + temps
  if [ -d "$BASE/$EXP" ]; then
    find "$BASE/$EXP" -maxdepth 1 -type d -regex ".*/[0-9]+" -exec rm -rf {} + 2>/dev/null || true
    find "$BASE/$EXP" \( -name "*.tmp*" -o -name "*orbax-checkpoint-tmp*" -o -name "*_tmp_*" \) -print -exec rm -rf {} + 2>/dev/null || true
    rm -f "$BASE/$EXP"/_pipeline_config_iter*.pkl
  fi
  # other b192 arms' numbered ckpts to free disk when switching arms
  for other in sim_rl_awr_sample_b192 sim_rl_awr_sample_b192_vanilla sim_rl_awr_loss_b192; do
    if [ "$other" != "$EXP" ] && [ -d "$BASE/$other" ]; then
      find "$BASE/$other" -maxdepth 1 -type d -regex ".*/[0-9]+" -exec rm -rf {} + 2>/dev/null || true
    fi
  done
}

run_arm() {
  local NAME="$1"
  local CFG="$2"
  local EXP="$3"
  local LOG="logs/awr_${NAME}_b${BATCH}.log"
  echo "" | tee -a "$STATUS"
  echo "## arm=$NAME $(date -Is)" | tee -a "$STATUS"
  echo "config=$CFG exp_name=$EXP log=$LOG batch=$BATCH fresh=true" | tee -a "$STATUS"

  clean_b192_numbered "$EXP"
  seed_assets "$EXP" || return 3

  : > "$LOG"
  set +e
  .venv/bin/python -u scripts/run_rl_pipeline.py \
    --config="$CFG" \
    --train_only \
    --steps_per_iteration "$STEPS" \
    --batch_size "$BATCH" \
    --exp_name "$EXP" \
    > "$LOG" 2>&1
  EC=$?
  set -e
  echo "exit=$EC finished=$(date -Is)" | tee -a "$STATUS"
  grep -E "Step 0:|Training complete|ERROR|Progress on:|Seeded checkpoint|Normalization|Loaded norm" "$LOG" | head -12 | tee -a "$STATUS"
  grep -E "Step 0:|Training complete|ERROR|Progress on:" "$LOG" | tail -10 | tee -a "$STATUS"
  df -h /root/autodl-tmp | tail -1 | tee -a "$STATUS"
  return $EC
}

# baseline: sampler AWR (awr_loss_weighted=false)
run_arm baseline configs/rl_pipeline_sim.yaml sim_rl_awr_sample_b192_vanilla
BASE_EC=$?

# loss-weighted AWR
run_arm loss_weighted configs/rl_pipeline_sim_loss_weighted.yaml sim_rl_awr_loss_b192
LOSS_EC=$?

echo "" | tee -a "$STATUS"
echo "DONE_ABLATION_B192 base_ec=$BASE_EC loss_ec=$LOSS_EC $(date -Is)" | tee -a "$STATUS"

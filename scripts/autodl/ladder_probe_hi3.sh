#!/bin/bash
set -uo pipefail
# deliberately NO set -e
exec 9>/root/autodl-tmp/gpu_exclusive.lock
flock -n 9 || { echo "GPU lock held"; exit 1; }
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

CKPT_ROOT=outputs/checkpoints/pi_modified_bc_rl/sim_rl
RESULT=/root/autodl-tmp/max_batch_probe.md
YAML=configs/rl_pipeline_sim.yaml
STEPS=3
ORIG_SAVE=$(grep -E "^save_interval:" "$YAML" | awk '{print $2}')
sed -i "s/^save_interval:.*/save_interval: 100000/" "$YAML"
restore_yaml() { sed -i "s/^save_interval:.*/save_interval: ${ORIG_SAVE}/" "$YAML"; echo "yaml_restored save_interval=$ORIG_SAVE" | tee -a "$RESULT"; }
trap restore_yaml EXIT

cleanup_ckpts() {
  find "$CKPT_ROOT" -maxdepth 1 -type d -regex ".*/[0-9]+" -print -exec rm -rf {} + 2>/dev/null || true
  find "$CKPT_ROOT" -maxdepth 2 \( -name "*.tmp*" -o -name "*orbax-checkpoint-tmp*" \) -exec rm -rf {} + 2>/dev/null || true
  rm -f "$CKPT_ROOT/_pipeline_config_iter0.pkl"
  sync
  df -h /root/autodl-tmp | tail -1
}

reset_state() {
  .venv/bin/python - <<PY
import json
from pathlib import Path
p = Path("outputs/checkpoints/pi_modified_bc_rl/sim_rl/pipeline_state.json")
st = {
  "iteration": 0, "phase": "warmup", "current_train_steps": 0,
  "bc_sampling_share": 1.0, "rl_datasets": [], "dagger_datasets": [],
  "wandb_run_id": None, "checkpoint_dir": str(p.parent),
  "latest_checkpoint_step": None, "consumed_rollout_ids": [],
}
p.write_text(json.dumps(st, indent=2)+"\n")
print("state_reset")
PY
}

sample_loop() {
  local out="$1"
  : > "$out"
  while true; do
    ts=$(date -Is)
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d " ")
    cggib=$(awk '{printf "%.2f", $1/1024/1024/1024}' /sys/fs/cgroup/memory/memory.usage_in_bytes)
    avail=$(awk '/MemAvailable/ {printf "%.2f", $2/1024/1024}' /proc/meminfo)
    echo "$ts gpu_mib=${gpu%,*} util=${gpu#*,} cgroup_gib=$cggib avail_gib=$avail" >> "$out"
    sleep 2
  done
}

echo "" | tee -a "$RESULT"
echo "# continue probe HI3 $(date -Is) tiers=192 prior_MAX_OK=176" | tee -a "$RESULT"
MAX_OK=176

for B in 192; do
  echo "" | tee -a "$RESULT"
  echo "## batch_size=$B $(date -Is)" | tee -a "$RESULT"
  cleanup_ckpts | tee -a "$RESULT"
  reset_state
  free -h | head -2 | tee -a "$RESULT"
  LOG=/root/autodl-tmp/probe_b${B}.log
  MON=/root/autodl-tmp/probe_b${B}_mon.log
  rm -f "$LOG" "$MON"
  sample_loop "$MON" &
  MON_PID=$!

  .venv/bin/python -u scripts/run_rl_pipeline.py \
    --config=configs/rl_pipeline_sim.yaml \
    --train_only \
    --steps_per_iteration "$STEPS" \
    --batch_size "$B" \
    --exp_name sim_rl \
    > "$LOG" 2>&1
  EC=$?

  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true
  pkill -KILL -f "multiprocessing.spawn" 2>/dev/null || true
  pkill -KILL -f "multiprocessing.resource_tracker" 2>/dev/null || true
  sleep 2

  PEAK_GPU=$(grep -oE "gpu_mib=[0-9.]+" "$MON" 2>/dev/null | cut -d= -f2 | sort -n | tail -1)
  PEAK_CG=$(grep -oE "cgroup_gib=[0-9.]+" "$MON" 2>/dev/null | cut -d= -f2 | sort -n | tail -1)
  MIN_AVAIL=$(grep -oE "avail_gib=[0-9.]+" "$MON" 2>/dev/null | cut -d= -f2 | sort -n | head -1)
  STEPS_N=$(grep -cE "Step [0-9]+:" "$LOG" 2>/dev/null); STEPS_N=${STEPS_N:-0}
  TLOG=outputs/checkpoints/pi_modified_bc_rl/sim_rl/logs/training/train_iter0.log
  LOSS=$(grep -E "Step [0-9]+:" "$TLOG" 2>/dev/null | tail -1 | grep -oE " loss=[0-9.]+" | head -1 | cut -d= -f2 || true)
  RATE=$(grep -E "Progress on:" "$TLOG" 2>/dev/null | tail -5 || true)
  REMAT=$(grep -E "Can't reduce memory|rematerialization" "$TLOG" 2>/dev/null | tail -3 || true)
  DMESG=$(dmesg -T 2>/dev/null | grep -iE "oom|killed process" | tail -8 || true)

  TAG=OK
  if grep -qiE "out of memory|RESOURCE_EXHAUSTED|CUDA out of memory" "$LOG" "$TLOG" 2>/dev/null; then TAG=GPU_OOM
  elif [ "$EC" -eq 137 ] || [ "$EC" -eq 9 ]; then TAG=HOST_KILLED_137
  elif [ "$EC" -ne 0 ]; then TAG=FAIL
  elif [ "$STEPS_N" -lt 1 ]; then TAG=NO_STEP
  fi

  {
    echo "result=$TAG exit=$EC steps_logged=$STEPS_N peak_vram_MiB=${PEAK_GPU:-?} peak_cgroup_GiB=${PEAK_CG:-?} min_avail_GiB=${MIN_AVAIL:-?} last_loss=${LOSS:-?}"
    echo "$RATE"
    [ -n "$REMAT" ] && echo "$REMAT"
    [ -n "$DMESG" ] && { echo "dmesg:"; echo "$DMESG"; }
    if [ "$TAG" != "OK" ]; then echo "--- tail train ---"; tail -60 "$TLOG"; echo "--- tail pipeline ---"; tail -40 "$LOG"; fi
    true
  } | tee -a "$RESULT"

  echo "cleanup_begin $(date -Is)" | tee -a "$RESULT"
  cleanup_ckpts | tee -a "$RESULT"
  echo "cleanup_done $(date -Is)" | tee -a "$RESULT"
  free -h | head -2 | tee -a "$RESULT"
  awk '{printf "cgroup_after_GiB=%.2f\n", $1/1024/1024/1024}' /sys/fs/cgroup/memory/memory.usage_in_bytes | tee -a "$RESULT"
  test -d outputs/checkpoints/pi_modified_bc_rl/sim_rl_awr_sample_b128/500/params && echo ablation_ckpt500_OK | tee -a "$RESULT"

  if [ "$TAG" != "OK" ]; then
    echo "MAX_OK=$MAX_OK (stop at $B: $TAG)" | tee -a "$RESULT"
    echo "DONE_PROBE_HI3 $(date -Is)" | tee -a "$RESULT"
    exit 0
  fi
  MAX_OK=$B
  echo "MAX_OK_so_far=$MAX_OK" | tee -a "$RESULT"
done
echo "MAX_OK=$MAX_OK" | tee -a "$RESULT"
echo "DONE_PROBE_HI3 $(date -Is)" | tee -a "$RESULT"

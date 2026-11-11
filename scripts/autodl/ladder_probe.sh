#!/bin/bash
set -uo pipefail
exec 9>/root/autodl-tmp/gpu_exclusive.lock
flock -n 9 || { echo "GPU lock held"; exit 1; }

EXIST=$(ps -eo pid,cmd | grep -E "scripts/train.py|run_rl_pipeline|multiprocessing.spawn" | grep -v grep | wc -l)
if [ "$EXIST" -gt 0 ]; then
  echo "ABORT leftover:"; ps -eo pid,rss,cmd | grep -E "train.py|run_rl|spawn" | grep -v grep
  exit 2
fi

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

cleanup_ckpts() {
  find "$CKPT_ROOT" -maxdepth 1 -type d -regex ".*/[0-9]+" -exec rm -rf {} + 2>/dev/null || true
  rm -rf "$CKPT_ROOT"/*.orbax-checkpoint-tmp-* 2>/dev/null || true
  rm -f "$CKPT_ROOT/_pipeline_config_iter0.pkl"
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
    cggib=$(awk "{printf \"%.2f\", \$1/1024/1024/1024}" /sys/fs/cgroup/memory/memory.usage_in_bytes)
    avail=$(awk "/MemAvailable/ {printf \"%.2f\", \$2/1024/1024}" /proc/meminfo)
    echo "$ts gpu_mib=${gpu%,*} util=${gpu#*,} cgroup_gib=$cggib avail_gib=$avail" >> "$out"
    sleep 2
  done
}

echo "" >> "$RESULT"
echo "# continue probe $(date -Is) tiers=96,128" >> "$RESULT"
MAX_OK=64
for B in 96 128; do
  echo "" | tee -a "$RESULT"
  echo "## batch_size=$B $(date -Is)" | tee -a "$RESULT"
  N=$(ps -eo cmd | grep -E "scripts/train.py|run_rl_pipeline" | grep -v grep | wc -l)
  if [ "$N" -gt 0 ]; then echo "ABORT concurrent" | tee -a "$RESULT"; exit 3; fi
  cleanup_ckpts
  reset_state
  LOG=/root/autodl-tmp/probe_b${B}.log
  MON=/root/autodl-tmp/probe_b${B}_mon.log
  rm -f "$LOG" "$MON"
  sample_loop "$MON" &
  MON_PID=$!
  set +e
  .venv/bin/python -u scripts/run_rl_pipeline.py \
    --config=configs/rl_pipeline_sim.yaml \
    --train_only \
    --steps_per_iteration 2 \
    --batch_size "$B" \
    --num_workers 2 \
    > "$LOG" 2>&1
  EC=$?
  set -e
  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true

  ORPH=$(ps -eo pid,cmd | grep "multiprocessing.spawn" | grep -v grep | awk "{print \$1}" || true)
  for p in $ORPH; do kill -9 $p 2>/dev/null || true; done
  sleep 1

  PEAK_GPU=$(grep -oE "gpu_mib=[0-9.]+" "$MON" 2>/dev/null | cut -d= -f2 | sort -n | tail -1 || true)
  PEAK_CG=$(grep -oE "cgroup_gib=[0-9.]+" "$MON" 2>/dev/null | cut -d= -f2 | sort -n | tail -1 || true)
  MIN_AVAIL=$(grep -oE "avail_gib=[0-9.]+" "$MON" 2>/dev/null | cut -d= -f2 | sort -n | head -1 || true)
  STEPS=$(grep -cE "Step [0-9]+:" "$LOG" 2>/dev/null || echo 0)
  RATE=$(grep -E "Progress on:" "$LOG" 2>/dev/null | tail -3 || true)
  LOSS=$(grep -E "Step [0-9]+:" "$LOG" 2>/dev/null | tail -1 | sed -n "s/.*loss=\\([0-9.]*\\).*/\\1/p" || true)
  DMESG=$(dmesg -T 2>/dev/null | grep -iE "oom|killed process" | tail -5 || true)

  if grep -qiE "out of memory|RESOURCE_EXHAUSTED|CUDA out of memory" "$LOG" 2>/dev/null; then TAG=GPU_OOM
  elif [ "$EC" -eq 137 ] || [ "$EC" -eq 9 ]; then TAG=HOST_KILLED
  elif [ "$EC" -ne 0 ]; then TAG=FAIL
  elif [ "${STEPS:-0}" -lt 1 ]; then TAG=NO_STEP
  else TAG=OK
  fi

  {
    echo "result=$TAG exit=$EC steps=$STEPS peak_vram_MiB=${PEAK_GPU:-?} peak_cgroup_GiB=${PEAK_CG:-?} min_avail_GiB=${MIN_AVAIL:-?} loss=${LOSS:-?}"
    echo "$RATE"
    [ -n "$DMESG" ] && { echo "dmesg:"; echo "$DMESG"; }
    [ "$TAG" != "OK" ] && { echo "--- tail ---"; tail -60 "$LOG"; }
  } | tee -a "$RESULT"

  cleanup_ckpts
  free -h | head -2 | tee -a "$RESULT"
  awk "{printf \"cgroup_after_GiB=%.2f\\n\", \$1/1024/1024/1024}" /sys/fs/cgroup/memory/memory.usage_in_bytes | tee -a "$RESULT"
  df -h /root/autodl-tmp | tail -1 | tee -a "$RESULT"

  if [ "$TAG" != "OK" ]; then
    echo "MAX_OK=$MAX_OK (stop at $B: $TAG)" | tee -a "$RESULT"
    echo "DONE_PROBE $(date -Is)" | tee -a "$RESULT"
    exit 0
  fi
  MAX_OK=$B
done
echo "MAX_OK=$MAX_OK" | tee -a "$RESULT"
echo "DONE_PROBE $(date -Is)" | tee -a "$RESULT"

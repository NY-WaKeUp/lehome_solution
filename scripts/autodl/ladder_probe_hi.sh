#!/bin/bash
set -uo pipefail
exec 9>/root/autodl-tmp/gpu_exclusive.lock
flock -n 9 || { echo "GPU lock held"; exit 1; }

EXIST=$(ps -eo pid,cmd | grep -E "scripts/train.py|run_rl_pipeline" | grep -v grep | wc -l)
if [ "$EXIST" -gt 0 ]; then
  echo "ABORT_BUSY"; ps -eo pid,cmd | grep -E "train.py|run_rl" | grep -v grep; exit 2
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
YAML=configs/rl_pipeline_sim.yaml
STEPS=3

# Temporarily raise save_interval to avoid mid-probe ckpt pressure; restore on exit
ORIG_SAVE=$(grep -E "^save_interval:" "$YAML" | awk "{print \$2}")
sed -i "s/^save_interval:.*/save_interval: 100000/" "$YAML"
restore_yaml() { sed -i "s/^save_interval:.*/save_interval: ${ORIG_SAVE}/" "$YAML"; }
trap restore_yaml EXIT

cleanup_ckpts() {
  find "$CKPT_ROOT" -maxdepth 1 -type d -regex ".*/[0-9]+" -exec rm -rf {} + 2>/dev/null || true
  find "$CKPT_ROOT" -maxdepth 2 \( -name "*.tmp*" -o -name "*orbax-checkpoint-tmp*" \) -exec rm -rf {} + 2>/dev/null || true
  rm -f "$CKPT_ROOT/_pipeline_config_iter0.pkl"
  # never touch sim_rl_awr_sample_b128
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
    freeh=$(free -b | awk "/Mem:/ {printf \"%.2f\", \$7/1024/1024/1024}")
    echo "$ts gpu_mib=${gpu%,*} util=${gpu#*,} cgroup_gib=$cggib avail_gib=$avail free_avail_gib=$freeh" >> "$out"
    sleep 2
  done
}

{
  echo ""
  echo "# continue probe HI $(date -Is) tiers=160,176,192 steps=$STEPS num_workers(dataloader hardcode)=2 save_interval_tmp=100000"
  echo "cgroup_limit_GiB=$(awk "{printf \"%.1f\", \$1/1024/1024/1024}" /sys/fs/cgroup/memory/memory.limit_in_bytes)"
  echo "NOTE: ablation ckpt sim_rl_awr_sample_b128/500 preserved untouched"
} | tee -a "$RESULT"

MAX_OK=128
for B in 160 176 192; do
  echo "" | tee -a "$RESULT"
  echo "## batch_size=$B $(date -Is)" | tee -a "$RESULT"
  N=$(ps -eo cmd | grep -E "scripts/train.py|run_rl_pipeline" | grep -v grep | wc -l)
  if [ "$N" -gt 0 ]; then echo "ABORT concurrent" | tee -a "$RESULT"; exit 3; fi
  cleanup_ckpts
  reset_state
  # light disk clear of probe leftovers
  find /root/autodl-tmp -maxdepth 1 -name "probe_b*_mon.log" -delete 2>/dev/null || true
  df -h /root/autodl-tmp | tail -1 | tee -a "$RESULT"
  free -h | head -2 | tee -a "$RESULT"

  LOG=/root/autodl-tmp/probe_b${B}.log
  MON=/root/autodl-tmp/probe_b${B}_mon.log
  rm -f "$LOG" "$MON"
  sample_loop "$MON" &
  MON_PID=$!

  set +e
  .venv/bin/python -u scripts/run_rl_pipeline.py \
    --config=configs/rl_pipeline_sim.yaml \
    --train_only \
    --steps_per_iteration "$STEPS" \
    --batch_size "$B" \
    --exp_name sim_rl \
    > "$LOG" 2>&1
  EC=$?
  set -e

  kill "$MON_PID" 2>/dev/null || true
  wait "$MON_PID" 2>/dev/null || true

  # orphan workers
  ORPH=$(ps -eo pid,cmd | grep -E "multiprocessing.spawn|multiprocessing.resource_tracker" | grep -v grep | awk "{print \$1}" || true)
  for p in $ORPH; do kill -9 $p 2>/dev/null || true; done
  sleep 2

  PEAK_GPU=$(grep -oE "gpu_mib=[0-9.]+" "$MON" 2>/dev/null | cut -d= -f2 | sort -n | tail -1 || true)
  PEAK_CG=$(grep -oE "cgroup_gib=[0-9.]+" "$MON" 2>/dev/null | cut -d= -f2 | sort -n | tail -1 || true)
  MIN_AVAIL=$(grep -oE "avail_gib=[0-9.]+" "$MON" 2>/dev/null | cut -d= -f2 | sort -n | head -1 || true)
  STEPS_N=$(grep -cE "Step [0-9]+:" "$LOG" 2>/dev/null || echo 0)
  RATE=$(grep -E "Progress on:" "$LOG" 2>/dev/null | tail -5 || true)
  LOSS=$(grep -E "Step [0-9]+:" "$LOG" 2>/dev/null | tail -1 | sed -n "s/.*loss=\\([0-9.]*\\).*/\\1/p" || true)
  DMESG=$(dmesg -T 2>/dev/null | grep -iE "oom|killed process" | tail -8 || true)

  if grep -qiE "out of memory|RESOURCE_EXHAUSTED|CUDA out of memory" "$LOG" 2>/dev/null; then TAG=GPU_OOM
  elif [ "$EC" -eq 137 ] || [ "$EC" -eq 9 ]; then TAG=HOST_KILLED_137
  elif [ "$EC" -ne 0 ]; then TAG=FAIL
  elif [ "${STEPS_N:-0}" -lt 1 ]; then TAG=NO_STEP
  else TAG=OK
  fi

  {
    echo "result=$TAG exit=$EC steps_logged=$STEPS_N peak_vram_MiB=${PEAK_GPU:-?} peak_cgroup_GiB=${PEAK_CG:-?} min_avail_GiB=${MIN_AVAIL:-?} last_loss=${LOSS:-?}"
    echo "$RATE"
    [ -n "$DMESG" ] && { echo "dmesg:"; echo "$DMESG"; }
    [ "$TAG" != "OK" ] && { echo "--- tail ---"; tail -80 "$LOG"; }
  } | tee -a "$RESULT"

  cleanup_ckpts
  free -h | head -2 | tee -a "$RESULT"
  awk "{printf \"cgroup_after_GiB=%.2f\\n\", \$1/1024/1024/1024}" /sys/fs/cgroup/memory/memory.usage_in_bytes | tee -a "$RESULT"
  df -h /root/autodl-tmp | tail -1 | tee -a "$RESULT"
  # confirm ablation ckpt still present
  test -d outputs/checkpoints/pi_modified_bc_rl/sim_rl_awr_sample_b128/500/params && echo "ablation_ckpt500_OK" | tee -a "$RESULT"

  if [ "$TAG" != "OK" ]; then
    echo "MAX_OK=$MAX_OK (stop at $B: $TAG)" | tee -a "$RESULT"
    echo "DONE_PROBE_HI $(date -Is)" | tee -a "$RESULT"
    exit 0
  fi
  MAX_OK=$B
done
echo "MAX_OK=$MAX_OK" | tee -a "$RESULT"
echo "DONE_PROBE_HI $(date -Is)" | tee -a "$RESULT"

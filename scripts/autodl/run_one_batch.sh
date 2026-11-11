#!/bin/bash
set -uo pipefail
B=${1:?batch}
STEPS=${2:-2}
exec 9>/root/autodl-tmp/gpu_exclusive.lock
flock -n 9 || { echo lock_held; exit 1; }
EXIST=$(ps -eo pid,cmd | grep -E "scripts/train.py|run_rl_pipeline" | grep -v grep | wc -l)
if [ "$EXIST" -gt 0 ]; then echo ABORT_BUSY; ps -eo pid,cmd | grep -E "train.py|run_rl" | grep -v grep; exit 2; fi

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
LOG=/root/autodl-tmp/probe_b${B}.log
MON=/root/autodl-tmp/probe_b${B}_mon.log

find "$CKPT_ROOT" -maxdepth 1 -type d -regex ".*/[0-9]+" -exec rm -rf {} + 2>/dev/null || true
rm -f "$CKPT_ROOT/_pipeline_config_iter0.pkl"
.venv/bin/python - <<PY
import json
from pathlib import Path
p=Path("outputs/checkpoints/pi_modified_bc_rl/sim_rl/pipeline_state.json")
st=json.loads(p.read_text()) if p.exists() else {}
st.update({"iteration":0,"phase":"warmup","current_train_steps":0,"latest_checkpoint_step":None})
st.setdefault("checkpoint_dir", str(p.parent))
p.write_text(json.dumps(st, indent=2)+"\n")
PY

{
  echo ""
  echo "## batch_size=$B $(date -Is)"
} | tee -a "$RESULT"

: > "$MON"
(
  while true; do
    ts=$(date -Is)
    gpu=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | tr -d " ")
    cggib=$(awk "{printf \"%.2f\", \$1/1024/1024/1024}" /sys/fs/cgroup/memory/memory.usage_in_bytes)
    avail=$(awk "/MemAvailable/ {printf \"%.2f\", \$2/1024/1024}" /proc/meminfo)
    echo "$ts gpu_mib=${gpu%,*} util=${gpu#*,} cgroup_gib=$cggib avail_gib=$avail" >> "$MON"
    sleep 2
  done
) &
MON_PID=$!

set +e
.venv/bin/python -u scripts/run_rl_pipeline.py \
  --config=configs/rl_pipeline_sim.yaml \
  --train_only --steps_per_iteration "$STEPS" --batch_size "$B" \
  > "$LOG" 2>&1
EC=$?
set -e

kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true

PEAK_GPU=$(grep -oE "gpu_mib=[0-9.]+" "$MON" | cut -d= -f2 | sort -n | tail -1)
PEAK_CG=$(grep -oE "cgroup_gib=[0-9.]+" "$MON" | cut -d= -f2 | sort -n | tail -1)
MIN_AVAIL=$(grep -oE "avail_gib=[0-9.]+" "$MON" | cut -d= -f2 | sort -n | head -1)
STEPS_N=$(grep -cE "Step [0-9]+:" "$LOG" || true)
LOSS=$(grep -E "Step [0-9]+:" "$LOG" | tail -1 | sed -n "s/.*loss=\\([0-9.]*\\).*/\\1/p")
RATE=$(grep -E "Progress on:" "$LOG" | tail -3)
OOM=$(dmesg -T 2>/dev/null | grep -iE "oom|killed process" | tail -3 || true)

if grep -qiE "out of memory|RESOURCE_EXHAUSTED|CUDA out of memory" "$LOG"; then TAG=GPU_OOM
elif [ "$EC" -eq 137 ] || [ "$EC" -eq 9 ]; then TAG=HOST_KILLED
elif [ "$EC" -ne 0 ]; then TAG=FAIL
elif [ "${STEPS_N:-0}" -lt 1 ]; then TAG=NO_STEP
else TAG=OK
fi

# WRITE RESULT FIRST (before heavy cleanup)
{
  echo "result=$TAG exit=$EC steps_logged=$STEPS_N peak_vram_MiB=${PEAK_GPU:-0} peak_cgroup_GiB=${PEAK_CG:-0} min_avail_GiB=${MIN_AVAIL:-?} last_loss=${LOSS:-}"
  [ -n "$RATE" ] && echo "$RATE"
  [ -n "$OOM" ] && { echo "dmesg_oom:"; echo "$OOM"; }
  if [ "$TAG" != "OK" ]; then echo "--- tail ---"; tail -40 "$LOG"; fi
} | tee -a "$RESULT"

# then free disk
find "$CKPT_ROOT" -maxdepth 1 -type d -regex ".*/[0-9]+" -exec rm -rf {} + 2>/dev/null || true
df -h /root/autodl-tmp | tail -1 | tee -a "$RESULT"
echo "DONE_ONE $B $TAG $(date -Is)" | tee -a "$RESULT"
[ "$TAG" = "OK" ]

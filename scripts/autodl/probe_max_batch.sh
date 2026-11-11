#!/bin/bash
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi-cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-home
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
cd /root/autodl-tmp/lehome_solution

CKPT_ROOT=outputs/checkpoints/pi_modified_bc_rl/sim_rl
RESULT=/root/autodl-tmp/max_batch_probe.md
echo "# A800 max-batch probe $(date -Is)" > "$RESULT"
echo "baseline b32: Step0 loss=1.8687, peak~15371MiB, step0~162s/it (JIT), later ~15-29s/it avg" >> "$RESULT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$RESULT"

cleanup_ckpts() {
  find "$CKPT_ROOT" -maxdepth 1 -type d -regex ".*/[0-9]+" -exec rm -rf {} +
  df -h /root/autodl-tmp | tail -1 | tee -a "$RESULT"
}

reset_state() {
  rm -f "$CKPT_ROOT/_pipeline_config_iter0.pkl"
  .venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path("outputs/checkpoints/pi_modified_bc_rl/sim_rl/pipeline_state.json")
st = json.loads(p.read_text()) if p.exists() else {}
st.update({
  "iteration": 0,
  "phase": "warmup",
  "current_train_steps": 0,
  "latest_checkpoint_step": None,
})
p.write_text(json.dumps(st, indent=2) + "\n")
print("state_reset", st.get("current_train_steps"))
PY
}

.venv/bin/python -c "from torchcodec.decoders import VideoDecoder; print('torchcodec_ok')" | tee -a "$RESULT"

cleanup_ckpts
MAX_OK=32
for B in 64 96 128 192; do
  echo "" | tee -a "$RESULT"
  echo "## batch_size=$B $(date -Is)" | tee -a "$RESULT"
  cleanup_ckpts
  reset_state
  LOG=/root/autodl-tmp/probe_b${B}.log
  PEAK_LOG=/root/autodl-tmp/probe_b${B}_smi.log
  rm -f "$LOG" "$PEAK_LOG"
  (
    while true; do
      nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits >> "$PEAK_LOG"
      sleep 2
    done
  ) &
  SMI_PID=$!
  set +e
  .venv/bin/python -u scripts/run_rl_pipeline.py \
    --config=configs/rl_pipeline_sim.yaml \
    --train_only \
    --steps_per_iteration 2 \
    --batch_size "$B" \
    > "$LOG" 2>&1
  EC=$?
  set -e
  kill "$SMI_PID" 2>/dev/null || true
  wait "$SMI_PID" 2>/dev/null || true
  PEAK=$(awk -F',' 'BEGIN{m=0} {gsub(/ /,\"\"); v=$1+0; if(v>m)m=v} END{print m}' "$PEAK_LOG" 2>/dev/null || echo 0)
  STEPS=$(grep -E "Step [0-9]+:" "$LOG" | wc -l)
  RATE=$(grep -E "rate:[0-9]" "$LOG" | tail -3)
  LOSS=$(grep -E "Step [0-9]+:" "$LOG" | tail -1 | sed -n 's/.*loss=\([0-9.]*\).*/\1/p')
  if grep -qiE "out of memory|RESOURCE_EXHAUSTED|CUDA out of memory" "$LOG"; then
    TAG=OOM
  elif [ "$EC" -eq 137 ] || [ "$EC" -eq 9 ]; then
    TAG=KILLED
  elif [ "$EC" -ne 0 ]; then
    TAG=FAIL
  elif [ "$STEPS" -lt 1 ]; then
    TAG=NO_STEP
  else
    TAG=OK
  fi
  echo "result=$TAG exit=$EC steps_logged=$STEPS peak_vram_MiB=$PEAK last_loss=$LOSS" | tee -a "$RESULT"
  echo "$RATE" | tee -a "$RESULT"
  if [ "$TAG" != "OK" ]; then
    echo "--- tail ---" | tee -a "$RESULT"
    tail -40 "$LOG" | tee -a "$RESULT"
    echo "MAX_OK=$MAX_OK (failed at $B: $TAG)" | tee -a "$RESULT"
    cleanup_ckpts
    echo "DONE_PROBE $(date -Is)" | tee -a "$RESULT"
    exit 0
  fi
  MAX_OK=$B
  cleanup_ckpts
done
echo "MAX_OK=$MAX_OK" | tee -a "$RESULT"
echo "DONE_PROBE $(date -Is)" | tee -a "$RESULT"

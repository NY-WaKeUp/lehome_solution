#!/bin/bash
set -uo pipefail
exec 9>/root/autodl-tmp/smoke.once.lock
flock -n 9 || { echo already_running; exit 0; }
export PATH="$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi-cache
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-home
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
cd /root/autodl-tmp/lehome_solution
RESULT=/root/autodl-tmp/batch_smoke_results.md
echo "# A800 batch smoke $(date -Is)" > "$RESULT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a "$RESULT"
MAX_OK=""
for B in 192 128 96 64; do
  echo "" | tee -a "$RESULT"
  echo "## batch_size=$B $(date -Is)" | tee -a "$RESULT"
  LOG=/root/autodl-tmp/smoke_b${B}.log
  rm -f "$LOG"
  if .venv/bin/python -u scripts/run_rl_pipeline.py       --config=configs/rl_pipeline_sim.yaml       --train_only       --steps_per_iteration 3       --batch_size "$B"       > "$LOG" 2>&1; then
    echo "OK batch=$B" | tee -a "$RESULT"
    if [ -z "$MAX_OK" ]; then MAX_OK=$B; fi
  else
    ec=$?
    echo "FAIL batch=$B exit=$ec" | tee -a "$RESULT"
    tail -80 "$LOG" | tee -a "$RESULT"
    continue
  fi
  grep -E "Step |loss=|steps/s|sec/step|samples/s" "$LOG" | tail -30 | tee -a "$RESULT" || true
  nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader | tee -a "$RESULT"
done
echo "MAX_OK=${MAX_OK:-none}" | tee -a "$RESULT"
echo "DONE_SMOKE $(date -Is)" | tee -a "$RESULT"

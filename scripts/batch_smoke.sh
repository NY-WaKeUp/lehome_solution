#!/bin/bash
# Max global-batch smoke on 1x A800. Records peak VRAM / step time / OOM.
set -u
export PATH="$HOME/.local/bin:$PATH"
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi-cache
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
cd /root/autodl-tmp/lehome_solution
mkdir -p logs outputs
RESULT=logs/batch_smoke_results.tsv
echo -e "batch\tstatus\tpeak_mib\tstep_s\tnote" > "$RESULT"

# Disable saving / HF / wandb for smoke; few steps only.
# Prefer train_only with tiny steps_per_iteration.
run_one() {
  local B=$1
  local LOG=logs/batch_smoke_b${B}.log
  echo "===== SMOKE batch=$B $(date -Is) =====" | tee "$LOG"
  # Clear GPU before each try
  nvidia-smi --gpu-reset 2>/dev/null || true
  sleep 2
  # Monitor peak VRAM in background
  local MON=logs/batch_smoke_b${B}.vram
  ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 1; done ) > "$MON" &
  local MPID=$!
  # Run few steps; disable checkpoint spam via config overrides if supported
  set +e
  .venv/bin/python -u scripts/run_rl_pipeline.py \
    --config=configs/rl_pipeline_sim.yaml \
    --train_only \
    --steps_per_iteration 3 \
    --batch_size "$B" \
    --num_iterations 1 \
    --save_interval 999999 \
    --wandb false \
    --hf_sync false \
    >>"$LOG" 2>&1
  local RC=$?
  set -e
  kill $MPID 2>/dev/null || true
  wait $MPID 2>/dev/null || true
  local PEAK=$(awk "BEGIN{m=0} {if(\$1+0>m)m=\$1+0} END{print m}" "$MON")
  # Parse step timing if present
  local STEP=$(grep -Eo "([0-9]+\.[0-9]+)s/it|step_time[=:][ ]*[0-9.]+|samples/s[=:][ ]*[0-9.]+" "$LOG" | tail -5 | tr "\n" " ")
  local STATUS=FAIL
  local NOTE="rc=$RC"
  if grep -qiE "out of memory|RESOURCE_EXHAUSTED|OOM" "$LOG"; then
    STATUS=OOM
    NOTE="oom peak=${PEAK}MiB"
  elif grep -qiE "Step 0|step 0|loss=|train_loss" "$LOG" && [ $RC -eq 0 ]; then
    STATUS=OK
    NOTE="ok"
  elif grep -qiE "Step 0|step 0|loss=" "$LOG"; then
    STATUS=PARTIAL
    NOTE="saw_step rc=$RC"
  fi
  echo -e "${B}\t${STATUS}\t${PEAK}\t${STEP}\t${NOTE}" | tee -a "$RESULT"
  # cleanup failed checkpoints / iter outputs to save disk
  rm -rf outputs/checkpoints/pi_modified_bc_rl/sim_rl/iter* 2>/dev/null || true
  find outputs -name "*.orbax-checkpoint*" -prune -exec rm -rf {} + 2>/dev/null || true
  echo "RESULT batch=$B status=$STATUS peak=${PEAK}MiB"
}

# Sweep high -> low
for B in 192 128 96 64 48 32; do
  run_one "$B"
  # stop early once we found a working max (first OK from high to low)
  if grep -q "^${B}	OK" "$RESULT"; then
    echo "MAX_STABLE_BATCH=$B" | tee -a "$RESULT"
    break
  fi
done
cat "$RESULT"
df -h /root/autodl-tmp

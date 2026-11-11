#!/bin/bash
# Auto-clean autodl-tmp so orbax ckpt writes never die on ENOSPC.
# Policy: when Avail < THRESH_GB, delete orbax *.tmp* first, then older
# numbered checkpoints (keep latest complete digit-only dir per exp).
set -u
THRESH_GB="${DISK_GUARD_THRESH_GB:-15}"
INTERVAL_SEC="${DISK_GUARD_INTERVAL_SEC:-30}"
ROOT="/root/autodl-tmp"
CKPT_ROOT="$ROOT/lehome_solution/outputs/checkpoints"
LOG="$ROOT/ablation/disk_guard.log"
mkdir -p "$ROOT/ablation"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] disk_guard start thresh=${THRESH_GB}G interval=${INTERVAL_SEC}s pid=$$"

avail_gb() {
  df -BG "$ROOT" | awk 'NR==2 {gsub(/G/,"",$4); print $4}'
}

clean_once() {
  local avail
  avail=$(avail_gb)
  if [ "$avail" -ge "$THRESH_GB" ]; then
    return 0
  fi
  echo "[$(date -Is)] LOW_DISK avail=${avail}G < ${THRESH_GB}G — cleaning"

  # 1) truncated orbax temps anywhere under checkpoints
  find "$CKPT_ROOT" -maxdepth 5 \( -type d -name "*.tmp*" -o -type d -name "*orbax-checkpoint-tmp*" \) -print -exec rm -rf {} + 2>/dev/null || true
  find "$CKPT_ROOT" -maxdepth 6 \( -name "*.tmp" -o -name "*_tmp_*" \) -print -exec rm -rf {} + 2>/dev/null || true

  # 2) per-exp: keep only highest complete numbered step (digit-only + params/)
  if [ -d "$CKPT_ROOT" ]; then
    find "$CKPT_ROOT" -mindepth 2 -maxdepth 2 -type d 2>/dev/null | while read -r exp; do
      mapfile -t steps < <(find "$exp" -maxdepth 1 -type d -regex ".*/[0-9]+" | while read -r d; do
        base=$(basename "$d")
        if [ -d "$d/params" ]; then echo "$base"; fi
      done | sort -n)
      n=${#steps[@]}
      if [ "$n" -le 1 ]; then
        continue
      fi
      # drop all but last
      for ((i=0; i<n-1; i++)); do
        echo "  rm old ckpt $exp/${steps[i]}"
        rm -rf "$exp/${steps[i]}"
      done
    done
  fi

  # 3) nonessential bulky logs / profiles if still tight
  avail=$(avail_gb)
  if [ "$avail" -lt "$THRESH_GB" ]; then
    find "$CKPT_ROOT" -name "memory_profile.prof" -delete 2>/dev/null || true
    find "$ROOT/lehome_solution/logs" -type f -name "*.log" -size +50M -delete 2>/dev/null || true
  fi

  echo "[$(date -Is)] after_clean avail=$(avail_gb)G"
  df -h "$ROOT" | tail -1
}

while true; do
  clean_once
  sleep "$INTERVAL_SEC"
done

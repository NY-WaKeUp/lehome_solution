#!/usr/bin/env bash
# Sidecar offload for AutoDL AWR: sync ONLY finalized Orbax checkpoints + small
# artifacts to Hugging Face Hub (this repo's HFSync destination), without
# touching in-flight *.orbax-checkpoint-tmp* and without stopping train.py.
#
# Why sidecar (not enabling pipeline hf_sync mid-run):
#   configs/rl_pipeline_sim.yaml has hf_sync_enabled=false / hf_model_repo=null.
#   Flipping that requires a pipeline restart. This watcher attaches to a live run.
#
# Finalize rule (must match disk_guard / Orbax ops docs):
#   digit-only dir N/ AND N/params/ exists  => safe to sync
#   *.orbax-checkpoint-tmp*                 => NEVER sync / NEVER delete here
#
# Usage (AutoDL):
#   export HF_TOKEN=hf_xxx
#   export HF_MODEL_REPO=YourUser/awr-b192-vanilla-ckpt   # private model repo
#   # optional: export HF_ENDPOINT=https://hf-mirror.com
#   nohup bash /root/autodl-tmp/lehome_solution/scripts/autodl/hfsync_finalized_ckpt.sh watch \
#     >> /root/autodl-tmp/ablation/hfsync_ckpt.log 2>&1 &
#
# One-shot / dry-run / list:
#   bash scripts/autodl/hfsync_finalized_ckpt.sh once
#   DRY_RUN=1 bash scripts/autodl/hfsync_finalized_ckpt.sh once
#   bash scripts/autodl/hfsync_finalized_ckpt.sh list
set -euo pipefail

MODE="${1:-watch}"
EXP_DIR="${EXP_DIR:-/root/autodl-tmp/lehome_solution/outputs/checkpoints/pi_modified_bc_rl/sim_rl_awr_sample_b192_vanilla}"
TB_DIR="${TB_DIR:-/root/tf-logs/sim_rl_awr_sample_b192_vanilla}"
STATE_FILE="${STATE_FILE:-/root/autodl-tmp/ablation/hfsync_ckpt_state.txt}"
LOG_PREFIX="${LOG_PREFIX:-[hfsync_ckpt]}"
POLL_SEC="${POLL_SEC:-60}"
DRY_RUN="${DRY_RUN:-0}"
# 1 = full tree (params+assets+train_state) for resume; 0 = params+assets only.
INCLUDE_TRAIN_STATE="${INCLUDE_TRAIN_STATE:-1}"
ALSO_LOGS="${ALSO_LOGS:-1}"
ALSO_TB="${ALSO_TB:-1}"
HF_BIN="${HF_BIN:-/root/autodl-tmp/lehome_solution/.venv/bin/hf}"
REPO_TYPE="${REPO_TYPE:-model}"

mkdir -p "$(dirname "$STATE_FILE")"
touch "$STATE_FILE"

log() { echo "$(date -Is) ${LOG_PREFIX} $*"; }

require_hf_creds() {
  if [[ -z "${HF_TOKEN:-}" ]]; then
    log "ERROR: HF_TOKEN unset. Create a write token and: export HF_TOKEN=..."
    exit 2
  fi
  if [[ -z "${HF_MODEL_REPO:-}" ]]; then
    log "ERROR: HF_MODEL_REPO unset. Example: export HF_MODEL_REPO=user/awr-b192-vanilla-ckpt"
    exit 2
  fi
  if [[ ! -x "$HF_BIN" ]]; then
    log "ERROR: hf CLI not executable at $HF_BIN"
    exit 2
  fi
}

already_synced() {
  local step="$1"
  grep -qx "ckpt:${step}" "$STATE_FILE"
}

mark_synced() {
  local step="$1"
  echo "ckpt:${step}" >>"$STATE_FILE"
}

is_finalized_ckpt() {
  local d="$1"
  local base
  base="$(basename "$d")"
  [[ "$base" =~ ^[0-9]+$ ]] || return 1
  [[ -d "$d/params" ]] || return 1
  [[ "$base" != *orbax-checkpoint-tmp* ]] || return 1
  return 0
}

list_finalized_steps() {
  local d base
  if [[ ! -d "$EXP_DIR" ]]; then
    return 0
  fi
  find "$EXP_DIR" -maxdepth 1 -mindepth 1 -type d | while read -r d; do
    base="$(basename "$d")"
    if is_finalized_ckpt "$d"; then
      echo "$base"
    fi
  done | sort -n
}

# Stage on the SAME filesystem as EXP_DIR so cp -al can hardlink (no 12G clone).
# Never stage under /tmp (often overlay) — that forces full copy and can OOD.
make_stage_dir() {
  local step="$1"
  mktemp -d "${EXP_DIR}/.hfsync_stage_${step}.XXXXXX"
}

stage_tree_for_step() {
  local step="$1"
  local src="${EXP_DIR}/${step}"
  local stage="$2"
  local dest="${stage}/${step}"

  mkdir -p "$dest"
  if [[ "$INCLUDE_TRAIN_STATE" == "1" ]]; then
    cp -al "${src}/." "${dest}/" 2>/dev/null || cp -a "${src}/." "${dest}/"
  else
    if [[ -d "${src}/params" ]]; then
      cp -al "${src}/params" "${dest}/params" 2>/dev/null || cp -a "${src}/params" "${dest}/params"
    fi
    if [[ -d "${src}/assets" ]]; then
      cp -al "${src}/assets" "${dest}/assets" 2>/dev/null || cp -a "${src}/assets" "${dest}/assets"
    fi
  fi
}

upload_staged() {
  local stage="$1"
  local step="$2"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN would: hf upload-large-folder $HF_MODEL_REPO $stage"
    du -sh "$stage" "${stage}/${step}" 2>/dev/null || true
    find "$stage" -maxdepth 3 -type d | head -40
    return 0
  fi
  log "uploading step $step (~$(du -sh "${stage}/${step}" | awk '{print $1}')) -> $HF_MODEL_REPO ..."
  "$HF_BIN" upload-large-folder "$HF_MODEL_REPO" "$stage" \
    --repo-type "$REPO_TYPE" \
    --private \
    --token "$HF_TOKEN" \
    --num-workers 2 \
    --exclude "**/*.orbax-checkpoint-tmp*" \
    --exclude "**/.*orbax*"
  log "uploaded step $step to $HF_MODEL_REPO"
}

upload_small_artifacts() {
  local tag="$1"
  local logs_dir="${EXP_DIR}/logs"
  if [[ "$ALSO_LOGS" == "1" && -d "$logs_dir" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      log "DRY_RUN would upload logs/ as artifacts/${tag}/logs"
    else
      "$HF_BIN" upload "$HF_MODEL_REPO" "$logs_dir" "artifacts/${tag}/logs" \
        --repo-type "$REPO_TYPE" \
        --token "$HF_TOKEN" \
        --commit-message "logs ${tag}"
      log "uploaded logs for $tag"
    fi
  fi
  if [[ "$ALSO_TB" == "1" && -d "$TB_DIR" ]]; then
    if [[ "$DRY_RUN" == "1" ]]; then
      log "DRY_RUN would upload TB as artifacts/${tag}/tb"
    else
      "$HF_BIN" upload "$HF_MODEL_REPO" "$TB_DIR" "artifacts/${tag}/tb" \
        --repo-type "$REPO_TYPE" \
        --token "$HF_TOKEN" \
        --commit-message "tb ${tag}"
      log "uploaded TB for $tag"
    fi
  fi
}

sync_once() {
  local step stage
  local any=0
  while read -r step; do
    [[ -n "$step" ]] || continue
    any=1
    if already_synced "$step"; then
      log "skip step $step (already in $STATE_FILE)"
      continue
    fi
    log "finalized ckpt detected: ${EXP_DIR}/${step}"
    stage="$(make_stage_dir "$step")"
    stage_tree_for_step "$step" "$stage"
    upload_staged "$stage" "$step"
    rm -rf "$stage"
    upload_small_artifacts "step_${step}"
    if [[ "$DRY_RUN" != "1" ]]; then
      mark_synced "$step"
    fi
    log "done step $step"
  done < <(list_finalized_steps)

  if [[ "$any" -eq 0 ]]; then
    log "no finalized numbered ckpt yet under $EXP_DIR (tmp dirs ignored)"
  fi
}

case "$MODE" in
  once)
    require_hf_creds
    sync_once
    ;;
  watch)
    require_hf_creds
    log "watch start EXP_DIR=$EXP_DIR REPO=$HF_MODEL_REPO POLL=${POLL_SEC}s INCLUDE_TRAIN_STATE=$INCLUDE_TRAIN_STATE DRY_RUN=$DRY_RUN"
    log "policy: sync digit dirs with params/ only; never touch *orbax-checkpoint-tmp*"
    while true; do
      sync_once
      sleep "$POLL_SEC"
    done
    ;;
  list)
    log "finalized steps under $EXP_DIR:"
    list_finalized_steps | sed 's/^/  /' || true
    log "state file: $STATE_FILE"
    while read -r step; do
      if already_synced "$step"; then
        echo "  $step SYNCED"
      else
        echo "  $step PENDING"
      fi
    done < <(list_finalized_steps)
    ;;
  *)
    echo "usage: $0 {watch|once|list}" >&2
    exit 1
    ;;
esac

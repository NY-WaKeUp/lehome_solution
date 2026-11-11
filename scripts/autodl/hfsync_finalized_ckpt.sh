#!/usr/bin/env bash
# Sidecar offload for AutoDL AWR: sync ONLY finalized Orbax checkpoints + small
# artifacts to Hugging Face Hub, without touching in-flight
# *.orbax-checkpoint-tmp* and without stopping train.py.
#
# Why sidecar (not enabling pipeline hf_sync mid-run):
#   configs/rl_pipeline_sim.yaml has hf_sync_enabled=false / hf_model_repo=null.
#   Flipping that requires a pipeline restart. This watcher attaches to a live run.
#
# AutoDL disk ceiling (hard constraint; no expand):
#   /                system disk ~30G (image-persistent) — do NOT put ckpt here
#   /root/autodl-tmp data disk  ~50G (fast, not in image) — ONLY place for ckpt
# Under 50G, HF upload+verify then prune local is the PRIMARY retention strategy.
#
# Finalize rule (must match disk_guard / Orbax ops docs):
#   digit-only dir N/ AND N/params/ exists  => safe to sync / eligible to prune
#   *.orbax-checkpoint-tmp*                 => NEVER sync / NEVER delete here
#
# Prune policy (default on):
#   upload -> Hub verify (params[, train_state]) -> delete older local digit dirs
#   KEEP_LAST=1  keep newest local finalized step for instant resume (recommended)
#   KEEP_LAST=0  delete all local finalized after verify; resume via `download`
#
# Usage (AutoDL):
#   export HF_TOKEN=hf_xxx
#   export HF_MODEL_REPO=YourUser/awr-b192-vanilla-ckpt   # private model repo
#   # optional: export HF_ENDPOINT=https://hf-mirror.com
#   # optional: KEEP_LAST=1 PRUNE_LOCAL=1 (defaults under 50G)
#   nohup bash /root/autodl-tmp/lehome_solution/scripts/autodl/hfsync_finalized_ckpt.sh watch \
#     >> /root/autodl-tmp/ablation/hfsync_ckpt.log 2>&1 &
#
# Modes:
#   bash scripts/autodl/hfsync_finalized_ckpt.sh once
#   DRY_RUN=1 bash scripts/autodl/hfsync_finalized_ckpt.sh once
#   bash scripts/autodl/hfsync_finalized_ckpt.sh list
#   bash scripts/autodl/hfsync_finalized_ckpt.sh download 500
set -euo pipefail

MODE="${1:-watch}"
STEP_ARG="${2:-}"
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
# After Hub verify: drop local finalized steps outside the newest KEEP_LAST.
PRUNE_LOCAL="${PRUNE_LOCAL:-1}"
KEEP_LAST="${KEEP_LAST:-1}"
HF_BIN="${HF_BIN:-/root/autodl-tmp/lehome_solution/.venv/bin/hf}"
HF_PY="${HF_PY:-/root/autodl-tmp/lehome_solution/.venv/bin/python}"
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
  if [[ ! -x "$HF_PY" ]]; then
    log "ERROR: python not executable at $HF_PY"
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

# Fail loud if Hub lacks step/params (and train_state when required).
verify_step_on_hub() {
  local step="$1"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN skip Hub verify for step $step"
    return 0
  fi
  log "verifying Hub path ${step}/params on $HF_MODEL_REPO ..."
  HF_TOKEN="$HF_TOKEN" HF_MODEL_REPO="$HF_MODEL_REPO" REPO_TYPE="$REPO_TYPE" \
    STEP="$step" INCLUDE_TRAIN_STATE="$INCLUDE_TRAIN_STATE" \
    "$HF_PY" - <<'PY'
import os
import sys
from huggingface_hub import HfApi

repo = os.environ["HF_MODEL_REPO"]
step = os.environ["STEP"]
repo_type = os.environ["REPO_TYPE"]
token = os.environ["HF_TOKEN"]
need_train = os.environ.get("INCLUDE_TRAIN_STATE", "1") == "1"

need = [f"{step}/params"]
if need_train:
    need.append(f"{step}/train_state")

api = HfApi(token=token)
infos = api.get_paths_info(repo, need, repo_type=repo_type)
found = set()
for p in infos:
    path = getattr(p, "path", None)
    if path is not None:
        found.add(path)
missing = [p for p in need if p not in found]
if missing:
    print(f"ERROR: Hub missing paths for step {step}: {missing}", file=sys.stderr)
    print(f"found={sorted(found)}", file=sys.stderr)
    sys.exit(3)
print(f"ok: Hub has {sorted(found)}")
PY
  log "Hub verify ok for step $step"
}

# Delete finalized local digit dirs outside the newest KEEP_LAST window.
# Requires state sync + Hub verify. Never touches *.orbax-checkpoint-tmp*.
prune_older_local_ckpts() {
  local -a steps=()
  local n keep_from i step path
  if [[ "$PRUNE_LOCAL" != "1" ]]; then
    log "PRUNE_LOCAL=0 — skip local prune"
    return 0
  fi
  if [[ ! "$KEEP_LAST" =~ ^[0-9]+$ ]]; then
    log "ERROR: KEEP_LAST must be a non-negative integer (got: $KEEP_LAST)"
    exit 2
  fi
  mapfile -t steps < <(list_finalized_steps)
  n="${#steps[@]}"
  if [[ "$n" -eq 0 ]]; then
    log "local prune: no finalized steps"
    return 0
  fi
  if [[ "$n" -le "$KEEP_LAST" ]]; then
    log "local prune: ${n} finalized step(s) <= KEEP_LAST=${KEEP_LAST}; nothing to delete"
    return 0
  fi
  keep_from=$((n - KEEP_LAST))
  for ((i = 0; i < keep_from; i++)); do
    step="${steps[i]}"
    path="${EXP_DIR}/${step}"
    if [[ ! "$step" =~ ^[0-9]+$ ]]; then
      log "ERROR: refuse prune non-digit path: $path"
      exit 2
    fi
    if [[ "$path" == *orbax-checkpoint-tmp* ]]; then
      log "ERROR: refuse prune tmp-looking path: $path"
      exit 2
    fi
    if ! already_synced "$step"; then
      log "local prune: keep ${step} (not in state / not uploaded yet)"
      continue
    fi
    verify_step_on_hub "$step"
    if [[ "$DRY_RUN" == "1" ]]; then
      log "DRY_RUN would rm -rf $path (Hub verified; KEEP_LAST=$KEEP_LAST)"
      continue
    fi
    log "local prune: rm -rf $path (Hub verified; KEEP_LAST=$KEEP_LAST)"
    rm -rf "$path"
  done
}

# Pull a finalized step from Hub into EXP_DIR for resume when local was pruned.
download_step_from_hub() {
  local step="$1"
  local dest="${EXP_DIR}/${step}"
  if [[ ! "$step" =~ ^[0-9]+$ ]]; then
    log "ERROR: download step must be digits (got: $step)"
    exit 2
  fi
  require_hf_creds
  verify_step_on_hub "$step"
  if [[ -d "$dest/params" ]]; then
    log "ERROR: local finalized ${dest} already exists; refuse overwrite"
    exit 2
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN would: hf download $HF_MODEL_REPO --include ${step}/** --local-dir $EXP_DIR"
    return 0
  fi
  mkdir -p "$EXP_DIR"
  log "downloading Hub ${step}/ -> $EXP_DIR ..."
  "$HF_BIN" download "$HF_MODEL_REPO" \
    --repo-type "$REPO_TYPE" \
    --token "$HF_TOKEN" \
    --include "${step}/**" \
    --local-dir "$EXP_DIR"
  if ! is_finalized_ckpt "$dest"; then
    log "ERROR: after download, ${dest} is not a finalized ckpt (missing params/)"
    exit 3
  fi
  log "download ok: $dest"
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
      verify_step_on_hub "$step"
      mark_synced "$step"
    else
      log "DRY_RUN skip mark_synced / Hub verify for step $step"
    fi
    log "done step $step"
  done < <(list_finalized_steps)

  if [[ "$any" -eq 0 ]]; then
    log "no finalized numbered ckpt yet under $EXP_DIR (tmp dirs ignored)"
  fi

  # Prune after this poll's uploads so Hub verify for new steps is already done.
  prune_older_local_ckpts
}

case "$MODE" in
  once)
    require_hf_creds
    sync_once
    ;;
  watch)
    require_hf_creds
    log "watch start EXP_DIR=$EXP_DIR REPO=$HF_MODEL_REPO POLL=${POLL_SEC}s INCLUDE_TRAIN_STATE=$INCLUDE_TRAIN_STATE DRY_RUN=$DRY_RUN PRUNE_LOCAL=$PRUNE_LOCAL KEEP_LAST=$KEEP_LAST"
    log "policy: sync digit dirs with params/ only; never touch *orbax-checkpoint-tmp*; prune only after Hub verify"
    while true; do
      sync_once
      sleep "$POLL_SEC"
    done
    ;;
  list)
    log "finalized steps under $EXP_DIR:"
    list_finalized_steps | sed 's/^/  /' || true
    log "state file: $STATE_FILE"
    log "prune: PRUNE_LOCAL=$PRUNE_LOCAL KEEP_LAST=$KEEP_LAST"
    while read -r step; do
      if already_synced "$step"; then
        echo "  $step SYNCED"
      else
        echo "  $step PENDING"
      fi
    done < <(list_finalized_steps)
    ;;
  download)
    if [[ -z "$STEP_ARG" ]]; then
      echo "usage: $0 download <step>" >&2
      exit 1
    fi
    download_step_from_hub "$STEP_ARG"
    ;;
  *)
    echo "usage: $0 {watch|once|list|download <step>}" >&2
    exit 1
    ;;
esac

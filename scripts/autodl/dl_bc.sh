#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf-home
mkdir -p "$HF_HOME"
cd /root/autodl-tmp/lehome_solution
echo START $(date -Is)
.venv/bin/python scripts/download_hf_assets.py --four-types
echo DL_BC_OK $(date -Is)
du -sh data/lehome_challenge_merged
df -h /root/autodl-tmp

#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR=/root/autodl-tmp/uv-cache
export UV_PROJECT_ENVIRONMENT=/root/autodl-tmp/lehome_solution/.venv
export UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
export http_proxy=http://127.0.0.1:17897
export https_proxy=http://127.0.0.1:17897
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy
export no_proxy=mirrors.aliyun.com,127.0.0.1,localhost
export NO_PROXY=$no_proxy
cd /root/autodl-tmp/lehome_solution
rm -rf .venv
echo "START $(date -Is)"
uv sync --extra dev
test -x .venv/bin/python
echo UV_SYNC_OK
.venv/bin/python -c "import torch,jax; print(\"torch\", torch.__version__, torch.cuda.is_available()); print(\"jax\", jax.__version__, jax.devices())"
du -sh .venv "$UV_CACHE_DIR"
df -h /root/autodl-tmp
# free space after install
uv cache prune -y || true
df -h /root/autodl-tmp

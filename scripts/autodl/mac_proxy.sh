#!/usr/bin/env bash
# Use Mac FlClash via SSH reverse tunnel. Do NOT use AutoDL 10.7.26.129:7890.
# On Mac first:
#   ssh -f -N -o ExitOnForwardFailure=yes -o ControlMaster=no -o ControlPath=none \
#     -R 127.0.0.1:17897:127.0.0.1:7897 -p <PORT> -i ~/.ssh/id_ed25519 root@<HOST>
# Or: Host autodl-a800 with RemoteForward 127.0.0.1:17897 127.0.0.1:7897 then ssh -f -N autodl-a800

set -euo pipefail

REMOTE_PORT="${MAC_PROXY_REMOTE_PORT:-17897}"
PROXY_URL="http://127.0.0.1:${REMOTE_PORT}"

cmd="${1:-on}"

case "$cmd" in
  on)
    export http_proxy="$PROXY_URL"
    export https_proxy="$PROXY_URL"
    export HTTP_PROXY="$PROXY_URL"
    export HTTPS_PROXY="$PROXY_URL"
    export ALL_PROXY="$PROXY_URL"
    export all_proxy="$PROXY_URL"
    export no_proxy="localhost,127.0.0.1,::1"
    export NO_PROXY="$no_proxy"
    git config --global http.proxy "$PROXY_URL"
    git config --global https.proxy "$PROXY_URL"
    echo "mac_proxy ON -> $PROXY_URL (env + git)"
    ;;
  off)
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
    git config --global --unset-all http.proxy || true
    git config --global --unset-all https.proxy || true
    echo "mac_proxy OFF (env + git proxy cleared)"
    ;;
  status)
    echo "env:"; env | grep -i proxy || echo "(none)"
    echo "git:"; git config --global --get-regexp "http.*proxy" || echo "(none)"
    if (echo >/dev/tcp/127.0.0.1/"$REMOTE_PORT") >/dev/null 2>&1; then
      echo "tunnel: 127.0.0.1:$REMOTE_PORT OPEN"
    else
      echo "tunnel: 127.0.0.1:$REMOTE_PORT CLOSED"
    fi
    ;;
  test)
    export http_proxy="$PROXY_URL" https_proxy="$PROXY_URL"
    curl -sS -o /dev/null -w "curl_github=%{http_code} time=%{time_total}\n" --connect-timeout 15 --max-time 45 https://github.com
    git -c http.proxy="$PROXY_URL" -c https.proxy="$PROXY_URL" ls-remote https://github.com/IliaLarchenko/lehome_solution.git HEAD
    ;;
  *)
    echo "usage: source scripts/autodl/mac_proxy.sh {on|off|status|test}"
    exit 2
    ;;
esac

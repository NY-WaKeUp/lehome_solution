# AutoDL ops scripts

Operational scripts used on AutoDL (`/root/autodl-tmp`) for A800 ablation,
MAX_OK batch probing, disk guard, and related helpers.

Canonical copies live here under `scripts/autodl/`. On AutoDL, paths like
`/root/autodl-tmp/*.sh` are symlinks into this directory so absolute-path
launchers keep working without interrupting long-running jobs.

Paths inside scripts often assume `/root/autodl-tmp/lehome_solution` and
`/root/autodl-tmp` locks/logs; adjust when reusing elsewhere.

## Network / GitHub

- Default path: Mac FlClash (usually http://127.0.0.1:7897) exposed on AutoDL as http://127.0.0.1:17897 via SSH reverse tunnel (`ssh -R`).
- On Mac (keep this process alive):

```bash
ssh -f -N -o ExitOnForwardFailure=yes -o ControlMaster=no -o ControlPath=none \
  -R 127.0.0.1:17897:127.0.0.1:7897 -p 35794 -i ~/.ssh/id_ed25519 \
  root@connect.nma1.seetacloud.com
```

- On AutoDL:

```bash
source /root/autodl-tmp/lehome_solution/scripts/autodl/mac_proxy.sh on
bash /root/autodl-tmp/lehome_solution/scripts/autodl/mac_proxy.sh test
```

- Do **not** use AutoDL side `10.7.26.129:7890` as the default (timeouts / fails for GitHub).

## Checkpoint offload (HFSync sidecar)

There is **no** standalone `hfsync` binary. In this repo, "hfsync" means the
HuggingFace sync path (`HFSyncClient` / `hf_sync_daemon.py`). The current AWR
sim config has `hf_sync_enabled: false` — do **not** restart a live train just
to flip it. Use the sidecar instead:

```bash
export HF_TOKEN=hf_xxx
export HF_MODEL_REPO=YourUser/awr-b192-vanilla-ckpt
# optional: export HF_ENDPOINT=https://hf-mirror.com
nohup bash /root/autodl-tmp/lehome_solution/scripts/autodl/hfsync_finalized_ckpt.sh watch \
  >> /root/autodl-tmp/ablation/hfsync_ckpt.log 2>&1 &
```

Policy: only sync finalized digit dirs with `params/`; never touch
`*.orbax-checkpoint-tmp*`. Safe alongside `disk_guard.sh`.

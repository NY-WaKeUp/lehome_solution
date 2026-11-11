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

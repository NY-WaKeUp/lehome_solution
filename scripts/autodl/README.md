# AutoDL ops scripts

Operational scripts used on AutoDL (`/root/autodl-tmp`) for A800 ablation,
MAX_OK batch probing, disk guard, and related helpers.

Canonical copies live here under `scripts/autodl/`. On AutoDL, paths like
`/root/autodl-tmp/*.sh` are symlinks into this directory so absolute-path
launchers keep working without interrupting long-running jobs.

Paths inside scripts often assume `/root/autodl-tmp/lehome_solution` and
`/root/autodl-tmp` locks/logs; adjust when reusing elsewhere.

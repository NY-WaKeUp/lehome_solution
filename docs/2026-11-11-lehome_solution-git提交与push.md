# 2026-11-11 lehome_solution fork 提交与 push

## 结论

以 **Aserver** `/home/ningyu/lehome_solution` 为权威侧（含最新 AutoDL 代码改动合并），向 `origin`（`NY-WaKeUp/lehome_solution`）推送成功。`upstream` 仍指向 Ilia。

- 仓库：https://github.com/NY-WaKeUp/lehome_solution
- 当前 `main` tip：`552955d`（https://github.com/NY-WaKeUp/lehome_solution/commit/552955db2f04bbd9b721b1b6529d065738110978）
- Push：成功（`91b3a3a..552955d`）

## 本次新提交

| Hash (short) | Full | Message | URL |
|---|---|---|---|
| `f9e789d` | `f9e789d74843bf30f71d816d70b5c06723655b6c` | Adapt A800 training path and seed checkpoint assets for fresh runs. | https://github.com/NY-WaKeUp/lehome_solution/commit/f9e789d74843bf30f71d816d70b5c06723655b6c |
| `36aaaec` | `36aaaecd808d28f882d0ccd9aee2647f0a685d04` | Track AutoDL ablation/probe/disk-guard scripts and TB log bridge. | https://github.com/NY-WaKeUp/lehome_solution/commit/36aaaecd808d28f882d0ccd9aee2647f0a685d04 |
| `496d028` | `496d02809ab031c6d718150df5db7b25a0dbd76c` | Add AutoDL/AWR/MAX_OK session docs under docs/. | https://github.com/NY-WaKeUp/lehome_solution/commit/496d02809ab031c6d718150df5db7b25a0dbd76c |
| `552955d` | `552955db2f04bbd9b721b1b6529d065738110978` | Tighten ignore rules for caches, venvs, and weight dumps. | https://github.com/NY-WaKeUp/lehome_solution/commit/552955db2f04bbd9b721b1b6529d065738110978 |

此前已在 fork 上的相关提交（未在本次新建，但属于同一主题）：

| Hash | Message | URL |
|---|---|---|
| `91b3a3a` | Add AWR loss-weighted path and dual-GPU 24GB training adaptations. | https://github.com/NY-WaKeUp/lehome_solution/commit/91b3a3abb8c12bc677fc36ce81df13a26092e5eb |

## 纳入内容摘要

1. **代码/配置（以 AutoDL 较新侧为准合并到 Aserver）**
   - `scripts/run_rl_pipeline.py`：A800 1x80GB（`fsdp_devices=1`, `num_flow_samples=5`）、保留 heads、`_ensure_checkpoint_assets` 播种 `norm_stats`
   - `configs/rl_pipeline_sim.yaml` / `rl_pipeline_sim_loss_weighted.yaml`：`num_workers=2`, `batch_size=192`, `save_interval=1000`, `keep_period=null`
   - `scripts/batch_smoke.sh`
2. **AutoDL 仓库外脚本 → `scripts/autodl/`**
   - ablation / probe / disk_guard / ladder_probe* / run_awr_ablation* / run_smoke* / dl_bc / run_uv_sync 等
3. **TB 旁路 → `scripts/tb_bridge/log_to_tb.py`**
4. **文档 → `docs/`**（本机 `/Users/ningyu/doc` 相关 md + AutoDL 巡检/probe 结果 md）
   - 仓内原 `doc/` 仍在 `.gitignore`（本地笔记）；可跟踪笔记统一放 `docs/`
5. **`.gitignore`**：补强 `*.orbax*`、`uv-cache/`、`hf-home/`、`openpi-cache/`、`*.ckpt`/`*.safetensors`/`*.pt`/`*.pkl`/`*.npz`、`**/.venv/` 等

## 未纳入（有意）

- `challenge_done.txt` / `main_done.txt` / `submodule_done.txt` / `verify_out.txt`（安装/完成标记与输出）
- 权重/ckpt、`outputs/`、`checkpoints/`、`assets/`、数据集、`wandb/`、`.venv/`、`uv-cache/`、`hf-home/`、`openpi-cache/`
- AutoDL 临时盘上的 `*.log`、lock 文件、DISABLED 脚本
- 本机无关文档：`2026-11-11-Ryubing玩家2Xbox键盘匹配.md`、`2026-10-04-ssh-Manjaro-zerotier.md`
- 真实密钥/密码：未提交（文档仅有 `WANDB_API_KEY=...` 占位说明）

## 主机核对

- `gh auth`：`NY-WaKeUp` 已登录（ssh）
- Aserver SSH Host `Aserver`：可用
- AutoDL：`ssh -p 35794 -i ~/.ssh/id_ed25519 root@connect.nma1.seetacloud.com` 可用
- AutoDL 仓库因 ownership（uid 501 vs root）无法直接 `git` 操作；以 Aserver 合并后 push 为准。若需 AutoDL 对齐，可在该机 `chown` 后 `git pull`，或重新 clone。

## 本机 `/Users/ningyu/doc`

- 非独立 git 仓；相关 md 已拷入 fork 的 `docs/` 并跟踪。
- 本节记录文件：本文件本身仍在本机 `doc/`；可按需再同步进仓库 `docs/`。

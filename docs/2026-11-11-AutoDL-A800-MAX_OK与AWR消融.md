# AutoDL A800：MAX_OK 与 AWR 消融

日期：2026-11-11  
机器：`connect.nma1.seetacloud.com:35794`（A800 80GB PCIe；cgroup RAM **120 GiB**；数据盘 `/root/autodl-tmp` **50G**）

> **2026-11-11 12:21 更新：正式 MAX_OK=192**（160/176/192 exclusive OK）。  
> 消融已改为 fresh **`batch=192`**，exp=`sim_rl_awr_sample_b192` / `sim_rl_awr_loss_b192`。  
> 下文大量「MAX_OK=128 / b128」为当日早段记录，**以文首与 `2026-11-11-MAX_OK192确认并重开AWR-b192消融.md` 为准**。

## 结论

- **MAX_OK = 192**（`fsdp_devices=1`，`num_flow_samples=5`，`num_workers=2`，`ema_decay=None`；peak VRAM ~70GB）
- 早先 192 defer / exit=137 为争用场景；独占重测 160→176→192 全 OK
- 稳态 `nvidia-smi` 常落在 ~15GB；**峰值出现在 JIT / gemm autotune / remat / ckpt**，必须以 2s 采样 peak 为准

## MAX 表（train_only 冒烟）

| batch | result | Step0 loss | peak_vram (MiB) | peak_cgroup (GiB) | 备注 |
|------:|:------:|-----------:|----------------:|------------------:|------|
| 32 | OK | 1.8687 | ~15371（粗） | — | JIT~162s，其后 ~15–29s/it |
| 48 | SKIP | — | — | — | 早期被并发打断，b64 通过后未重测 |
| 64 | OK | 2.1736 | **41317** | 71.03 | Training complete step 2 |
| 96 | OK | 2.1687 | **53971** | 71.26 | remat: 52.65→49.86 GiB（floor 46.50） |
| 128 | OK | 1.9997 | **60557** | 71.52 | remat: 62.43→58.77 GiB；**MAX_OK** |
| 192 | 慎试 | — | — | — | 曾 HOST_KILLED(137)；未作为 MAX |

原始记录：`/root/autodl-tmp/max_batch_probe.md`

## 部署要点

- GitHub 直连 clone 不稳 → 代码从 Aserver 精简 rsync；BC / pi05 走 `HF_ENDPOINT=https://hf-mirror.com`
- 需系统 FFmpeg，否则 `torchcodec` 无法加载
- 仓库里为双 4090 写死的 `fsdp_devices=2` 已在 AutoDL 改为 **1**；`num_workers=2`
- 50G 盘紧：每次长训前删 numbered orbax ckpt / tmp；`openpi-cache`≈12G、`lehome_solution`（含 data+venv）≈18G

## AWR 消融（batch=128）

### 设计

| arm | config | `awr_loss_weighted` | `--exp_name` |
|-----|--------|---------------------|--------------|
| baseline（采样 AWR） | `configs/rl_pipeline_sim.yaml` | false | `sim_rl_awr_sample_b128` |
| loss-weighted | `configs/rl_pipeline_sim_loss_weighted.yaml` | true | `sim_rl_awr_loss_b128` |

单卡串行：先 baseline，再 loss-weighted（避免双 ckpt 撑爆 50G）。  
`steps_per_iteration=10000`，`batch_size=128`，`save_interval=500`。

### 启动命令

```bash
# 独占锁脚本（已落盘）
setsid bash /root/autodl-tmp/run_awr_ablation.sh \
  </dev/null >/root/autodl-tmp/ablation/outer.log 2>&1 &
```

等价单臂：

```bash
cd /root/autodl-tmp/lehome_solution
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi-cache
export HF_ENDPOINT=https://hf-mirror.com
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90

.venv/bin/python -u scripts/run_rl_pipeline.py \
  --config=configs/rl_pipeline_sim.yaml \
  --train_only --steps_per_iteration 10000 --batch_size 128 \
  --exp_name sim_rl_awr_sample_b128 --num_workers 2 \
  > logs/awr_baseline_b128.log 2>&1

.venv/bin/python -u scripts/run_rl_pipeline.py \
  --config=configs/rl_pipeline_sim_loss_weighted.yaml \
  --train_only --steps_per_iteration 10000 --batch_size 128 \
  --exp_name sim_rl_awr_loss_b128 --num_workers 2 \
  > logs/awr_loss_weighted_b128.log 2>&1
```

### 日志路径

| 项 | 路径 |
|----|------|
| 消融编排状态 | `/root/autodl-tmp/ablation/STATUS.md` |
| 编排 outer | `/root/autodl-tmp/ablation/outer.log` |
| baseline 训练日志 | `/root/autodl-tmp/lehome_solution/logs/awr_baseline_b128.log` |
| loss-weighted 训练日志 | `/root/autodl-tmp/lehome_solution/logs/awr_loss_weighted_b128.log` |
| pipeline 内 train log | `outputs/checkpoints/pi_modified_bc_rl/<exp>/logs/training/train_iter0.log` |

### 启动情况（会话记录）

- 清理后曾尝试启动 `run_awr_ablation.sh`，因 GPU 锁被占用而 **abort**（正确避让）。
- 当时已有短跑：`batch=128 steps_per_iteration=30` 基线 config（非完整 1e4 消融）。
- 策略：**不杀短跑**；结束后再拉起完整串行消融。

## 安全

- 聊天中暴露过 SSH 密码 → **在 AutoDL 控制台改密**；本机已用 ed25519。

## 正式消融启动跟踪（loop 续）

**MAX_OK=128**。短跑（30 steps sampling）占卡时不叠跑；结束后用：

```bash
ssh -p 35794 -i ~/.ssh/id_ed25519 root@connect.nma1.seetacloud.com
setsid bash /root/autodl-tmp/run_awr_ablation.sh \
  </dev/null >/root/autodl-tmp/ablation/outer.log 2>&1 &
```

### 正式消融参数

| 项 | 值 |
|----|-----|
| batch_size | 128 |
| steps_per_iteration | 10000 |
| fsdp_devices | 1 |
| num_flow_samples | 5 |
| num_workers | 2 |
| 顺序 | baseline → loss_weighted（串行，腾盘） |

### exp / config / 日志

| arm | config | exp_name | 日志 |
|-----|--------|----------|------|
| sampling 基线 | `configs/rl_pipeline_sim.yaml` | `sim_rl_awr_sample_b128` | `/root/autodl-tmp/lehome_solution/logs/awr_baseline_b128.log` |
| loss-weighted | `configs/rl_pipeline_sim_loss_weighted.yaml` | `sim_rl_awr_loss_b128` | `/root/autodl-tmp/lehome_solution/logs/awr_loss_weighted_b128.log` |
| 状态 | — | — | `/root/autodl-tmp/ablation/STATUS.md` |
| outer | — | — | `/root/autodl-tmp/ablation/outer.log` |

管道内 train log：`outputs/checkpoints/pi_modified_bc_rl/<exp_name>/logs/training/train_iter0.log`

### 进度补充

- `max_batch_probe.md` 已定稿：**MAX_OK=128**
- 另有短跑 `run_ablation_baseline.sh`（`steps=30`, batch=128, sampling）占卡；长消融脚本 `run_awr_ablation.sh`（1e4×2 臂）已写好，**等短跑结束后独占启动**，避免叠跑。

### loop20 tick（2026-11-11 05:10 CST）只查勿杀

| 项 | 状态 |
|----|------|
| 30-step 短跑 | **未结束** — `Progress ~14/33`，~11.1s/it，ETA ~3.5min |
| 正式 `run_awr_ablation.sh` | **未启动**（无进程；`ablation/STATUS.md` 不存在） |
| 当前脚本 | `run_ablation_baseline.sh` → `run_rl_pipeline --steps_per_iteration 30 --batch_size 128` |
| batch / steps | **128 / 30**（目标 33 it） |
| GPU | **100%**，60539 MiB / 81920 |
| 催促 | **否**（卡忙）；短跑结束后需独占启动正式消融 |

### loop21 tick（2026-11-11 05:13 CST）只查勿杀

| 项 | 状态 |
|----|------|
| 30-step 短跑 | **未结束** — `run_ablation_baseline.sh` + `train.py` 仍在；日志仅 Step 0，进程 ~8min，GPU 满载 |
| 正式 `run_awr_ablation.sh` | **未启动**（无进程；无 `STATUS.md`；`outer.log`=lock abort） |
| GPU | **100%**，60539 MiB |
| 催促 | **否**（非空闲） |

## 实际启动确认（2026-11-11 05:14）

1. 短跑 `run_ablation_baseline.sh`（30 steps）已完成：`sampling exit=0`，Step0 loss=2.1655，latest step 32；日志 `/root/autodl-tmp/ablation_sampling_b128.log`
2. **长消融已独占启动**（未叠跑）：
   - 编排：`/root/autodl-tmp/run_awr_ablation.sh`
   - 状态：`/root/autodl-tmp/ablation/STATUS.md`（`started: 2026-11-11T05:14:25+08:00`）
   - **当前臂**：baseline `awr_loss_weighted=false`，`--exp_name sim_rl_awr_sample_b128`，**target 10001 steps**，`batch_size=128`
   - 日志：`/root/autodl-tmp/lehome_solution/logs/awr_baseline_b128.log`
   - ckpt 目录：`outputs/checkpoints/pi_modified_bc_rl/sim_rl_awr_sample_b128/`
   - 完成后脚本会自动跑 loss-weighted 臂 → `sim_rl_awr_loss_b128` / `logs/awr_loss_weighted_b128.log`
3. 盘：清 numbered ckpt 后约 **21G free** / 50G

粗算：若 ~25–40 s/step，单臂 1e4 steps ≈ **70–110 h**；两臂串行需预留租期或中途改 steps。

### loop22 tick（2026-11-11 05:16 CST）只查勿杀

| 项 | 状态 |
|----|------|
| 正式消融重启 | **已启后秒挂** — `STATUS.md` started 05:14:25；baseline `exit=1` @ 05:14:40；无 train/ablation 进程 |
| Step | **无** |
| 异常 | `ValueError: Normalization stats not found`（`train_iter0.log`） |
| GPU | **0% / 0 MiB** |
| 催促 | **是** — 补 assets 后重开正式消融 |

## 秒挂根因与修复（2026-11-11 05:16–05:19）

### 根因

`run_rl_pipeline` 把 `AssetsConfig.assets_dir` 指到 **`checkpoint_dir/assets`**（按 exp 自包含）。  
新 `--exp_name`（`sim_rl_awr_sample_b128` / `sim_rl_awr_loss_b128`）目录空，缺 `assets/sim_rl/norm_stats.json` → `ValueError: Normalization stats not found`。  
短跑用默认 `exp_name=sim_rl` 已有 assets，故未暴露。

### 修复（勿再开第二份）

1. **种子拷贝**（两臂）：
   - 源：`/root/autodl-tmp/lehome_solution/outputs/assets/pi_modified_bc_rl/sim_rl/`
   - 目标：
     - `.../checkpoints/pi_modified_bc_rl/sim_rl_awr_sample_b128/assets/sim_rl/`
     - `.../checkpoints/pi_modified_bc_rl/sim_rl_awr_loss_b128/assets/sim_rl/`
2. **fork 补丁**：`scripts/run_rl_pipeline.py` 增加 `_ensure_checkpoint_assets`（`run_training` 入口自动从 `outputs/assets/...` 或 sibling exp 拷贝）
3. **`run_awr_ablation.sh`**：每臂启动前再 `seed_assets` 一次

### 重启确认（独占，已放着跑）

| 项 | 值 |
|----|-----|
| 启动 | `2026-11-11T05:16:55+08:00`（PID ablation≈55805） |
| 当前臂 | baseline `sim_rl_awr_sample_b128`，target **10001** steps，batch=128 |
| norm_stats | **Loaded** from `.../sim_rl_awr_sample_b128/assets/sim_rl` |
| **Step 0** | **05:18:53**，`loss=2.1983`，`action_loss=2.1069` |
| VRAM | **60539 MiB**，GPU 100% |
| 后续 | 同脚本串行 → `sim_rl_awr_loss_b128` |
| STATUS | `/root/autodl-tmp/ablation/STATUS.md` |
| 日志 | `logs/awr_baseline_b128.log`；`.../sim_rl_awr_sample_b128/logs/training/train_iter0.log` |

**不要重复启动**第二份 `run_awr_ablation.sh`。

## loop 已停止（2026-11-11 ~05:19）

正式 AWR 消融已确认 Step 0（baseline，batch=128，loss≈2.20，VRAM≈60GB）。停止 `/loop` 每 3 分钟巡检，避免重复唤醒。


## 状态快照（2026-11-11 ~10:58 CST 巡检）

**结论：baseline 臂已挂，GPU 空闲；根因盘满写 ckpt 失败；loss_weighted 未开。**

| 项 | 值 |
|----|----|
| 进程 | 无 train / run_awr_ablation；仅 tensorboard/jupyter |
| GPU | A800 0 MiB / 0% |
| baseline | 目标 10001；日志末 **Step 1500**；Step0 loss=2.1983 → Step100≈1.91 → 中段降到 ~0.6 量级 |
| 失败 | `09:55:16` exit=1；orbax 写 `1000` ckpt：`No space left on device` |
| 残留 | 完整 `500`(~12G) + 残缺 `1000.orbax-checkpoint-tmp-4`(~9.5G) |
| loss_weighted | **未启动**（无日志） |
| 盘 | `/root/autodl-tmp` **50G/50G 100%，余 17M** |
| 背景仍成立 | MAX_OK=128；192 deferred；05:16 norm_stats 修复有效（已跑到 1500） |

**下一步（未执行）**：删残缺/旧 ckpt 腾盘（或扩容）→ 从可用 ckpt/resume 重开 baseline；勿叠跑第二份消融。

## 盘满中断修复 + 可持续清理（2026-11-11 ~11:03–11:07）

**结论：已腾盘约 22G 可用；从完整 ckpt step 500 resume；Step 已推进到 513+；单实例在跑；边跑边清已落地。**

### 清盘动作

| 动作 | 效果 |
|------|------|
| 删残缺 `1000.orbax-checkpoint-tmp-4` | ~9.5G |
| `openpi-cache` 迁到系统盘 `/root/openpi-cache` + symlink | ~12G 数据盘释放 |
| 保留完整 `sim_rl_awr_sample_b128/500` | resume 源 |
| 结果 | `/root/autodl-tmp`：**29G used / 22G avail**（此前 100% / 17M） |

### resume

| 项 | 值 |
|----|-----|
| 编排 | `/root/autodl-tmp/run_awr_ablation_resume.sh`（flock，勿叠跑） |
| restore | **step 500**（orbax Finished restoring …/500） |
| 目标 | **10001**（`steps_per_iteration=9500` = 10001−500−1） |
| 配置 | `resume=True`，`save_interval=1000`，`keep_period=None`，`batch=128` |
| 已推进 | Progress **501→513+**，~11–12 s/it；VRAM **60537 MiB / 100%** |
| PID | disk_guard=**65960**，orchestrator=**65961**，pipeline=**65994**，train=**66135** |
| 日志 | `logs/awr_baseline_b128_resume.log`；`.../sim_rl_awr_sample_b128/logs/training/train_iter0.log` |
| outer | `/root/autodl-tmp/ablation/outer_resume.log` |
| 后续 | baseline exit=0 后同脚本自动开 `sim_rl_awr_loss_b128` |

### 可持续机制（不是只清一次）

1. **写入更稀**：两臂 yaml `save_interval: 1000`；`keep_period: null`
2. **orbax 只留 1 份**：`max_to_keep=1`（代码已有）
3. **后台 disk_guard**（`/root/autodl-tmp/disk_guard.sh`，thresh=**15G** / 30s）：
   - 可用 <15G → 先删 orbax `*.tmp*` → 每 exp 只留最大完整 numbered ckpt → 必要时删大 log/profile
   - 日志：`/root/autodl-tmp/ablation/disk_guard.log`
4. **openpi-cache 常驻系统盘 symlink**，数据盘优先留给 ckpt/data
5. STATUS 已写：`/root/autodl-tmp/ablation/STATUS.md`

**不要再启动第二份训练/消融编排。**

## Wandb / TensorBoard 现状（2026-11-11 ~11:13）

### 结论

**当前这次 AWR baseline resume（`sim_rl_awr_sample_b128`）既没有可用的 wandb run，也没有 TensorBoard event。** 只能看本地文本 log。logger 栈是 **wandb-only**（openpi/lehome），不是 TB。

置信度：**高**（yaml + pkl + 进程环境 + 磁盘检索交叉一致）。

### 配置 / 环境证据

| 项 | 值 |
|----|-----|
| `configs/rl_pipeline_sim.yaml` | `wandb_enabled: false`；`project_name: LeHome`；默认 `exp_name: sim_rl`（CLI 覆盖为 `sim_rl_awr_sample_b128`） |
| `configs/rl_pipeline_sim_loss_weighted.yaml` | 同样 `wandb_enabled: false` |
| `configs/rl_pipeline_sim_to_real.yaml` | `wandb_enabled: true`（**非本次实验**） |
| 当前 TrainConfig pkl | `wandb_enabled=False`，`project_name=LeHome`，`exp_name=sim_rl_awr_sample_b128`，`batch_size=128` |
| `WANDB_*` 环境 | 无（login shell / train 进程 environ 均无） |
| `~/.netrc` / `~/.config/wandb` | 不存在 → **未 login** |
| `wandb_id.txt` | 无 |
| `wandb/` run 目录 | 无（仅 `.venv` 里的包） |
| `events.out.tfevents*` | 全盘无 |
| `/root/tf-logs` | 空 |
| 代码 | `scripts/train.py` → `init_wandb(... enabled=config.wandb_enabled)`；`enabled=False` 时 `wandb.init(mode="disabled")`；**无 TensorBoard / SummaryWriter** |
| 训练进程 | 已有默认 AutoDL TB：`tensorboard --host 0.0.0.0 --port 6007 --logdir /root/tf-logs`（与本次训练无关，logdir 空） |

### 若有：URL / 路径 / 怎么开

- **Wandb**：无 URL、无 entity/run。
- **TB**：无训练 logdir。实例自带 TB 指空目录 `/root/tf-logs`，AutoPanel 打开也是空。
- **本地唯一入口**：
  - `.../sim_rl_awr_sample_b128/logs/training/train_iter0.log`
  - 编排 log：`logs/awr_baseline_b128_resume.log`（以 resume 脚本为准）

### AutoDL 端口说明（官方，未瞎编）

- 内置 TB：默认看 `/root/tf-logs`；换目录需 kill 后 `tensorboard --port 6007 --logdir <path>`，再从 AutoPanel 进 TB。见 [AutoDL TensorBoard](https://www.autodl.com/docs/tensorboard/)
- 自定义服务映射实例内 **6006 / 6008**；公网地址在控制台「自定义服务」复制。见 [开放端口](https://www.autodl.com/docs/port/)
- 若自起 TB：`tensorboard --bind_all --port 6006 --logdir <dir>`，再在「自定义服务」开 6006；**前提是先有 tfevents**（本仓库训练路径目前不写）

### 若无：原因 + 最小启用（下次跑 / 需重启）

**原因**：sim yaml 显式关了 wandb；机器无 API key；训练路径不写 TB。

**最小开 wandb（推荐，与代码一致）**：

1. 设 API key 并 login（`setup.sh` 已支持）：`export WANDB_API_KEY=...` → `uv run wandb login`（或跑 setup 对应段）
2. 改 yaml：`configs/rl_pipeline_sim.yaml`（及 loss_weighted 若要）`wandb_enabled: true`
3. **重启该 arm**（当前已 `mode=disabled`，热改无效；resume 会读/写 `wandb_id.txt`）
4. 打开：W&B 项目 **`LeHome`**，run name ≈ **`sim_rl_awr_sample_b128`**（entity 取决于 key 所属账号）

**不建议为「现在看一眼」硬接 TB**：要改 `scripts/train.py` 加 SummaryWriter，成本高于翻 wandb 开关；且当前已无 event。

**现状可看指标**：`tail -f` 上述 train log（Progress / TIMING）；丰富标量曲线需开 wandb 后重跑/续跑。


# 2026-11-11 Aserver 训练状态与 TB / wandb

查询时刻：约 **11:02 CST**（SSH 实测）

## 结论

- **训练在跑且接近收尾**：`sim_rl` warmup / train_only，约 **Step 9700 / 10001**。
- **没有可用的 TensorBoard 或 wandb 面板**：本机配置 `wandb_enabled: false`；产物目录里也没有 `events.out.tfevents*` / 活跃 wandb run。
- **看曲线只能看日志**（或事后自己 parse）。

置信度：**高**（直接读 GPU、进程、YAML、日志与 checkpoint 目录）。

## 当前状态

| 项 | 值 |
|----|----|
| 机器 | Aserver `wmtlab-ubuntu-server1`（2× RTX 4090） |
| GPU | GPU0 ~15464 MiB / 100%；GPU1 ~15369 MiB / 99% |
| 主进程 | `train.py` PID **3703169**（已跑约 **8h40m**，~201% CPU） |
| 父管线 | `run_rl_pipeline.py --train_only --steps_per_iteration 10000 --batch_size 16` |
| 配置 | `configs/rl_pipeline_sim.yaml` → exp `sim_rl`；`awr_loss_weighted: false` |
| Step 0 | 02:23:49，`loss≈2.0922` |
| 最新 Step | **9700** @ 11:01:36，`loss≈0.3062`，`action_loss≈0.2725`，`garment_type_accuracy≈0.998` |
| 步速 | 近端约 **3.2 s/step**（按每 100 step 日志点估） |
| 预计结束 | 距 10001 约 **300 steps → ~16 min**（约 11:18 前后；随 JIT/存盘会浮动） |
| Checkpoint | 已有 **5000**、**9500**：`outputs/checkpoints/pi_modified_bc_rl/sim_rl/{5000,9500}` |

## TB / wandb？

| 通道 | 有没有 | 说明 |
|------|--------|------|
| **wandb** | **无** | YAML L166：`wandb_enabled: false`；远程续训文档也写了本地关 wandb/hf_sync；无 run URL |
| **TensorBoard** | **无** | `outputs/.../sim_rl` 下无 tfevents；无 6006/6007 上的训练 TB；训练写的是文本 log |
| 文本日志 | **有** | 主入口：`/home/ningyu/lehome_solution/logs/train_baseline.log`；镜像：`.../sim_rl/logs/training/train_iter0.log` |

SSH 里 `31080` / `30091` 在听，对应本机 `LocalForward`（浏览器/远程桌面类），**不是**这次训练的 TB。

## 怎么看进度（现在）

```bash
ssh Aserver 'tail -f /home/ningyu/lehome_solution/logs/train_baseline.log'
# 或
ssh Aserver 'tail -n 5 /home/ningyu/lehome_solution/outputs/checkpoints/pi_modified_bc_rl/sim_rl/logs/training/train_iter0.log'
```

抽关键字段示例：`Step`、`loss`、`action_loss`、`garment_type_accuracy`。

## Loss 走势（日志抽样，便于记忆）

| Step | 时间 | loss（约） |
|-----:|------|----------:|
| 0 | 02:23 | 2.09 |
| 5000 | （有 ckpt） | — |
| 6000 | 07:44 | 0.34 |
| 8000 | 09:30 | 0.32 |
| 9000 | 10:24 | 0.29 |
| 9700 | 11:01 | 0.31 |

趋势：**整体从 ~2.1 降到 ~0.3 一带**；后半段在 0.29–0.33 震荡属正常波动，不是崩了。

## 与 AutoDL 消融的关系（别混）

- **Aserver 这条**：`sim_rl`、`batch_size 16`、目标 ~10001、日志 `train_baseline.log`。
- **AutoDL A800 消融**：另一台机、`batch 128`、`sim_rl_awr_sample_b128` 等；那边 YAML 同样 `wandb_enabled: false`。见 `doc/2026-11-11-AutoDL-A800-MAX_OK与AWR消融.md`。

## 若以后要面板（未执行，仅备忘）

1. 开 wandb：改 YAML `wandb_enabled: true` 并 `wandb login` 后**新开 run**（当前 run 不会自动补历史）。
2. 或对已有 `train_baseline.log` 做离线 parse → CSV / 本地 TB writer（事后可视化）。

## 优势 / 风险

- **优势**：双卡满载、日志持续更新、已有 9500 ckpt，收尾风险低。
- **风险 / 劣势**：无在线面板，只能 tail 日志；本轮 batch=16 远小于作者 192，**不能**据此对齐 Ilia 最终效果（本轮是 Aserver 可跑规模的 warmup 续训）。

## 证据来源

- SSH `Aserver`：`nvidia-smi`、`ps`、`train_baseline.log`、`rl_pipeline_sim.yaml`、`outputs/checkpoints/...`
- 远程笔记：`/home/ningyu/lehome_solution/doc/2026-11-11_Aserver续训_a4091f0e.md`

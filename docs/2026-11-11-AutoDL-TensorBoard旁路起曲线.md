# 2026-11-11 AutoDL TensorBoard 旁路起曲线

## 结论

- **方案 A（旁路）已落地**：未改训练代码、**未重启训练进程**。
- 默认 AutoPanel TensorBoard（`port=6007`，`--logdir /root/tf-logs`）已能读到 run **`sim_rl_awr_sample_b128`**，含 **loss** 等标量，并在 resume 打出新 `Step` 后持续更新。
- 置信度：**高**（本机 `EventAccumulator` + `curl :6007/data/runs` + live Step 600 写入均已核对）。

## 怎么打开面板

按 [AutoDL TensorBoard 文档](https://www.autodl.com/docs/tensorboard/)：

1. 打开实例控制台 → **AutoPanel**。
2. 点内置 **TensorBoard** 入口（平台进程已在跑：`tensorboard --host 0.0.0.0 --port 6007 --logdir /root/tf-logs`，`--reload=5`）。
3. 左侧 Runs 勾选 **`sim_rl_awr_sample_b128`**；Scalars 看 **`loss`** / **`train/loss`** / **`train/action_loss`** 等。

可选（一般不必）：

- [开放端口 / 自定义服务](https://www.autodl.com/docs/port/) 映射的是实例 **6006 / 6008**。本次**没有**另起 6006 上的 TB，以免和默认 6007 面板混淆；事件已写在平台钉死的 `/root/tf-logs`。
- 若以后要换 logdir：先杀默认 TB，再 `tensorboard --port 6007 --logdir <新目录>`，然后重新进 AutoPanel TB。

## 路径与进程

| 项 | 值 |
|---|---|
| TB logdir | `/root/tf-logs/sim_rl_awr_sample_b128` |
| 平台 TB | PID 见 `ps`：`... tensorboard --port 6007 --logdir /root/tf-logs` |
| 旁路脚本 | `/root/autodl-tmp/tb-bridge/log_to_tb.py` |
| 旁路 stdout | `/root/autodl-tmp/tb-bridge/log_to_tb.out` |
| 历史 metric 源 | `/root/autodl-tmp/lehome_solution/logs/awr_baseline_b128.log`（`--max-step 500`） |
| 实时 metric 源 | `.../sim_rl_awr_sample_b128/logs/training/train_iter0.log`（无 cap） |
| 训练进程 | **未动**（`train.py` 仍为 resume 中的 PID，当时约 66135） |
| disk_guard | 仍在跑，未改 |

启动命令（已在后台 nohup）：

```bash
/root/miniconda3/bin/python /root/autodl-tmp/tb-bridge/log_to_tb.py --poll 5 \
  --max-step 500 --max-step -1 \
  /root/autodl-tmp/lehome_solution/logs/awr_baseline_b128.log \
  /root/autodl-tmp/lehome_solution/outputs/checkpoints/pi_modified_bc_rl/sim_rl_awr_sample_b128/logs/training/train_iter0.log
```

## 仓库 metric 机制（查证）

- 主路径：`scripts/train.py` → `init_wandb` / `wandb.log(...)`；`wandb_enabled: false` 时 **不写** `events.out.tfevents*`。
- 控制台：`step % log_interval == 0` 时 `logging.info(f"Step {step}: {k=v,...}")`（约每 100 step）。
- pipeline 把 train stdout/stderr tee 到 `train_iter0.log`，并把含 `Step ` / `loss=` 的行 mirror 到外层 `logs/awr_baseline_*.log`。
- **未走方案 B**（改 `SummaryWriter` + resume 重启）：成本高于旁路，且会打断当前 batch=128 resume。

## 曲线状态（核验时）

- 历史灌入：step `0..500`（来自崩溃前 log，截断到 ckpt=500）。
- 实时：resume 已写入 **Step 600，`loss≈0.488`**；bridge 日志 `wrote 1 step-lines`；API 与 EventAccumulator 一致为  
  `steps=[0,100,200,300,400,500,600]`。
- 之后每 ~100 step（约 18 分钟）会出现新点；`progress/it`、`timing/*` 也会更新。

## 优势 / 风险

| | |
|---|---|
| 优势 | 零打断训练；立刻有历史 loss；落在平台默认 logdir，AutoPanel 可直接开 |
| 风险 | 旁路依赖 console 文本格式；bridge 进程若挂需重拉；非 wandb 结构化字段的完整等价物 |

## Sources

- [AutoDL TensorBoard](https://www.autodl.com/docs/tensorboard/)
- [AutoDL 开放端口 / 自定义服务](https://www.autodl.com/docs/port/)

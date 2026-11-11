# /loop 第 5 次 tick · AutoDL 冒烟根因巡检

日期：2026-11-11 04:25–04:26（UTC+8）  
主机：`connect.nma1.seetacloud.com:35794`（A800 80GB）

## 结论

- **根因（上一轮失败）**：`batch=64` 被系统 `Killed` → **exit 137**（内存杀进程）；日志里曾短暂出现 `_pickle.UnpicklingError: pickle data was truncated`，属杀进程打断 `_pipeline_config_iter0.pkl` 写入/读取的连带症状。更早一轮启动失败根因是单卡上的 `fsdp_devices=2`（已改）。
- **当前动作**：**勿打断**。`batch_smoke` 第三轮已在跑：`run_rl_pipeline … --batch_size 192`（pid 16990）+ `train.py`（pid 17135），`train_iter0.log` / `smoke_b192.log` 时间戳 **04:25**；尚在数据加载阶段，GPU ~433MiB。
- **是否需人工**：**否**（重试进行中；等冒烟出 `OK/FAIL` 再决定）。

## 证据摘要

| 项 | 值 |
|----|-----|
| 当前 `train_iter0.log` | 已被新 run 覆盖，**无**完整 traceback（仅启动 INFO） |
| `batch_smoke_nohup.log` | `FAIL batch=64 exit=137`；`Killed` pid 15872 |
| 瞬时 grep | `train_iter0.log` 曾见 `UnpicklingError: pickle data was truncated` |
| 现进程 | pipeline 16990 / train 17135，batch=192 |
| results | `## batch_size=192 2026-11-11T04:25:32+08:00`（未完） |

## 备注

当前盘上 `train_iter0.log` 无法再读到「完整 traceback 头尾」——文件已被重试覆盖；根因以 nohup 的 137 + 瞬时 Unpickling 为准。

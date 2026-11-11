# AWR 消融：norm_stats 秒挂修复与重启

日期：2026-11-11  
机器：AutoDL A800 `connect.nma1.seetacloud.com:35794`

## 结论

- **根因**：新 exp 缺 `assets/sim_rl/norm_stats.json` → `Normalization stats not found`
- **已修**：从 `outputs/assets/pi_modified_bc_rl/sim_rl/` 拷到两臂 checkpoint assets；pipeline 加 `_ensure_checkpoint_assets`
- **已独占重启**：`05:16:55`；**Step 0 @ 05:18:53**，`loss=2.1983`，VRAM≈60539 MiB
- **勿叠跑**

## MAX 表（摘要）

| batch | result | peak_vram (MiB) |
|------:|:------:|----------------:|
| 32 | OK | ~15k |
| 64 | OK | 41317 |
| 96 | OK | 53971 |
| **128** | **MAX_OK** | **60557** |
| 192 | 暂不冲 | — |

## 启动 / 日志

```bash
setsid bash /root/autodl-tmp/run_awr_ablation.sh \
  </dev/null >/root/autodl-tmp/ablation/outer.log 2>&1 &
```

| 项 | 路径 |
|----|------|
| STATUS | `/root/autodl-tmp/ablation/STATUS.md` |
| outer | `/root/autodl-tmp/ablation/outer.log` |
| baseline | `/root/autodl-tmp/lehome_solution/logs/awr_baseline_b128.log` |
| loss-weighted | `/root/autodl-tmp/lehome_solution/logs/awr_loss_weighted_b128.log` |
| train_iter0 | `.../sim_rl_awr_sample_b128/logs/training/train_iter0.log` |

详细上下文见：`2026-11-11-AutoDL-A800-MAX_OK与AWR消融.md`

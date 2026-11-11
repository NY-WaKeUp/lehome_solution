# lehome_solution 训练硬件与关键超参查证

日期：2026-11-11  
任务：查证 IliaLarchenko/lehome_solution（Learning to Fold）作者公开写明的训练硬件与关键超参。  
检索：`parallel-cli` 不可用；改用 cursor WebSearch / WebFetch；XSearch 账号不可用。

## 结论摘要

- **训练 GPU**：1× NVIDIA **H200**（单卡训练机）
- **Rollout / DAgger**：主要为 **RTX PRO 6000**（可多机；每机约 3–5 个并行 Isaac Sim）
- **公开训练数字（论文 §2.6）**：batch size **192**；总约 **300k** steps（另有 ad-hoc）；BC warm-up 论文写 **20k**；cosine LR $10^{-4}\to 10^{-5}$ over **100k**；每 iteration 约 **1000** steps；checkpoint 约每 **500** steps
- **仓库默认 yaml**：`batch_size: 192`、`steps_per_iteration: 1000`、`num_iterations: 300`、`warmup_steps: 10000`、`save_interval: 500` —— 与常见默认一致；**warmup 与论文 20k 不一致**

## 来源摘录

| 事实 | 来源 |
|------|------|
| training on H200; rollouts RTX PRO 6000 | [README Setup](https://github.com/IliaLarchenko/lehome_solution) |
| 1× H200; checkpoint ~40 min | [项目页](https://ilialarchenko.com/projects/lehome2026/) |
| single H200, batch 192, ~300k steps, 20k BC warm-up, ~1000 steps/iter | [arXiv HTML §2.1/§2.6](https://arxiv.org/html/2606.27163) |
| yaml batch_size=192, steps_per_iteration=1000, warmup=10000 | [rl_pipeline_sim.yaml](https://raw.githubusercontent.com/IliaLarchenko/lehome_solution/main/configs/rl_pipeline_sim.yaml) |
| talk: one H200; RTX PRO 6000 rollouts | [YouTube Part 1](https://www.youtube.com/watch?v=mTohakalKz8)（转写无精确时间戳） |

## 不确定项

- YouTube 转写无时间码，无法给出可跳转的精确秒数。
- 论文 warm-up **20k** vs yaml **10000**：以何者为最终竞赛跑不确定；作者 README 亦提示命名/遗留与报告不完全一致。

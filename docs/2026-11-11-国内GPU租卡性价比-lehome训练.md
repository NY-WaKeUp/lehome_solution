# 国内 GPU 租卡性价比（lehome / Learning to Fold / OpenPI-JAX）

日期：2026-11-11（检索时点）  
任务：针对「复现/接近作者大 batch 长跑」vs「小 warmup/消融」给出租卡建议。  
检索：`parallel-cli` 不可用；改用 WebSearch / WebFetch。价格会变，下单前以各平台结算页为准。

关联笔记：`doc/2026-11-11-lehome_solution训练硬件与超参查证.md`（作者：1×H200、batch=192、~300k steps）。

---

## 结论先行

| 目标 | 优先卡型 | 检索时点公开小时价区间（约） | 计费建议 |
|------|----------|------------------------------|----------|
| 小 warmup / 消融 / 通管线 | 1–2× RTX 4090（同机） | AutoDL 约 ¥1.9–2.2/卡时；国内低价竞价可到 ~¥1.2 | 按量；调试用无卡模式 |
| 接近作者大 batch / 长跑 | **单卡大显存优先**：A800/A100 80GB 或 H20 96GB；真要对齐 H200 再看共绩等 | A800 约 ¥5–8.5；A100-40G ~¥3.5；H20 AutoDL 公示 ~¥10；国内 H200 稀缺约 ¥35 | **包日/包周预留**；禁止无保障抢占长跑 |
| 不推荐作主路径 | 阿里云 PAI/EGS 个人长训；趋动云无透明卡型号价 | 阿里 L20 ~¥9.6；H20 目录价量级远高于零售 GPU 云 | 要发票/合规再考虑 |

**核心原则**：比的是有效吞吐 `(batch × step/s) / 元`，不是卡单价。双 4090、batch=16 已能开训 ≠ 性价比最优复现路径。

---

## 1. 检索到的公开价格（2026-11-10～11）

### AutoDL 官网公示（[autodl.com](https://www.autodl.com/)）

- 公告：因供应成本波动，**2026-11-12 将调整部分 GPU 价格**；包月可提前下单锁价。
- 页面可见标价示例：H800 / H20 **¥9.98/时**（会员 95 折）；A4000 ¥0.92；2080Ti ¥0.88；MTT S4000 ¥1.86。
- **官网首页未列出 H200 零售价**（检索时点）。
- 具体 4090/A800 等以**算力市场实时主机价**为准（机房不同价不同）。

### 第三方比价聚合（非官网成交价）

| 卡型 | 价格 | 来源 |
|------|------|------|
| AutoDL RTX 4090 | ¥1.88/时 | [gpuprice.cn AutoDL 4090](https://gpuprice.cn/platform/autodl/rtx-4090)（更新 11月10日 18:17） |
| AutoDL A800 80GB | ¥5.59/时 | [gpuprice.cn AutoDL A800](https://gpuprice.cn/platform/autodl/a800-80g) |
| AutoDL 多卡型表 | 4090 ¥2.19；A100-40G ¥3.45；A800 ¥5.88 | [墨元 AutoDL](https://gpu.inkmetaai.com/platform/autodl) |
| 国内 4090 最低 | ~¥1.16/时（PPIO 竞价） | [rentgpu.link/china](https://rentgpu.link/china)（抓取 2026-11-11） |
| 国内 A800 最低 | ~¥5.15/时（晨涧云，包日折算口径） | 同上 / [rentgpu.link](https://rentgpu.link/) |
| 国内 H800 | AutoDL ~¥10.50/时 | rentgpu.link/china |
| 国内 H200 141GB | 共绩算力 ~¥35/时（仅见 1 家在售） | rentgpu.link |
| 国内 L40S | 共绩 ~¥4.98/时 | rentgpu.link/china |
| 恒源云 4090 | ~¥2.00–2.50/时 | [rentgpu 恒源云](https://rentgpu.link/platform/gpushare) / [墨元恒源云](https://gpu.inkmetaai.com/platform/gpushare) |
| 恒源云 A800 | ~¥6.40–6.80/时 | 同上 |
| 阿里云 EGS L20 | ~¥9.6/时 | [阿里云开发者社区 2026 汇总](https://developer.aliyun.com/article/1772695) |
| 阿里云 H20 | 公开讨论量级约 ¥80–120/卡时（按量目录价区间，非零售 GPU 云） | [jimmy·research Alibaba Cloud](https://jimmyresearch.com/entities/alibaba-cloud/) |

### 趋动云 / 算能云

- [趋动云资源页](https://www.virtaicloud.com/resource/)：按 B/S/P 规格打包，**未公开按 RTX 4090 / A100 型号的清晰时价**；历史文案称池化、分钟计费、曾宣传低至 0.49 元/卡时——**不能当作当前可对比报价**。
- 算能云：本次检索**未拿到可引用的 2026 公开卡型号价目**。

**置信度**：AutoDL 规则页与官网公告 = 高；聚合站小时价 = 中（抓取延迟、机房差价、竞价 vs 按量口径混杂）；阿里 H20 区间 = 中低（二手汇总）。

---

## 2. 性价比框架（第一性原理）

定义有效吞吐：

\[
\text{有效吞吐} = \frac{\text{batch} \times \text{steps/s}}{\text{元/小时}}
\]

或单位样本成本：

\[
\text{元/百万样本} = \frac{\text{元/小时}}{\text{batch} \times \text{steps/s} \times 3600} \times 10^{6}
\]

### JAX / OpenPI 场景取舍

| 选项 | 优势 | 劣势（风险） |
|------|------|--------------|
| **H200 单卡** | 与作者设定最近；大显存撑大 batch（作者 192） | 国内公开货极少；~¥35/时量级；长跑总成本高 |
| **H20 96GB** | AutoDL 有公示价；显存大，利于逼近大 batch | 算力/带宽与 H200 不等价；需实测 step/s |
| **A800 / A100 80GB** | 合规训练主力；单卡大显存；零售价常低于 H 系 | 仍难直接等于 H200；库存紧；NVLink 多卡价更高 |
| **A100 40GB** | 单价较低（~¥3.5） | 全参微调显存吃紧；OpenPI 文档倾向 full FT >70GB |
| **4090×N** | 单价低、库存多；适合 warmup/消融 | 24GB 限制 batch；FSDP 工程差；通信与小 per-GPU batch 会拉低 MFU |
| **L40S 48GB** | 显存中等、数据中心卡 | 国内报价不一定比多卡 4090 更划算；需实测 |

OpenPI 官方量级（DeepWiki）：Inference >8GB；LoRA >22.5GB；**Full FT >70GB → A100-80 / H100 类**。  
社区 4×4090 JAX：全局 batch≈4、关 EMA、FSDP——与作者 batch=192 **差两个数量级**，长跑「同 steps」不等于「同样本量」。

用户现状：双 4090、batch=16「勉强开训」→ 适合通管线，**不适合作为接近作者复现的主算力**。

### 粗算（仅框架，非报价承诺）

作者量级 ~300k steps × batch 192 ≈ **5.76×10⁷ 样本**。  
若某配置只能跑 batch 16，要凑同一样本量约需 **12× steps**（或接受更少样本、结果不可比）。  
因此：**大显存单卡把 batch 拉上去**，往往比「很多 4090、很小 batch」更省钱——前提是 step/s 不会崩。

---

## 3. AutoDL 实操要点（官方文档）

来源：[计费](https://www.autodl.com/docs/price/) · [省钱](https://www.autodl.com/docs/save_money/) · [网络](https://www.autodl.com/docs/network/) · [数据](https://www.autodl.com/docs/instance_data/) · [QA](https://www.autodl.com/docs/qa/) · [本地盘](https://autodl.com/docs/local_disk/)

1. **按量**：开机计费、关机停 GPU 费；精确到秒；**按开关机时长，不是按 GPU 利用率**。
2. **包日/包周/包月**：更便宜且**预留 GPU**；租期内关机也计时。长跑优先。
3. **包年包月转按量退款**有阶梯规则；文档称「包完用完再转按量退」可能更经济——以当下规则页为准。
4. **付费数据盘**：按量实例**关机仍按日计费**；不用就缩容/释放。默认免费数据盘约 50GB。
5. **文件存储** >20GB：¥0.01/GB/日；**镜像** >30GB 同价量级。
6. **无卡模式**：¥0.1/时，调试/传数据；同账号通常限 1 个；会释放 GPU 库存。
7. **带宽**：同地区共享约 **3–10 Gbps**，不单计流量费。
8. **数据风险**：本地盘多数无冗余；连续关机 **15 天**释放实例；主机下架至少提前 7 天提示。
9. **多卡**：同实例同机可并行；多机多卡需联系客服。
10. **闲时/抢占**：第三方文章常提「闲时更便宜但可能被释放」；**本次未在官方计费页找到完整闲时规范**——租用页若有「闲时」选项再核对条款；**长跑不要赌抢占**。

---

## 4. 风险清单（客观）

1. **抢卡 / 空闲不足**：按量关机后 GPU 不预留；晚高峰 4090 紧俏（第三方经验）。
2. **断连与无监督计费**：SSH 断 ≠ 停机；训练结束务必 `shutdown` 或包期结束。
3. **数据迁入**：数据集 + 权重 + JAX 环境；跨区/网盘时间计入总成本。
4. **H200 单卡 vs 多卡 4090**：batch、通信、EMA、XLA 预分配、checkpoint 体积与恢复策略都不同；「能跑」≠「同分布」。
5. **11-12 调价**：AutoDL 已公告；短时比价结果可能失效。
6. **聚合站低价陷阱**：竞价/包日折算/无货/魔改 48G 4090 混列——必须点进官网确认库存与计费模式。

---

## 5. 可执行建议清单（租前核对）

1. 明确本轮是 **warmup/消融** 还是 **逼近 192 batch 的长跑**（目标不同，卡型不同）。
2. 打开 AutoDL / 恒源云 / 晨涧云 / 共绩 **官网结算页**，记录：卡型、显存、NVLink 与否、地区、按量 vs 包日、是否抢占。
3. 开 15–30 分钟实测：稳定 `batch`、`steps/s`、峰值显存；算 `batch×step/s/元`。
4. 长跑选 **同机多卡或单卡大显存**；确认能否包周锁卡。
5. 数据盘容量与**关机仍计费**项写进预算；checkpoint 同步到文件存储/网盘。
6. 镜像一次做对（CUDA/JAX/依赖），避免反复装环境烧开机费。
7. 看主机「可租截止/下架倒计时」；开到期与释放通知。
8. 余额留安全垫；欠费策略读文档（平台可能优先保数据继续扣存储）。

---

## Sources

- [AutoDL 官网](https://www.autodl.com/)
- [AutoDL 计费文档](https://www.autodl.com/docs/price/)
- [AutoDL 省钱绝招](https://www.autodl.com/docs/save_money/)
- [AutoDL 网络](https://www.autodl.com/docs/network/)
- [AutoDL 实例数据](https://www.autodl.com/docs/instance_data/)
- [GPU价网 AutoDL 4090](https://gpuprice.cn/platform/autodl/rtx-4090)
- [GPU价网 AutoDL A800](https://gpuprice.cn/platform/autodl/a800-80g)
- [墨元 AutoDL](https://gpu.inkmetaai.com/platform/autodl)
- [rentgpu.link 中国大陆比价](https://rentgpu.link/china)
- [rentgpu.link](https://rentgpu.link/)
- [恒源云 rentgpu](https://rentgpu.link/platform/gpushare)
- [趋动云资源页](https://www.virtaicloud.com/resource/)
- [阿里云开发者社区 GPU 价](https://developer.aliyun.com/article/1772695)
- [PAI DLC 计费](https://help.aliyun.com/zh/pai/product-overview/billing-of-dlc)
- [OpenPI GPU 需求 DeepWiki](https://deepwiki.com/Physical-Intelligence/openpi/2.1-installation-and-dependencies)

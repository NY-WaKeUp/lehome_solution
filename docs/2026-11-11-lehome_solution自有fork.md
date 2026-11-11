# 2026-11-11 lehome_solution 自有 fork

## 结论

- **维护源（fork）**：https://github.com/NY-WaKeUp/lehome_solution
- **上游**：https://github.com/IliaLarchenko/lehome_solution
- **工作副本**：Aserver `/home/ningyu/lehome_solution`（本机无同名仓）
- **已推送分支**：`main`
- **已推送 commit**：`91b3a3abb8c12bc677fc36ce81df13a26092e5eb`  
  信息：`Add AWR loss-weighted path and dual-GPU 24GB training adaptations.`
- **推送方式**：普通 `git push`（非 force）

## 远端布局（Aserver 仓内）

| remote   | URL |
|----------|-----|
| `origin` | `git@github.com:NY-WaKeUp/lehome_solution.git`（自有 fork，日常推送） |
| `upstream` | `git@github.com:IliaLarchenko/lehome_solution.git`（上游只读同步） |

GitHub 身份：`gh` 登录账号 `NY-WaKeUp`；fork 已存在（`isFork: true`，parent=`IliaLarchenko/lehome_solution`），无需新建。

## 本次纳入 commit 的改动

- AWR loss-weighted：`awr_loss_weighted` 开关、`normalized_exp_advantage_weights`、对应 yaml/测试
- 本地 24GB×2 训练适配：bf16 全参、CPU init 再 FSDP device_put、`uv run --no-sync`、关掉部分辅助头 / EMA / XSA 等
- `.gitignore` 增加 `doc/`

## 未提交、未推送（本地残留）

这些是环境/安装标记或临时验证，**故意未进仓**：

- `challenge_done.txt` / `main_done.txt` / `submodule_done.txt`
- `verify_env.py` / `verify_out.txt`

当前 `git status`：与 `origin/main` 同步；仅剩上述未跟踪文件。

## 以后怎么用

### Clone（新机器）

```bash
git clone git@github.com:NY-WaKeUp/lehome_solution.git
cd lehome_solution
git remote add upstream git@github.com:IliaLarchenko/lehome_solution.git
```

### 日常推自己的修改

```bash
# 在 Aserver 或任意 clone 上
git add … && git commit -m "…"
git push origin HEAD
```

### 从上游同步（不改本地未提交工作）

```bash
git fetch upstream
git merge upstream/main   # 或 rebase，按习惯二选一
git push origin main
```

### 子模块

上游有 `.gitmodules`；新 clone 后按仓库 README/`setup.sh` 拉子模块，勿把密钥或 `.env` 提交进 fork。

## 操作记录（本对话）

1. 定位：Aserver `/home/ningyu/lehome_solution` 有 AWR 等未提交修改；本机无仓。
2. 确认 fork：`NY-WaKeUp/lehome_solution` 已 fork 自 `IliaLarchenko/lehome_solution`。
3. 补 `upstream` remote；`origin` 已指向自有 fork。
4. 提交 9 个文件 → `91b3a3a`；`git push -u origin main` 成功。
5. 未触碰 AutoDL A800 部署冒烟任务。

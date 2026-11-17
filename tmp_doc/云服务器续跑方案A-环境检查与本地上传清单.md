# 云服务器续跑方案 A：环境检查与本地上传清单

**检查时间**：2026-11-17（在云主机上执行自检）  
**仓库路径（云）**：`/home/src/Reliable` → 实际目录 `/root/autodl-tmp/projects/reliable`  
**Git**：`f328b20`（`main`，与 `origin/main` 一致）

---

## 1. 结论：**当前不能续跑**

启动脚本 `start_bc_llm_cub_paper_wave_a_remaining.sh` 会在 preflight 阶段退出（缺少方案 B 的 `runs`）。即使绕过 preflight，仍缺 **Python 环境**、**CUB 数据**；且 **当前实例未检测到 GPU**（`nvidia-smi` → *No devices were found*，`torch.cuda.is_available()` → `False`）。BC-LLM 训练阶段需要 CUDA，请在 AutoDL **选择带 GPU 的实例**后再跑。

---

## 2. 云上现状（逐项）

| 检查项 | 状态 | 说明 |
|--------|------|------|
| 代码 / 脚本 | ✅ | `git` 在 `main`，含 `check_cub_paper_runs_ready.sh` |
| 路径 symlink | ✅ | `/home/src/Reliable` 已指向仓库 |
| `.env.llm` | ✅ | 云上已有（含 `DASHSCOPE_API_KEY`） |
| **方案 B `runs`** | ❌ | `runs/bc-llm/cub_paper_qwen/` 仅有空目录，**0** 个 `results.csv`（期望 **10**） |
| **`.venvs/bc-llm`** | ❌ | 不存在 |
| **CUB 数据集** | ❌ | 缺 `attributes.txt` 与 `CUB_200_2011/images/` |
| **GPU** | ❌ | 当前无可用 GPU 设备 |
| 磁盘 | ✅ | `autodl-tmp` 约 50G 可用；解冻 CUB + runs + venv 足够 |

---

## 3. 需要从本机上传的内容（你本地做）

下面路径默认本机与论文笔记一致为 **`/home/src/Reliable`**；若你本机仓库在别处，把「本机路径」整段替换即可。  
云上目标一律为 **`/home/src/Reliable/...`**（与 symlink 一致，等同于 **`/root/autodl-tmp/projects/reliable/...`**）。

### 3.0 大文件推荐：先 zip/tar 上传到 `tmp_doc`，再在云上解压到正确位置

**可以。** 大文件不必直传最终目录；先打到 **固定文件名** 的压缩包，上传到云：

```text
/root/autodl-tmp/projects/reliable/tmp_doc/
```

支持 **AutoDL 控制台「文件上传」** 或本机 `scp`（数据盘路径，空间约 50G，适合几百 MB～数 GB）。

| 你上传的文件名（放在 `tmp_doc/`） | 本机打包内容（压缩包内顶层目录名必须一致） | 解压后落点 |
|-----------------------------------|--------------------------------------------|------------|
| `reliable_cub_paper_qwen_*.tar.gz` | 本机 `pack_cub_paper_runs_for_cloud.sh` 产出（已是 `cub_paper_qwen/`） | `runs/bc-llm/` |
| **`upload_cub_paper_qwen.zip`** | 目录 **`cub_paper_qwen/`** 整包 | `runs/bc-llm/` |
| **`upload_bc_llm_venv.zip`** 或 `.tar.gz` | 目录 **`bc-llm/`** 整包 | `.venvs/` |
| **`upload_CUB_200_2011.zip`** 或 `.tar.gz` | 目录 **`CUB_200_2011/`**（含 `attributes.txt`、`CUB_200_2011/images/`） | `src/bc-llm/exp_cub_birds_existing/data/` |

**本机打包示例（zip）**

```bash
cd /home/src/Reliable

# 方案 B（必传之一）
cd runs/bc-llm && zip -r ~/upload_cub_paper_qwen.zip cub_paper_qwen && cd -

# Python 环境（二选一：上传 or 云上 pip 安装）
cd .venvs && zip -r ~/upload_bc_llm_venv.zip bc-llm && cd -

# CUB（也可不传，见 §3.2 公共盘解压）
cd src/bc-llm/exp_cub_birds_existing/data
zip -r ~/upload_CUB_200_2011.zip CUB_200_2011
```

上传到云 `tmp_doc/`（示例）：

```bash
scp -P <端口> ~/upload_cub_paper_qwen.zip ~/upload_bc_llm_venv.zip \
  root@<主机>:/root/autodl-tmp/projects/reliable/tmp_doc/
```

**上传完成后，在云上由我/你执行一次解压**（仓库已带脚本）：

```bash
cd /home/src/Reliable
./scripts/llm_experiments/unpack_staging_from_tmp_doc.sh
# 仅查看将执行什么：./scripts/llm_experiments/unpack_staging_from_tmp_doc.sh --dry-run
```

脚本会解压到 §8 约定的正式路径并打印 `results.csv` 数量、CUB、venv 是否就绪。  
**注意**：zip 内必须是「一层目录名 + 内容」，不要多包一层父目录（例如 zip 里应是 `cub_paper_qwen/tasks/...`，而不是 `Reliable/runs/...`）。

**体积粗估（便于选 zip 等级）**

| 包 | 约大小 |
|----|--------|
| `upload_cub_paper_qwen` | ~300–400 MB（含 DuckDB；`PACK_STRIP_LOGS=1` 打 tar 可更小） |
| `upload_bc_llm_venv` | 往往 2–4 GB+（可选改为云上按 freeze 安装，不传） |
| `upload_CUB_200_2011` | 解压后约 1.1 GB 图像（也可在云上 `tar -xzf /autodl-pub/...`，见 §3.2） |

---

### 3.1 必传：方案 B 实验目录（runs tarball / rsync）

**用途**：10 个已完成任务的 `results.csv` + 各任务 `api_cache.duckdb`（续跑跳过已完成任务、避免重复 API 计费）。  
**preflight 硬性要求**：`tasks` 下至少 **10** 个 `results.csv`。

| | 路径 |
|---|------|
| **本机（源）** | `/home/src/Reliable/runs/bc-llm/cub_paper_qwen/` |
| **云上（目标）** | `/home/src/Reliable/runs/bc-llm/cub_paper_qwen/` |

**方式 A — 本机一键 rsync（推荐）**

在本机执行（把 `用户@云SSH` 换成 AutoDL 控制台给出的 SSH，例如 `root@connect.xxx.autodl.com -p 端口`）：

```bash
cd /home/src/Reliable
export CLOUD_SSH='root@<云SSH主机> -p <端口>'   # 若需端口，可写成 rsync 的 -e "ssh -p 端口"
# 官方脚本（REMOTE 默认 /home/src/Reliable/runs/bc-llm/）
CLOUD_SSH='root@<主机>' ./scripts/llm_experiments/sync_cub_paper_runs_to_cloud.sh
```

若 `sync` 脚本不支持 `-p`，可用手动 rsync：

```bash
rsync -avz --progress \
  -e 'ssh -p <端口>' \
  /home/src/Reliable/runs/bc-llm/cub_paper_qwen/ \
  root@<主机>:/home/src/Reliable/runs/bc-llm/cub_paper_qwen/
```

**方式 B — 打 tar 再 scp（网络不稳时）**

本机：

```bash
cd /home/src/Reliable
./scripts/llm_experiments/pack_cub_paper_runs_for_cloud.sh
# 可选减小体积（去掉 *.log）：PACK_STRIP_LOGS=1 ./scripts/llm_experiments/pack_cub_paper_runs_for_cloud.sh
scp -P <端口> /tmp/reliable_cub_paper_qwen_*.tar.gz root@<主机>:/tmp/
```

云上解压：

```bash
mkdir -p /home/src/Reliable/runs/bc-llm
tar -xzf /tmp/reliable_cub_paper_qwen_*.tar.gz -C /home/src/Reliable/runs/bc-llm/
```

**上传后验收（在云 SSH 执行）**：

```bash
find /home/src/Reliable/runs/bc-llm/cub_paper_qwen/tasks -name results.csv | wc -l
# 期望输出：10

bash /home/src/Reliable/scripts/llm_experiments/check_cub_paper_runs_ready.sh
# 期望：OK: 10 tasks with results.csv ...
```

参考体量（报告 §8.3）：整个 `cub_paper_qwen` 约 **380MB**（含 DuckDB 缓存与日志）。

---

### 3.2 二选一：CUB 图像数据

**用途**：`assemble_cub_birds.py` 读图；没有则任何任务无法开始。

| | 路径 |
|---|------|
| **本机（源，若 rsync）** | `/home/src/Reliable/src/bc-llm/exp_cub_birds_existing/data/CUB_200_2011/` |
| **云上（目标）** | `/home/src/Reliable/src/bc-llm/exp_cub_birds_existing/data/CUB_200_2011/` |

目录内需至少包含：

- `attributes.txt`
- `CUB_200_2011/images/`（及 `images.txt`、`image_class_labels.txt` 等，与笔记本一致即可）

**本机 rsync 示例**：

```bash
rsync -avz --progress -e 'ssh -p <端口>' \
  /home/src/Reliable/src/bc-llm/exp_cub_birds_existing/data/CUB_200_2011/ \
  root@<主机>:/home/src/Reliable/src/bc-llm/exp_cub_birds_existing/data/CUB_200_2011/
```

**可不从本机传**：若云上 AutoDL 有公共数据，可在 GPU 实例上本地解压（约 1.1GB 压缩包）：

```bash
mkdir -p /home/src/Reliable/src/bc-llm/exp_cub_birds_existing/data/CUB_200_2011
tar -xzf /autodl-pub/data/CUB200-2011/CUB_200_2011.tgz \
  -C /home/src/Reliable/src/bc-llm/exp_cub_birds_existing/data/CUB_200_2011
test -f /home/src/Reliable/src/bc-llm/exp_cub_birds_existing/data/CUB_200_2011/attributes.txt && \
  test -d /home/src/Reliable/src/bc-llm/exp_cub_birds_existing/data/CUB_200_2011/CUB_200_2011/images && echo CUB OK
```

---

### 3.3 建议上传或云上自建：`.venvs/bc-llm`

**用途**：跑 `run_bc_llm_cub_paper_task.sh` 中的 Python 阶段（含 `train_bayesian.py` 等）。

| | 路径 |
|---|------|
| **本机（源，若 rsync）** | `/home/src/Reliable/.venvs/bc-llm/` |
| **云上（目标）** | `/home/src/Reliable/.venvs/bc-llm/` |

**本机 rsync（省时间，体积较大）**：

```bash
rsync -avz --progress -e 'ssh -p <端口>' \
  /home/src/Reliable/.venvs/bc-llm/ \
  root@<主机>:/home/src/Reliable/.venvs/bc-llm/
```

**或在云上按 freeze 安装**（不传 venv 时，在 GPU 实例上执行，耗时长）：

```bash
cd /home/src/Reliable
# 需 Python 3.10 + CUDA 与 environments/bc-llm.freeze.txt 一致；详见 environments/README.md
python3.10 -m venv .venvs/bc-llm
.venvs/bc-llm/bin/pip install -r environments/bc-llm.freeze.txt
.venvs/bc-llm/bin/python -c "import torch, duckdb; print(torch.__version__, torch.cuda.is_available())"
```

---

### 3.4 一般无需再传

| 内容 | 云上状态 |
|------|----------|
| `.env.llm` | ✅ 已有；仅当本机 Key/模型名更新时才 `scp` 覆盖 |
| Git 仓库 | ✅ 已在云上；本机有新提交时再 `git pull` |
| `runs/c2bm`、`runs/planned_api_experiments` 等 | 与方案 A 无关，可不传 |

本机密钥上传示例（仅当需要覆盖时）：

```bash
scp -P <端口> /home/src/Reliable/.env.llm root@<主机>:/home/src/Reliable/.env.llm
chmod 600 /home/src/Reliable/.env.llm   # 在云执行
```

---

## 4. 上传完成后的云上自检清单

在 **已挂载 GPU** 的实例上依次执行：

```bash
cd /home/src/Reliable

nvidia-smi
test -x .venvs/bc-llm/bin/python && .venvs/bc-llm/bin/python -c "import torch; print(torch.cuda.get_device_name(0))"

bash scripts/llm_experiments/check_cub_paper_runs_ready.sh

test -f src/bc-llm/exp_cub_birds_existing/data/CUB_200_2011/attributes.txt
test -d src/bc-llm/exp_cub_birds_existing/data/CUB_200_2011/CUB_200_2011/images

./scripts/llm_experiments/check_api_connections.sh qwen   # 可选；默认用 c2bm venv，可改用 bc-llm python 调 API
```

全部通过后启动续跑（后台）：

```bash
./scripts/llm_experiments/start_bc_llm_cub_paper_wave_a_remaining.sh
tail -f runs/bc-llm/cub_paper_qwen/pipeline.log
./scripts/llm_experiments/status.sh
```

---

## 5. 续跑任务范围（供核对）

- 清单文件：`scripts/llm_experiments/bc_llm_paper_all_tasks.txt`（34 任务）
- 已有 `results.csv` 的 **10** 个任务自动 **SKIP**
- 待跑 **24** 个任务名称见 `report/20261022实验复现与后续研究设计.md` §8.1

---

## 6. 费用提醒

24 任务前请在百炼控制台核对余额。无本机 `api_cache.duckdb` 迁移时，已完成的 10 个任务若被重跑会 **重复计费**（故 §3.1 为必传项）。

---

## 7. 相关仓库文件

| 文件 | 作用 |
|------|------|
| `scripts/llm_experiments/pack_cub_paper_runs_for_cloud.sh` | 本机打包 `cub_paper_qwen` |
| `scripts/llm_experiments/sync_cub_paper_runs_to_cloud.sh` | 本机 rsync 到云 |
| **`scripts/llm_experiments/unpack_staging_from_tmp_doc.sh`** | **从 `tmp_doc/` 解压到正式路径** |
| `scripts/llm_experiments/check_cub_paper_runs_ready.sh` | 云上 preflight |
| `scripts/llm_experiments/start_bc_llm_cub_paper_wave_a_remaining.sh` | 启动剩余方案 A |
| `report/20261022实验复现与后续研究设计.md` §8 | 完整操作笔记 |
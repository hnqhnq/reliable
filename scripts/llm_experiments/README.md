# 国产 LLM API 实验运行说明

## 1. 填写密钥

编辑根目录的 `/home/src/Reliable/.env.llm`，只填写：

```bash
DASHSCOPE_API_KEY='你的百炼 Key'
DEEPSEEK_API_KEY='你的 DeepSeek Key'
```

该文件已加入根目录 `.gitignore`，权限应保持为 `600`。实验日志不会输出 Key。

## 2. 先运行低成本 demo 门禁

```bash
cd /home/src/Reliable
./scripts/llm_experiments/start_demo_pipeline.sh
./scripts/llm_experiments/status.sh
```

跟踪 demo：

```bash
tail -f /home/src/Reliable/runs/llm_route_demo/pipeline.log
```

这个 demo 会依次验证：

1. 百炼、DeepSeek 的 Key、模型名和严格 A/B/C 输出；
2. C²BM 的 `GES → RAG → Qwen 图补全 → 3 epoch GPU 训练/测试`；
3. BC-LLM 的 Qwen 图像输入、Pydantic 结构化抽取、候选概念生成和 DuckDB 缓存。

C²BM demo 每条边只查询 1 次，BC-LLM 只发送 1 张图片和 1 个文本概念请求，不能作为论文数值结果。只有全部检查通过才会生成：

```text
/home/src/Reliable/runs/llm_route_demo/acceptance.txt
```

若只想单独检查两个供应商的连接，可执行：

```bash
./scripts/llm_experiments/check_api_connections.sh all
```

## 3. demo 通过后，再运行当前阶段实验

```bash
cd /home/src/Reliable
./scripts/llm_experiments/start_planned_api_experiments.sh
```

执行顺序为：API 检查 → C²BM/Qwen → C²BM/DeepSeek → BC-LLM/Qwen CUB 小样本。
它们串行执行，避免抢占同一块 GPU。脚本启动后可以按 `Ctrl+C`、关闭终端或断开 SSH，任务不会停止。
若 demo 状态不是 `SUCCEEDED`，该脚本会拒绝启动，避免直接进入较多 API 调用。

查看状态：

```bash
./scripts/llm_experiments/status.sh
```

跟踪总日志：

```bash
tail -f /home/src/Reliable/runs/planned_api_experiments/pipeline.log
```

## 4. 单独运行

```bash
# Asia + predicted graph + seed 1（千问）
./scripts/llm_experiments/start_c2bm_asia_api.sh qwen

# 同一实验（DeepSeek）
./scripts/llm_experiments/start_c2bm_asia_api.sh deepseek

# BC-LLM CUB 40 张图像小样本
./scripts/llm_experiments/start_bc_llm_cub_smoke.sh
```

## 5. 断点与重跑语义

- C²BM：逐次 LLM 回答保存在 `requests.jsonl`，RAG 文本保存在
  `rag_contexts.jsonl`，完成图保存在 `completed_graph.pkl`，训练从
  `hydra/checkpoints/last.ckpt` 恢复。Qwen 与 DeepSeek 共用
  `runs/c2bm/asia_predicted_shared_seed1/ges_cpdag.pkl`，保证比较时输入的是
  完全相同的 GES CPDAG。
- BC-LLM：每个完整阶段有 `.done` 标记，LLM 回答保存在
  `api_cache.duckdb`。若在某个 Bayesian 阶段中途停止，该阶段会从头计算，
  但已成功的 API 回答会从 DuckDB 读取，不会重复付费。
- 失败后无需删除文件，重新执行同一个 `start_*.sh` 即可。

BC-LLM 默认使用 `qwen3.8-flash` 生成候选概念，使用更便宜的
`qwen3.7-flash-2026-07-15` 做图像描述和概念抽取。先完成这次小样本，
再决定是否把 `.env.llm` 中的 `QWEN_ITER_MODEL` 升级为
`qwen3.8-max-0902`。

## 6. 阶段 5：五 seed 训练（复用 seed 1 的图，不再调用图补全 API）

在 `start_planned_api_experiments.sh` 全部 `SUCCEEDED` 后：

```bash
cd /home/src/Reliable
./scripts/llm_experiments/summarize_planned_results.py
./scripts/llm_experiments/start_c2bm_asia_multiseed_training.sh
tail -f /home/src/Reliable/runs/c2bm/asia_multiseed_training/pipeline.log
```

汇总表写入 `runs/planned_api_experiments/evaluation_summary.md`。
多 seed 任务串行占用 GPU，约 30～45 分钟完成 8 次训练（Qwen/DeepSeek × seed 2–5）。

单独补跑某一 seed：

```bash
./scripts/llm_experiments/run_c2bm_asia_train_seed.sh qwen 3
```

## 7. 阶段 6：BC-LLM CUB 扩大试点（计划 §6）

C²BM Asia 五 seed 完成后，扩大 CUB 样本与 Bayesian epoch（仍非论文 37 任务全量）：

```bash
cd /home/src/Reliable
./scripts/llm_experiments/start_bc_llm_cub_pilot.sh
tail -f /home/src/Reliable/runs/bc-llm/cub_api_pilot_qwen/pipeline.log
```

默认：`80` 张图、类别 34/35、`num-greedy-epochs=2`、`max-epochs=2`，独立目录
`runs/bc-llm/cub_api_pilot_qwen/` 与 DuckDB 缓存。可通过环境变量覆盖，例如
`BC_CUB_MAX_OBS=100 ./scripts/llm_experiments/run_bc_llm_cub_pilot.sh`。

## 8. 方案 B：BC-LLM 论文设定 10 任务子集

与 `exp_cub_birds_existing/sconscript` 对齐：`NUM_GREEDY=1`、`max_epochs=min(5,25/K)`、50/50、每任务独立 DuckDB。

```bash
cd /home/src/Reliable
./scripts/llm_experiments/start_bc_llm_cub_paper_wave_b.sh
tail -f runs/bc-llm/cub_paper_qwen/pipeline.log
cat runs/bc-llm/cub_paper_qwen/wave_summary.md   # 结果 + 粗算花费
```

补全方案 A（34 任务，**跳过已有 `results.csv` 的任务**；建议后台）：

```bash
./scripts/llm_experiments/start_bc_llm_cub_paper_wave_a_remaining.sh
tail -f runs/bc-llm/cub_paper_qwen/pipeline.log
```

前台调试时可运行 `./scripts/llm_experiments/run_bc_llm_cub_paper_wave_a_remaining.sh`。
云服务器迁移与 24 任务清单见 `report/20261022实验复现与后续研究设计.md` **§8**。

**`runs/` 不要进 Git**（体积大、DuckDB 缓存）。同步方案 B 产物到云：

```bash
CLOUD_SSH='user@host' ./scripts/llm_experiments/sync_cub_paper_runs_to_cloud.sh
# 或 ./scripts/llm_experiments/pack_cub_paper_runs_for_cloud.sh 后 scp tar 包
```

任务列表：`bc_llm_paper_wave_b_tasks.txt`（10）、`bc_llm_paper_all_tasks.txt`（34）。

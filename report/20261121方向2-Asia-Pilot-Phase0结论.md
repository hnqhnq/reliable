# 方向 2 Asia Pilot · Phase 0 结论（方案 a）

**执行时间**：2026-11-20 16:14–18:21（UTC+8）  
**状态**：**已闭合 · 未通过**（不进入 Phase 1 / 完整 MH）  
**设计稿**：`report/20261120方向2-Asia-Pilot实验设计.md`  
**原始数据**（不入 Git）：`runs/direction2/asia_pilot_phase0/phase0_runs/`

---

## 1. 结论先行（导师可读）

在 Asia 的 GES CPDAG 上，对 **2 条无向边** 各用 oracle **正确 vs 反向** 定向，在 **30 次** 独立数据划分下比较 C²BM **留出集 `val_loss`**（方案 a）。  
**正确方向的留出似然并未稳定优于反向**：两条边 win rate 分别为 **33.3%**、**46.7%**，整体接近随机。  
因此 **不宜** 将「神经 SCM 留出似然」作为 split-sample 边定向的接受得分；**不建议** 启动 Phase 1（LLM 1 票 + 同一接受率）或 E1 主实验。  
建议与导师在 **方案 b（结构级纠错）**、**方案 c（干预数据）**、**方向 4（概念不完整 × 公平）** 中择一下一阶段。

---

## 2. 实验设定（与 Phase 0 设计一致）

| 项 | 值 |
|----|-----|
| 数据 | Asia BN，`dataset_n_samples=10000`，`val_size=0.5`（留出作 \(D_S^c\)） |
| 图 | 以 Qwen seed1 `completed_graph.pkl` 为模板，仅翻转待测无向边 |
| 无向边 | `smoke--lung`、`smoke--bronc`（CPDAG 中共 2 条） |
| 划分 | \(R=30\)，seed \(=10000+\)split_id |
| 训练 | AE 最多 120 epoch；C²BM `max_epochs=80`，`patience=15`；无 LLM API |
| 指标 | 训练结束时的 `val_loss`（越低越好）；比较 \(\mathcal{L}^- - \mathcal{L}^+\) |

---

## 3. 结果

| 边 | 30 split 上正确方向胜出次数 | **win rate** |
|----|---------------------------|--------------|
| `smoke--lung` | 10 / 30 | **0.333** |
| `smoke--bronc` | 14 / 30 | **0.467** |
| **简单平均** | — | **0.400** |

- 总训练任务：**120**（2 边 × 2 定向 × 30 split）  
- 成功：**118** `ok` + **2** `cached`（冒烟复用），**0** 失败  
- 汇总文件：`phase0_win_rates.txt`、`phase0_edge_summary.csv`、`phase0_runs.csv`

---

## 4. 判定（对照设计稿 §5）

| 标准 | 结果 |
|------|------|
| 强通过（边 ≥0.7，均 ≥0.75） | 否 |
| 弱通过（均 ≥0.6） | 否 |
| **未通过**（均 ≤0.55 或近 0.5） | **是** |

**解读**：在纯观测、同一马尔可夫等价类内仅换箭头时，当前 C²BM 训练流程下的 `val_loss` **不能** 充当可靠的数据裁判。这与 §4.3 中「经典得分等价 → α≈1」的风险一致；方案 a 的「神经网络打破等价」在本设定中 **未带来可定向辨别力**。

---

## 5. 不建议的下一步

- Phase 1：LLM 提议 + 方案 a 接受（预计接受率退化，API 性价比低）  
- 加长网格 / 大幅调参重跑 Phase 0（win rate 与 0.5 差距不足以支撑再赌）

---

## 6. 建议的下一步（待导师拍板）

| 选项 | 内容 | 适合若… |
|------|------|---------|
| **方向 4** | CelebA 隐藏中介 → 公平/干预保证是否失效 | 希望换新故事、弱化「MEC 内定向」 |
| **方案 b Pilot** | 错边/删边/v-structure 注入 + 数据或 BIC 纠错 | 仍要「数据把关」，但不声称纠正等价类内箭头 |
| **方案 c Pilot** | Asia 上模拟干预样本 | 必须坚持「定向」且接受干预设定 |
| **E2 型实证** | RAG 偏差 × 10 票基线（不做 MH） | 先要可汇报曲线，再定机制 |

---

## 7. 仓库与复现

```bash
# 仅统计（runs 在本地/云上，不入 Git）
cat runs/direction2/asia_pilot_phase0/phase0_runs/phase0_win_rates.txt
```

脚本：`scripts/llm_experiments/direction2_pilot_phase0.py`、`run_direction2_pilot_phase0.sh`；`main.py` 支持 `+pilot_export_val`、`+pilot_skip_test`。

---

*文档版本：2026-11-21*
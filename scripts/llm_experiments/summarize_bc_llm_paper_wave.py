#!/usr/bin/env python3
"""Summarize BC-LLM paper-wave task runs, aggregate metrics, estimate API cache usage."""

from __future__ import annotations

import statistics
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path("/home/src/Reliable")
WAVE_ROOT = ROOT / "runs/bc-llm/cub_paper_qwen"
OUT = WAVE_ROOT / "wave_summary.md"

# Reference pricing (plan §9, Qwen blend rough estimate per cached call)
EST_YUAN_PER_CALL = 0.015


def load_task_list(path: Path) -> list[str]:
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tasks.append(line)
    return tasks


def cache_rows(db_path: Path) -> int:
    if not db_path.is_file():
        return 0
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        return int(con.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0])
    except duckdb.CatalogException:
        return 0
    finally:
        con.close()


def main() -> int:
    manifest_b = ROOT / "scripts/llm_experiments/bc_llm_paper_wave_b_tasks.txt"
    manifest_all = ROOT / "scripts/llm_experiments/bc_llm_paper_all_tasks.txt"
    wave_b = set(load_task_list(manifest_b))
    all_tasks = load_task_list(manifest_all)

    rows = []
    total_calls = 0
    for task in sorted(set(all_tasks) | wave_b):
        task_dir = WAVE_ROOT / "tasks" / task
        status = (task_dir / "status.txt").read_text().strip() if (task_dir / "status.txt").is_file() else "—"
        calls = cache_rows(task_dir / "api_cache.duckdb")
        total_calls += calls
        acc = auc = None
        res = task_dir / "results.csv"
        if res.is_file():
            df = pd.read_csv(res)
            if len(df):
                acc = float(df.iloc[-1]["acc"])
                auc = float(df.iloc[-1]["auc"])
        rows.append(
            {
                "task": task,
                "wave_b": task in wave_b,
                "status": status,
                "acc": acc,
                "auc": auc,
                "cache_calls": calls,
            }
        )

    df = pd.DataFrame(rows)
    agg_dir = WAVE_ROOT / "aggregate"
    agg_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(agg_dir / "tasks_status.csv", index=False)

    succeeded_b = df[(df["wave_b"]) & (df["status"] == "SUCCEEDED")]
    result_files = [
        str(WAVE_ROOT / "tasks" / t / "results.csv")
        for t in succeeded_b["task"]
        if (WAVE_ROOT / "tasks" / t / "results.csv").is_file()
    ]
    if result_files:
        parts = [pd.read_csv(p) for p in result_files]
        by_task = pd.concat(parts, ignore_index=True)
        by_task.to_csv(agg_dir / "results_by_task.csv", index=False)
        if "acc" in by_task.columns:
            mean_acc = by_task["acc"].mean()
            mean_auc = by_task["auc"].mean()
        else:
            mean_acc = mean_auc = None
    else:
        mean_acc = mean_auc = None

    est_cost = total_calls * EST_YUAN_PER_CALL
    lines = [
        "# BC-LLM paper wave summary",
        "",
        f"Wave root: `{WAVE_ROOT}`",
        "",
        "## Progress",
        "",
        f"- Wave B tasks (planned): **{len(wave_b)}**",
        f"- All repo tasks (scheme A total): **{len(all_tasks)}**",
        f"- Tasks with any status row below: **{len(df)}**",
        f"- Wave B SUCCEEDED: **{len(succeeded_b)}** / {len(wave_b)}",
        "",
        "## Aggregated metrics (completed Wave B tasks only)",
        "",
    ]
    if mean_acc is not None:
        lines.append(f"- Mean test accuracy (unweighted over tasks): **{mean_acc:.4f}**")
        lines.append(f"- Mean test AUC: **{mean_auc:.4f}**")
        lines.append(
            "- Compare to paper BC-LLM CUB: 0.680 (0.614, 0.747) acc — "
            "**only valid as a 10-task subset**, not full Table 1."
        )
    else:
        lines.append("_No completed task results yet._")

    lines.extend(
        [
            "",
            "## API usage (estimate)",
            "",
            f"- Total `llm_cache` rows (all tasks under wave root): **{total_calls}**",
            f"- Rough spend (@ ~¥{EST_YUAN_PER_CALL}/call heuristic): **¥{est_cost:.2f}**",
            "- Treat as lower-bound order of magnitude; verify against provider billing.",
            "",
            "## Per task",
            "",
            "| Task | Wave B | Status | Acc | AUC | Cache calls |",
            "|---|:---:|---|---:|---:|---:|",
        ]
    )
    for _, r in df.sort_values("task").iterrows():
        acc_s = f"{r['acc']:.4f}" if r["acc"] is not None and r["acc"] == r["acc"] else "—"
        auc_s = f"{r['auc']:.4f}" if r["auc"] is not None and r["auc"] == r["auc"] else "—"
        lines.append(
            f"| {r['task']} | {'yes' if r['wave_b'] else 'no'} | {r['status']} | {acc_s} | {auc_s} | {int(r['cache_calls'])} |"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

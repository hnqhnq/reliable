#!/usr/bin/env python3
"""Generate PNG charts for BC-LLM CUB paper Wave A progress report."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WAVE = Path("/home/src/Reliable/runs/bc-llm/cub_paper_qwen")
TASK_FILE = ROOT / "scripts/llm_experiments/bc_llm_paper_all_tasks.txt"
OUT_DIR = ROOT / "report" / "figures"
MANIFEST_B = ROOT / "scripts/llm_experiments/bc_llm_paper_wave_b_tasks.txt"


def load_tasks() -> list[str]:
    lines = TASK_FILE.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


def wave_b_tasks() -> set[str]:
    lines = MANIFEST_B.read_text(encoding="utf-8").splitlines()
    return {l.strip() for l in lines if l.strip() and not l.startswith("#")}


def task_metrics() -> pd.DataFrame:
    rows = []
    wb = wave_b_tasks()
    for task in load_tasks():
        td = WAVE / "tasks" / task
        has = td.joinpath("results.csv").is_file() and td.joinpath("results.csv").stat().st_size > 0
        acc = auc = None
        if has:
            df = pd.read_csv(td / "results.csv")
            if len(df):
                acc = float(df.iloc[-1]["acc"])
                auc = float(df.iloc[-1]["auc"])
        brier = None
        if has:
            df_res = pd.read_csv(td / "results.csv")
            if len(df_res):
                brier = float(df_res.iloc[-1]["brier"])
        rows.append(
            {
                "task": task,
                "short": task.replace("class_", "").replace("_", ",")[:28],
                "done": has,
                "wave_b": task in wb,
                "acc": acc,
                "auc": auc,
                "brier": brier,
            }
        )
    return pd.DataFrame(rows)


def pipeline_durations_since(ts_min: str) -> pd.DataFrame:
    plog = (WAVE / "pipeline.log").read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(r"(\d{4}-\d{2}-\d{2}T[\d:+]+) WAVE_A (START|DONE) (\S+)")
    starts: dict[str, datetime] = {}
    rows = []
    for tstr, op, task in pattern.findall(plog):
        if tstr < ts_min:
            continue
        if op == "START":
            starts[task] = datetime.fromisoformat(tstr.replace("+08:00", ""))
        elif op == "DONE" and task in starts:
            end = datetime.fromisoformat(tstr.replace("+08:00", ""))
            hours = (end - starts[task]).total_seconds() / 3600
            rows.append({"task": task, "start": starts[task], "end": end, "hours": hours})
            del starts[task]
    return pd.DataFrame(rows)


def chart_completion_pie(df: pd.DataFrame) -> None:
    done = int(df["done"].sum())
    total = len(df)
    pending = total - done
    fig, ax = plt.subplots(figsize=(5, 5), dpi=150)
    sizes = [done, pending]
    labels = [f"Done\n{done}", f"Pending\n{pending}"]
    colors = ["#2ecc71", "#ecf0f1"]
    explode = (0.02, 0)
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 5 else "",
        startangle=90,
        textprops={"fontsize": 11},
    )
    ax.set_title(f"Wave A progress ({done}/{total})", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wave_a_completion_pie.png", bbox_inches="tight")
    plt.close(fig)


def chart_metrics_scatter(df: pd.DataFrame) -> None:
    sub = df[df["done"]].copy()
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    colors = ["#3498db" if b else "#e67e22" for b in sub["wave_b"]]
    ax.scatter(sub["acc"], sub["auc"], c=colors, alpha=0.85, s=60, edgecolors="white", linewidths=0.5)
    for _, r in sub.iterrows():
        ax.annotate(
            r["short"],
            (r["acc"], r["auc"]),
            fontsize=6,
            alpha=0.75,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_xlabel("Test accuracy", fontsize=11)
    ax.set_ylabel("Test AUC", fontsize=11)
    ax.set_xlim(0.35, 1.02)
    ax.set_ylim(0.68, 1.02)
    ax.grid(True, alpha=0.3)
    from matplotlib.lines import Line2D

    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db", label="Wave B (10, prior)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22", label="Wave A (new)"),
        ],
        loc="lower right",
        fontsize=9,
    )
    ax.set_title("Test acc vs AUC (completed tasks)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wave_a_metrics_scatter.png", bbox_inches="tight")
    plt.close(fig)


def chart_timeline(dur_df: pd.DataFrame) -> None:
    if dur_df.empty:
        return
    df = dur_df.sort_values("start").tail(15)
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    labels = []
    for i, row in enumerate(df.itertuples()):
        start = mdates.date2num(row.start)
        width = mdates.date2num(row.end) - start
        ax.broken_barh([(start, width)], (i - 0.3, 0.6), facecolors="#3498db", alpha=0.85)
        labels.append(row.task.replace("class_", "")[:32])
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    ax.set_xlabel("Time (UTC+8)", fontsize=10)
    ax.set_title("Task wall-clock windows (stable run)", fontsize=12, fontweight="bold")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wave_a_task_timeline.png", bbox_inches="tight")
    plt.close(fig)


def chart_duration_bars(dur_df: pd.DataFrame) -> None:
    if dur_df.empty:
        return
    df = dur_df.sort_values("hours").tail(12)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    labels = [t.replace("class_", "")[:36] for t in df["task"]]
    ax.barh(labels, df["hours"], color="#9b59b6", alpha=0.9)
    ax.set_xlabel("Wall time (hours)", fontsize=10)
    ax.set_title("Recent task durations", fontsize=12, fontweight="bold")
    for i, h in enumerate(df["hours"]):
        ax.text(h + 0.03, i, f"{h:.2f}h", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wave_a_task_duration.png", bbox_inches="tight")
    plt.close(fig)


def chart_paper_comparison(df: pd.DataFrame) -> None:
    """Macro mean over completed tasks vs paper Table 1 BC-LLM (GPT-4o-mini)."""
    sub = df[df["done"]]
    if sub.empty:
        return
    paper = {
        "acc": (0.680, 0.614, 0.747),
        "auc": (0.874, 0.840, 0.907),
        "brier": (0.428, 0.357, 0.500),
    }
    ours = {
        "acc": sub["acc"].mean(),
        "auc": sub["auc"].mean(),
        "brier": sub["brier"].mean() if "brier" in sub.columns else None,
    }
    # add brier from results files
    briers = []
    for task in sub["task"]:
        p = WAVE / "tasks" / task / "results.csv"
        if p.is_file():
            briers.append(float(pd.read_csv(p).iloc[-1]["brier"]))
    if briers:
        ours["brier"] = float(np.mean(briers))

    metrics = ["acc", "auc", "brier"]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    x = np.arange(len(metrics))
    w = 0.35
    paper_pts = [paper[m][0] for m in metrics]
    ours_pts = [ours[m] for m in metrics]
    ax.bar(x - w / 2, paper_pts, w, label="Paper BC-LLM (GPT-4o-mini, Table 1)", color="#95a5a6", alpha=0.9)
    ax.bar(x + w / 2, ours_pts, w, label=f"Ours Qwen (macro, n={len(sub)})", color="#e67e22", alpha=0.9)
    for i, m in enumerate(metrics):
        _, lo, hi = paper[m]
        ax.hlines(lo, i - 0.5, i - 0.05, colors="#7f8c8d", linestyles="--", linewidth=1)
        ax.hlines(hi, i - 0.5, i - 0.05, colors="#7f8c8d", linestyles="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Value (lower Brier is better)")
    ax.set_title("Macro mean vs paper aggregate (interim, incomplete tasks)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wave_a_paper_comparison.png", bbox_inches="tight")
    plt.close(fig)


def chart_acc_bar_top_bottom(df: pd.DataFrame) -> None:
    sub = df[df["done"]].sort_values("acc")
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    x = np.arange(len(sub))
    w = 0.38
    ax.bar(x - w / 2, sub["acc"], w, label="acc", color="#2980b9", alpha=0.85)
    ax.bar(x + w / 2, sub["auc"], w, label="auc", color="#27ae60", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(sub["short"], rotation=60, ha="right", fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Metric", fontsize=10)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("acc / AUC by task (sorted by acc)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "wave_a_acc_auc_bars.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams["font.sans-serif"] = [
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = task_metrics()
    dur = pipeline_durations_since("2026-11-17T22:59")
    chart_completion_pie(df)
    chart_metrics_scatter(df)
    chart_paper_comparison(df)
    chart_acc_bar_top_bottom(df)
    # Optional diagnostics (not used in reproduction-facing report):
    # chart_timeline(dur); chart_duration_bars(dur)
    print(f"Wrote charts to {OUT_DIR} ({len(list(OUT_DIR.glob('*.png')))} PNG)")


if __name__ == "__main__":
    main()
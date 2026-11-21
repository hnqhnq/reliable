#!/usr/bin/env python3
"""Phase 0: oracle orientations — train on D_S, export val_loss as hold-out score."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "src/causally-reliable-cbm"
PYTHON = ROOT / ".venvs/c2bm/bin/python"
DEFAULT_MANIFEST = ROOT / "runs/direction2/asia_pilot_phase0/pilot_manifest.json"
DEFAULT_TEMPLATE = ROOT / "runs/c2bm/asia_predicted_qwen_seed1/completed_graph.pkl"
OUT_ROOT = ROOT / "runs/direction2/asia_pilot_phase0/phase0_runs"


def load_graph(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    raise TypeError(f"Expected DataFrame graph in {path}")


def orient_edge(graph: pd.DataFrame, node_i: str, node_j: str, forward_i_to_j: bool) -> pd.DataFrame:
    """Set directed edge between node_i and node_j; clear opposite arrow."""
    g = graph.copy()
    ii, jj = list(g.index).index(node_i), list(g.index).index(node_j)
    ni, nj = int(g.iloc[ii, jj]), int(g.iloc[jj, ii])
    if ni == -1 and nj == -1:
        pass
    elif (ni, nj) == (1, 0) or (ni, nj) == (0, 1):
        g.iloc[ii, jj] = 0
        g.iloc[jj, ii] = 0
    else:
        raise ValueError(f"Unexpected adjacency ({node_i},{node_j})=({ni},{nj})")
    if forward_i_to_j:
        g.iloc[ii, jj] = 1
        g.iloc[jj, ii] = 0
    else:
        g.iloc[ii, jj] = 0
        g.iloc[jj, ii] = 1
    return g


def run_training(
    *,
    graph_path: Path,
    run_dir: Path,
    seed: int,
    max_epochs: int,
    patience: int,
    val_size: float,
    export_path: Path,
) -> int:
    run_dir.mkdir(parents=True, exist_ok=True)
    hydra_dir = run_dir / "hydra"
    env = os.environ.copy()
    env.setdefault("C2BM_CACHE", str(ROOT / ".cache/c2bm"))
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    env.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
    env.setdefault("TORCH_HOME", str(ROOT / ".cache/torch"))
    env.setdefault("CUDA_VISIBLE_DEVICES", env.get("CUDA_VISIBLE_DEVICES", "0"))
    env["PYTHONUNBUFFERED"] = "1"

    cmd = [
        str(PYTHON),
        "main.py",
        "--config-name",
        "default",
        "dataset=asia",
        "model=c2bm",
        "causal_discovery=ges",
        "llm=qwen",
        "rag=standard",
        "+dataset.load_embeddings=false",
        "+dataset.load_true_graph=false",
        "+dataset.load_graph=true",
        "+dataset.num_workers=0",
        "dataset.batch_size=512",
        "dataset.loader.dataset_n_samples=10000",
        f"dataset.loader.val_size={val_size}",
        "dataset.loader.test_size=0.01",
        "dataset.autoencoder.noise=0.5",
        "dataset.autoencoder.epochs=120",
        "dataset.autoencoder.patience=25",
        "model.hidden_size=64",
        "model.concept_hidden_size=4",
        "model.concept_loss_weight=0.8",
        "engine.optim_kwargs.lr=0.0004",
        "engine.intervention_prob=0.8",
        "engine.test_interv_noise=0.8",
        "+policy=nodes_true",
        f"trainer.max_epochs={max_epochs}",
        f"trainer.patience={patience}",
        "+trainer.logger=null",
        "+trainer.devices=[0]",
        "+trainer.enable_progress_bar=false",
        f"completion.graph_path={graph_path}",
        f"seed={seed}",
        f"hydra.run.dir={hydra_dir}",
        f"+pilot_export_val={export_path}",
        "+pilot_skip_test=true",
    ]
    log_path = run_dir / "train.log"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n===== seed={seed} graph={graph_path} =====\n")
        proc = subprocess.run(
            cmd,
            cwd=PROJECT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--template-graph", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val-size", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--only-edge", type=str, default=None, help="e.g. smoke--lung")
    args = parser.parse_args()

    if not PYTHON.is_file():
        print(f"missing venv: {PYTHON}", file=sys.stderr)
        return 1
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    defaults = manifest.get("pilot_defaults", {})
    n_splits = args.n_splits if args.n_splits is not None else int(defaults.get("n_splits", 30))
    max_epochs = args.max_epochs if args.max_epochs is not None else int(defaults.get("phase0_max_epochs", 80))

    template = load_graph(args.template_graph)
    edges = manifest["undirected_edges"]
    if args.only_edge:
        edges = [e for e in edges if e["edge_id"] == args.only_edge]
        if not edges:
            print(f"edge not in manifest: {args.only_edge}", file=sys.stderr)
            return 1

    rows: list[dict] = []
    graphs_dir = OUT_ROOT / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_ROOT / "phase0_runs.csv"

    for edge in edges:
        node_i, node_j = edge["node_i"], edge["node_j"]
        oracle = edge.get("oracle") or {}
        for label in ("correct", "reversed"):
            correct_is_i_to_j = oracle.get("correct") == "i->j"
            forward = correct_is_i_to_j if label == "correct" else not correct_is_i_to_j
            g = orient_edge(template, node_i, node_j, forward_i_to_j=forward)
            graph_path = graphs_dir / f"{edge['edge_id']}_{label}.pkl"
            with graph_path.open("wb") as f:
                pickle.dump(g, f)

            for split_id in range(n_splits):
                seed = 10_000 + split_id
                run_dir = OUT_ROOT / edge["edge_id"] / label / f"split_{split_id:03d}"
                export_path = run_dir / "val_metrics.json"
                row = {
                    "edge_id": edge["edge_id"],
                    "orientation": label,
                    "split_id": split_id,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "graph_path": str(graph_path),
                    "status": "pending",
                    "val_loss": "",
                }
                if args.dry_run:
                    row["status"] = "dry_run"
                    rows.append(row)
                    continue
                if export_path.is_file():
                    metrics = json.loads(export_path.read_text(encoding="utf-8"))
                    row["val_loss"] = metrics.get("val_loss", "")
                    row["status"] = "cached"
                    rows.append(row)
                    continue
                rc = run_training(
                    graph_path=graph_path,
                    run_dir=run_dir,
                    seed=seed,
                    max_epochs=max_epochs,
                    patience=args.patience,
                    val_size=args.val_size,
                    export_path=export_path,
                )
                if rc != 0:
                    row["status"] = f"failed_rc_{rc}"
                elif export_path.is_file():
                    metrics = json.loads(export_path.read_text(encoding="utf-8"))
                    row["val_loss"] = metrics.get("val_loss", "")
                    row["status"] = "ok"
                else:
                    row["status"] = "missing_metrics"
                rows.append(row)
                _write_summary(rows, summary_path)
                print(f"{edge['edge_id']} {label} split={split_id} -> {row['status']} val_loss={row['val_loss']}")

    _write_summary(rows, summary_path)
    aggregate_path = OUT_ROOT / "phase0_edge_summary.csv"
    _aggregate(rows, aggregate_path)
    print(f"wrote {summary_path}")
    print(f"wrote {aggregate_path}")
    return 0


def _write_summary(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["edge_id", "orientation", "split_id", "seed", "status", "val_loss", "run_dir", "graph_path"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _aggregate(rows: list[dict], path: Path) -> None:
    from collections import defaultdict

    by_key: dict[tuple[str, str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("val_loss") == "" or row.get("val_loss") is None:
            continue
        try:
            loss = float(row["val_loss"])
        except (TypeError, ValueError):
            continue
        by_key[(row["edge_id"], row["split_id"])][row["orientation"]] = loss

    out_rows = []
    win_counts: dict[str, list[int]] = defaultdict(list)
    for (edge_id, split_id), losses in sorted(by_key.items()):
        if "correct" not in losses or "reversed" not in losses:
            continue
        delta = losses["reversed"] - losses["correct"]
        win = 1 if delta > 0 else 0
        win_counts[edge_id].append(win)
        out_rows.append(
            {
                "edge_id": edge_id,
                "split_id": split_id,
                "val_loss_correct": losses["correct"],
                "val_loss_reversed": losses["reversed"],
                "delta_reversed_minus_correct": delta,
                "correct_wins": win,
            }
        )

    with path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "edge_id",
            "split_id",
            "val_loss_correct",
            "val_loss_reversed",
            "delta_reversed_minus_correct",
            "correct_wins",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    summary_txt = OUT_ROOT / "phase0_win_rates.txt"
    with summary_txt.open("w", encoding="utf-8") as f:
        for edge_id, wins in sorted(win_counts.items()):
            rate = sum(wins) / len(wins) if wins else 0.0
            f.write(f"{edge_id}\tn_splits={len(wins)}\twin_rate={rate:.3f}\n")


if __name__ == "__main__":
    raise SystemExit(main())
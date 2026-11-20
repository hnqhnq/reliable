#!/usr/bin/env python3
"""List undirected edges in a GES CPDAG and label oracle orientations from the true DAG."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CPDAG = ROOT / "runs/c2bm/asia_predicted_shared_seed1/ges_cpdag.pkl"
DEFAULT_OUT = ROOT / "runs/direction2/asia_pilot_phase0/pilot_manifest.json"


def edge_type(adj: np.ndarray, i: int, j: int) -> str:
    a_ij, a_ji = adj[i, j], adj[j, i]
    if a_ij == 1 and a_ji == 0:
        return "i->j"
    if a_ij == 0 and a_ji == 1:
        return "j->i"
    if a_ij == -1 and a_ji == -1:
        return "undirected"
    if a_ij == 1 and a_ji == 1:
        return "bidirected"
    return "none"


def load_graph_pickle(path: Path) -> pd.DataFrame:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, pd.DataFrame):
        return obj
    if isinstance(obj, dict) and "graph" in obj:
        g = obj["graph"]
        if isinstance(g, pd.DataFrame):
            return g
    raise TypeError(f"Unsupported pickle type in {path}: {type(obj)}")


def load_true_dag_asia() -> pd.DataFrame | None:
    """Best-effort load of Asia BN true adjacency from project data."""
    try:
        sys.path.insert(0, str(ROOT / "src/causally-reliable-cbm"))
        from src.data.datasets.bndataset import BNDataset  # noqa: WPS433

        ds = BNDataset(dag_name="asia")
        adj = ds.load_ground_truth_graph()
        labels = list(adj.index)
        return pd.DataFrame(adj.values.astype(int), index=labels, columns=labels)
    except Exception as exc:  # noqa: BLE001
        print(f"warning: could not load Asia true graph ({exc})", file=sys.stderr)
    return None


def oracle_orientation(
    names: list[str], i: int, j: int, true_adj: np.ndarray | None
) -> dict | None:
    if true_adj is None:
        return None
    ti, tj = true_adj[i, j], true_adj[j, i]
    if ti == 1 and tj == 0:
        return {"correct": "i->j", "reversed": "j->i"}
    if ti == 0 and tj == 1:
        return {"correct": "j->i", "reversed": "i->j"}
    return {"correct": None, "reversed": None, "note": "not a directed edge in true DAG"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpdag", type=Path, default=DEFAULT_CPDAG)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.cpdag.is_file():
        print(f"error: missing CPDAG: {args.cpdag}", file=sys.stderr)
        return 1

    cpdag = load_graph_pickle(args.cpdag)
    names = list(cpdag.index)
    adj = cpdag.values.astype(int)
    n = len(names)

    true_graph = load_true_dag_asia()
    true_adj = None
    if true_graph is not None:
        if list(true_graph.index) != names:
            remap = {name: names.index(name) for name in true_graph.index if name in names}
            true_adj = np.zeros((n, n), dtype=int)
            for a, ia in remap.items():
                for b, ib in remap.items():
                    true_adj[ia, ib] = int(true_graph.loc[a, b])
        else:
            true_adj = true_graph.values.astype(int)

    undirected: list[dict] = []
    for i in range(n):
        for j in range(i + 1, n):
            if edge_type(adj, i, j) != "undirected":
                continue
            undirected.append(
                {
                    "edge_id": f"{names[i]}--{names[j]}",
                    "i": i,
                    "j": j,
                    "node_i": names[i],
                    "node_j": names[j],
                    "oracle": oracle_orientation(names, i, j, true_adj),
                }
            )

    payload = {
        "cpdag_path": str(args.cpdag),
        "n_nodes": n,
        "node_labels": names,
        "n_undirected_edges": len(undirected),
        "undirected_edges": undirected,
        "pilot_defaults": {"split_ratio": 0.5, "n_splits": 30, "phase0_max_epochs": 80},
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({len(undirected)} undirected edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
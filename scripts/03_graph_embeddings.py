#!/usr/bin/env python
"""Build speaker-subject graph embeddings from LIAR metadata.

Creates a bipartite graph between speakers and subjects, fits Node2Vec, and
exports per-claim graph features using speaker vectors plus mean subject vectors.

Input:
- `data/processed/liar_base.parquet`

Outputs:
- `data/interim/graph_embeddings.parquet`
- `data/interim/speaker_subject_edges.parquet`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import networkx as nx

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liar_mining.config import ensure_dirs, load_config
from liar_mining.io import load_parquet, save_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create speaker-subject graph embeddings.")
    parser.add_argument("--config", default="configs/liar_research.yaml")
    return parser.parse_args()


def subject_list(value: object) -> list[str]:
    if pd.isna(value):
        return []
    items = [s.strip().lower() for s in str(value).split(",")]
    return [s for s in items if s]


def build_graph(df: pd.DataFrame) -> nx.Graph:
    g = nx.Graph()
    for _, row in df.iterrows():
        spk = f"spk::{row['speaker']}"
        subs = subject_list(row["subject"])
        if not subs:
            subs = ["unknown"]
        for sub in subs:
            sub_node = f"sub::{sub}"
            if g.has_edge(spk, sub_node):
                g[spk][sub_node]["weight"] += 1
            else:
                g.add_edge(spk, sub_node, weight=1)
    return g


def fit_node2vec(g: nx.Graph, cfg: dict) -> dict[str, np.ndarray]:
    from node2vec import Node2Vec

    n2v = Node2Vec(
        g,
        dimensions=cfg["graph"]["dimensions"],
        walk_length=cfg["graph"]["walk_length"],
        num_walks=cfg["graph"]["num_walks"],
        workers=cfg["graph"]["workers"],
        p=cfg["graph"]["p"],
        q=cfg["graph"]["q"],
        weight_key="weight",
    )
    model = n2v.fit(window=10, min_count=1, batch_words=64)

    vectors: dict[str, np.ndarray] = {}
    for node in g.nodes():
        vectors[node] = np.asarray(model.wv[node], dtype=np.float32)
    return vectors


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    base_path = Path(cfg["paths"]["processed_dir"]) / "liar_base.parquet"
    df = load_parquet(base_path)
    df["speaker"] = df["speaker"].fillna("unknown").astype(str)

    g = build_graph(df)
    vectors = fit_node2vec(g, cfg)

    dim = cfg["graph"]["dimensions"]
    zero_vec = np.zeros(dim, dtype=np.float32)

    rows = []
    for _, row in df.iterrows():
        spk_node = f"spk::{row['speaker']}"
        spk_vec = vectors.get(spk_node, zero_vec)

        sub_nodes = [f"sub::{s}" for s in subject_list(row["subject"])]
        if sub_nodes:
            sub_vecs = [vectors.get(n, zero_vec) for n in sub_nodes]
            sub_mean = np.mean(np.vstack(sub_vecs), axis=0)
        else:
            sub_mean = zero_vec

        fused = np.concatenate([spk_vec, sub_mean]).astype(np.float32)
        out_row = {"id": row["id"]}
        for i, v in enumerate(fused):
            out_row[f"graph_emb_{i}"] = float(v)
        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    out_path = Path(cfg["paths"]["interim_dir"]) / "graph_embeddings.parquet"
    save_parquet(out_df, out_path)

    edge_path = Path(cfg["paths"]["interim_dir"]) / "speaker_subject_edges.parquet"
    edges = pd.DataFrame([(u, v, d.get("weight", 1)) for u, v, d in g.edges(data=True)], columns=["node_u", "node_v", "weight"])
    save_parquet(edges, edge_path)

    print(f"Graph nodes={g.number_of_nodes()}, edges={g.number_of_edges()}")
    print(f"Saved graph embeddings: {out_path}")
    print(f"Saved graph edges: {edge_path}")


if __name__ == "__main__":
    main()

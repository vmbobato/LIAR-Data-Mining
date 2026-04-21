#!/usr/bin/env python
"""Cluster LIAR claims in embedding space for thematic exploration.

Runs KMeans over text, graph, or fused embeddings (with optional SVD reduction)
and saves claim-level cluster assignments plus label composition summaries.

Inputs:
- `data/processed/liar_base.parquet`
- Embedding tables in `data/interim/`

Outputs:
- `data/processed/clusters_<source>.parquet`
- `data/processed/clusters_<source>_label_distribution.csv`
- `data/processed/clusters_<source>_quality_metrics.json`
- `data/processed/clusters_<source>_projection.parquet`
- `reports/figures/clusters_<source>_by_cluster.png`
- `reports/figures/clusters_<source>_by_truth.png`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liar_mining.config import ensure_dirs, load_config
from liar_mining.io import load_parquet, save_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster LIAR statements using text/graph embeddings.")
    parser.add_argument("--config", default="configs/liar_research.yaml")
    parser.add_argument("--embedding-source", choices=["text", "graph", "fused"], default="text")
    parser.add_argument("--text-file", default="text_embeddings_tfidf.parquet")
    parser.add_argument("--n-clusters", type=int, default=None)
    parser.add_argument("--svd-dim", type=int, default=100)
    return parser.parse_args()


def get_feature_table(cfg: dict, source: str, text_file: str) -> pd.DataFrame:
    interim = Path(cfg["paths"]["interim_dir"])
    text_df = load_parquet(interim / text_file)

    if source == "text":
        return text_df

    graph_df = load_parquet(interim / "graph_embeddings.parquet")
    if source == "graph":
        return graph_df

    return text_df.merge(graph_df, on="id", how="inner")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    base = load_parquet(Path(cfg["paths"]["processed_dir"]) / "liar_base.parquet")
    feat_df = get_feature_table(cfg, args.embedding_source, args.text_file)

    merged = base[["id", "label", "is_low_truth", "split", "statement", "subject", "speaker"]].merge(
        feat_df, on="id", how="inner"
    )

    feature_cols = [c for c in merged.columns if c.startswith("text_emb_") or c.startswith("graph_emb_")]
    x = merged[feature_cols].to_numpy(dtype=np.float32)

    if x.shape[1] > args.svd_dim:
        svd = TruncatedSVD(n_components=args.svd_dim, random_state=42)
        x_used = svd.fit_transform(x)
    else:
        x_used = x

    n_clusters = args.n_clusters or cfg["clustering"]["n_clusters"]
    km = KMeans(n_clusters=n_clusters, random_state=cfg["clustering"]["random_state"], n_init=10)
    cluster_ids = km.fit_predict(x_used)
    merged["cluster_id"] = cluster_ids

    unique_clusters = np.unique(cluster_ids)
    if len(unique_clusters) > 1:
        sil = float(silhouette_score(x_used, cluster_ids))
        ch = float(calinski_harabasz_score(x_used, cluster_ids))
        db = float(davies_bouldin_score(x_used, cluster_ids))
    else:
        sil = float("nan")
        ch = float("nan")
        db = float("nan")

    out_path = Path(cfg["paths"]["processed_dir"]) / f"clusters_{args.embedding_source}.parquet"
    save_parquet(merged, out_path)

    summary = (
        merged.groupby(["cluster_id", "label"]).size().reset_index(name="count")
        .sort_values(["cluster_id", "count"], ascending=[True, False])
    )
    summary.to_csv(Path(cfg["paths"]["processed_dir"]) / f"clusters_{args.embedding_source}_label_distribution.csv", index=False)

    metrics = {
        "embedding_source": args.embedding_source,
        "n_rows": int(merged.shape[0]),
        "n_features": int(len(feature_cols)),
        "n_clusters": int(n_clusters),
        "silhouette": sil,
        "calinski_harabasz": ch,
        "davies_bouldin": db,
    }
    metrics_path = Path(cfg["paths"]["processed_dir"]) / f"clusters_{args.embedding_source}_quality_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Build a 2D projection for visualization.
    if x_used.shape[1] > 2:
        proj = TruncatedSVD(n_components=2, random_state=42).fit_transform(x_used)
    elif x_used.shape[1] == 2:
        proj = x_used
    else:
        proj = np.hstack([x_used, np.zeros((x_used.shape[0], 1), dtype=np.float32)])

    proj_df = merged[["id", "label", "is_low_truth", "split", "cluster_id"]].copy()
    proj_df["dim1"] = proj[:, 0]
    proj_df["dim2"] = proj[:, 1]
    proj_path = Path(cfg["paths"]["processed_dir"]) / f"clusters_{args.embedding_source}_projection.parquet"
    save_parquet(proj_df, proj_path)

    fig_dir = Path(cfg["paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=proj_df.sample(min(5000, len(proj_df)), random_state=42),
        x="dim1",
        y="dim2",
        hue="cluster_id",
        palette="tab10",
        s=12,
        linewidth=0,
    )
    plt.title(f"Embedding Projection by Cluster ({args.embedding_source})")
    plt.tight_layout()
    by_cluster_path = fig_dir / f"clusters_{args.embedding_source}_by_cluster.png"
    plt.savefig(by_cluster_path, dpi=180)
    plt.close()

    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=proj_df.sample(min(5000, len(proj_df)), random_state=42),
        x="dim1",
        y="dim2",
        hue="is_low_truth",
        palette={0: "#1f77b4", 1: "#d62728"},
        s=12,
        linewidth=0,
    )
    plt.title(f"Embedding Projection by Low-Truth Flag ({args.embedding_source})")
    plt.tight_layout()
    by_truth_path = fig_dir / f"clusters_{args.embedding_source}_by_truth.png"
    plt.savefig(by_truth_path, dpi=180)
    plt.close()

    print(f"Saved clustering output: {out_path}")
    print(f"Feature source={args.embedding_source}, rows={merged.shape[0]}, cols={len(feature_cols)}")
    print(f"Saved cluster metrics: {metrics_path}")
    print(f"Saved projection: {proj_path}")
    print(f"Saved figures: {by_cluster_path} and {by_truth_path}")
    print(f"Silhouette={sil}, Calinski-Harabasz={ch}, Davies-Bouldin={db}")


if __name__ == "__main__":
    main()

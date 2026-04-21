#!/usr/bin/env python
"""Run anomaly detection on fused LIAR text+graph representations.

Combines text and graph embeddings, applies optional dimensionality reduction,
fits Isolation Forest, and reports anomalous claims and speaker-level anomaly
rates for qualitative analysis.

Inputs:
- `data/processed/liar_base.parquet`
- `data/interim/text_embeddings_*.parquet`
- `data/interim/graph_embeddings.parquet`

Outputs:
- `data/processed/anomaly_claims.parquet`
- `data/processed/anomaly_speakers.csv`
- `data/processed/anomaly_speakers_min_claims.csv`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import TruncatedSVD

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liar_mining.config import ensure_dirs, load_config
from liar_mining.io import load_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect anomalous claims/speakers in LIAR feature space.")
    parser.add_argument("--config", default="configs/liar_research.yaml")
    parser.add_argument("--text-file", default="text_embeddings_tfidf.parquet")
    parser.add_argument("--svd-dim", type=int, default=100)
    parser.add_argument("--min-speaker-claims", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    base = load_parquet(Path(cfg["paths"]["processed_dir"]) / "liar_base.parquet")
    text = load_parquet(Path(cfg["paths"]["interim_dir"]) / args.text_file)
    graph = load_parquet(Path(cfg["paths"]["interim_dir"]) / "graph_embeddings.parquet")

    fused = base[["id", "label", "is_low_truth", "speaker", "subject", "statement", "split"]].merge(
        text, on="id", how="inner"
    ).merge(graph, on="id", how="inner")

    feature_cols = [c for c in fused.columns if c.startswith("text_emb_") or c.startswith("graph_emb_")]
    x = fused[feature_cols].to_numpy(dtype=np.float32)

    if x.shape[1] > args.svd_dim:
        x = TruncatedSVD(n_components=args.svd_dim, random_state=42).fit_transform(x)

    iso = IsolationForest(
        contamination=cfg["anomaly"]["contamination"],
        random_state=cfg["anomaly"]["random_state"],
    )
    pred = iso.fit_predict(x)
    score = iso.decision_function(x)

    fused["anomaly_flag"] = (pred == -1).astype(int)
    fused["anomaly_score"] = score

    out_claims = Path(cfg["paths"]["processed_dir"]) / "anomaly_claims.parquet"
    fused.to_parquet(out_claims, index=False)

    speaker_summary = (
        fused.groupby("speaker")
        .agg(
            total_claims=("id", "count"),
            anomaly_rate=("anomaly_flag", "mean"),
            low_truth_rate=("is_low_truth", "mean"),
        )
        .reset_index()
        .sort_values(["anomaly_rate", "total_claims"], ascending=[False, False])
    )
    out_speakers = Path(cfg["paths"]["processed_dir"]) / "anomaly_speakers.csv"
    speaker_summary.to_csv(out_speakers, index=False)

    filtered_speaker_summary = (
        speaker_summary[speaker_summary["total_claims"] >= args.min_speaker_claims]
        .sort_values(["anomaly_rate", "total_claims"], ascending=[False, False])
    )
    out_speakers_filtered = Path(cfg["paths"]["processed_dir"]) / "anomaly_speakers_min_claims.csv"
    filtered_speaker_summary.to_csv(out_speakers_filtered, index=False)

    print(f"Saved claim-level anomalies: {out_claims}")
    print(f"Saved speaker-level anomaly summary: {out_speakers}")
    print(f"Saved filtered speaker anomaly summary: {out_speakers_filtered}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Estimate distribution drift across LIAR data stream slices.

Treats train/valid/test as ordered periods, computes Jensen-Shannon divergence
for categorical distributions (labels/party/state), and exports split-level
trend summaries including low-truth rate and top subjects.

Input:
- `data/processed/liar_base.parquet`

Outputs:
- `data/processed/stream_drift_summary.csv`
- `data/processed/stream_top_subjects.csv`
- `data/processed/stream_low_truth_rate.csv`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liar_mining.config import ensure_dirs, load_config
from liar_mining.io import load_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate temporal/data-stream drift across LIAR splits.")
    parser.add_argument("--config", default="configs/liar_research.yaml")
    return parser.parse_args()


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def cat_dist(df: pd.DataFrame, col: str) -> pd.Series:
    return df[col].value_counts(normalize=True)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    df = load_parquet(Path(cfg["paths"]["processed_dir"]) / "liar_base.parquet")

    splits = ["train", "valid", "test"]
    split_frames = {s: df[df["split"] == s].copy() for s in splits}

    rows = []
    for a, b in [("train", "valid"), ("valid", "test"), ("train", "test")]:
        for col in ["label", "party_grouped", "state_clean"]:
            da = cat_dist(split_frames[a], col)
            db = cat_dist(split_frames[b], col)
            idx = sorted(set(da.index).union(db.index))
            pa = da.reindex(idx, fill_value=0.0).values
            pb = db.reindex(idx, fill_value=0.0).values
            rows.append({"split_a": a, "split_b": b, "feature": col, "js_divergence": js_divergence(pa, pb)})

    drift_df = pd.DataFrame(rows)
    drift_out = Path(cfg["paths"]["processed_dir"]) / "stream_drift_summary.csv"
    drift_df.to_csv(drift_out, index=False)

    subj = df[["split", "subject"]].copy()
    subj["subject"] = subj["subject"].fillna("").str.split(",")
    subj = subj.explode("subject")
    subj["subject"] = subj["subject"].astype(str).str.strip().str.lower()
    subj = subj[subj["subject"] != ""]

    top_subj = (
        subj.groupby(["split", "subject"]).size().reset_index(name="count")
        .sort_values(["split", "count"], ascending=[True, False])
    )
    top_subj_out = Path(cfg["paths"]["processed_dir"]) / "stream_top_subjects.csv"
    top_subj.to_csv(top_subj_out, index=False)

    low_truth_by_split = (
        df.groupby("split")["is_low_truth"].mean().reset_index(name="low_truth_rate")
    )
    low_truth_out = Path(cfg["paths"]["processed_dir"]) / "stream_low_truth_rate.csv"
    low_truth_by_split.to_csv(low_truth_out, index=False)

    print(f"Saved drift summary: {drift_out}")
    print(f"Saved top subjects by split: {top_subj_out}")
    print(f"Saved low-truth split trend: {low_truth_out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Detect near-duplicate LIAR statements using MinHash LSH.

Tokenizes statements, builds MinHash signatures, queries LSH candidates, and
exports approximate-Jaccard duplicate pairs to help assess leakage/memorization.

Input:
- `data/processed/liar_base.parquet`

Output:
- `data/processed/near_duplicate_pairs.csv`
- `data/processed/near_duplicate_pairs_filtered.csv`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import pandas as pd
from datasketch import MinHash, MinHashLSH

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liar_mining.config import ensure_dirs, load_config
from liar_mining.io import load_parquet

TOKEN_RE = re.compile(r"[a-z0-9]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find near-duplicate statements using MinHash LSH.")
    parser.add_argument("--config", default="configs/liar_research.yaml")
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def to_minhash(tokens: list[str], num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for t in tokens:
        m.update(t.encode("utf-8"))
    return m


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    base = load_parquet(Path(cfg["paths"]["processed_dir"]) / "liar_base.parquet")

    num_perm = cfg["lsh"]["num_perm"]
    threshold = cfg["lsh"]["threshold"]

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}

    for _, row in base.iterrows():
        statement = str(row["statement"])
        tokens = tokenize(statement)
        if not tokens:
            continue
        m = to_minhash(tokens, num_perm)
        key = str(row["id"])
        minhashes[key] = m
        lsh.insert(key, m)

    pairs = []
    seen = set()
    for key, m in minhashes.items():
        candidates = lsh.query(m)
        for cand in candidates:
            if cand == key:
                continue
            a, b = sorted([key, cand])
            if (a, b) in seen:
                continue
            seen.add((a, b))
            sim = minhashes[a].jaccard(minhashes[b])
            pairs.append((a, b, sim))

    dup_df = pd.DataFrame(pairs, columns=["id_a", "id_b", "approx_jaccard"])
    out_path = Path(cfg["paths"]["processed_dir"]) / "near_duplicate_pairs.csv"
    dup_df = dup_df.sort_values("approx_jaccard", ascending=False)
    dup_df.to_csv(out_path, index=False)

    filtered_df = dup_df[dup_df["approx_jaccard"] >= threshold].copy()
    filtered_path = Path(cfg["paths"]["processed_dir"]) / "near_duplicate_pairs_filtered.csv"
    filtered_df.to_csv(filtered_path, index=False)

    print(f"Saved near-duplicate pairs: {out_path}")
    print(f"Saved threshold-filtered pairs: {filtered_path}")
    print(f"Pairs found (raw)={dup_df.shape[0]}, (filtered)={filtered_df.shape[0]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Prepare the base LIAR analysis table used by downstream scripts.

This script loads LIAR train/valid/test TSV files, applies lightweight cleaning
and feature engineering (party/state normalization, length features, low-truth
label), and saves a unified parquet file for later stages.

Input:
- Raw LIAR splits configured in `configs/liar_research.yaml`

Output:
- `data/processed/liar_base.parquet`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liar_mining.config import ensure_dirs, load_config
from liar_mining.io import load_liar_splits, save_parquet
from liar_mining.preprocessing import add_engineered_columns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cleaned LIAR base table for downstream analysis.")
    parser.add_argument(
        "--config",
        default="configs/liar_research.yaml",
        help="Path to YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    raw_df = load_liar_splits(cfg)
    base_df = add_engineered_columns(raw_df, cfg["labels"])

    out_path = Path(cfg["paths"]["processed_dir"]) / "liar_base.parquet"
    save_parquet(base_df, out_path)

    print(f"Saved base LIAR table: {out_path}")
    print(f"Rows={base_df.shape[0]}, Cols={base_df.shape[1]}")


if __name__ == "__main__":
    main()

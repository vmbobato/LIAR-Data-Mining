#!/usr/bin/env python
"""Mine frequent patterns and association rules from LIAR metadata.

Builds transactions from party/state/subject/label tokens and runs Apriori plus
association rules for:
- all claims, and
- low-truth claims only.

Input:
- `data/processed/liar_base.parquet`

Outputs:
- `data/processed/itemsets_all.csv`
- `data/processed/rules_all.csv`
- `data/processed/itemsets_low_truth.csv`
- `data/processed/rules_low_truth.csv`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liar_mining.config import ensure_dirs, load_config
from liar_mining.io import load_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine frequent itemsets and rules on LIAR metadata.")
    parser.add_argument("--config", default="configs/liar_research.yaml")
    parser.add_argument("--min-support", type=float, default=0.01)
    parser.add_argument("--min-confidence", type=float, default=0.2)
    return parser.parse_args()


def make_transactions(df: pd.DataFrame) -> list[list[str]]:
    tx = []
    for _, row in df.iterrows():
        items = []
        items.append(f"party={row['party_grouped']}")
        items.append(f"state={row['state_clean']}")
        subjects = [s.strip().lower() for s in str(row["subject"]).split(",") if s.strip()]
        for s in subjects:
            items.append(f"subject={s}")
        items.append(f"label={row['label']}")
        tx.append(items)
    return tx


def run_itemset_mining(df: pd.DataFrame, min_support: float, min_confidence: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    tx = make_transactions(df)
    te = TransactionEncoder()
    onehot = te.fit(tx).transform(tx)
    onehot_df = pd.DataFrame(onehot, columns=te.columns_)

    itemsets = apriori(onehot_df, min_support=min_support, use_colnames=True)
    itemsets["itemset_size"] = itemsets["itemsets"].apply(len)

    if itemsets.empty:
        return itemsets, pd.DataFrame()

    rules = association_rules(itemsets, metric="confidence", min_threshold=min_confidence)
    if not rules.empty:
        rules = rules.sort_values(["lift", "confidence"], ascending=False)
    return itemsets, rules


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    base_path = Path(cfg["paths"]["processed_dir"]) / "liar_base.parquet"
    df = load_parquet(base_path)

    itemsets_all, rules_all = run_itemset_mining(df, args.min_support, args.min_confidence)
    low_df = df[df["is_low_truth"] == 1].copy()
    itemsets_low, rules_low = run_itemset_mining(low_df, args.min_support, args.min_confidence)

    out_dir = Path(cfg["paths"]["processed_dir"])

    itemsets_all.to_csv(out_dir / "itemsets_all.csv", index=False)
    rules_all.to_csv(out_dir / "rules_all.csv", index=False)
    itemsets_low.to_csv(out_dir / "itemsets_low_truth.csv", index=False)
    rules_low.to_csv(out_dir / "rules_low_truth.csv", index=False)

    print("Saved itemset and rule tables to data/processed")
    print(f"All itemsets={itemsets_all.shape[0]}, low-truth itemsets={itemsets_low.shape[0]}")


if __name__ == "__main__":
    main()

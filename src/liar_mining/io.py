from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd

LIAR_COLUMNS: List[str] = [
    "id",
    "label",
    "statement",
    "subject",
    "speaker",
    "speaker_job",
    "state",
    "party",
    "barely_true_counts",
    "false_counts",
    "half_true_counts",
    "mostly_true_counts",
    "pants_on_fire_counts",
    "context",
]


def read_liar_tsv(path: str | Path, split_name: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=LIAR_COLUMNS)
    df["split"] = split_name
    return df


def load_liar_splits(cfg: Dict) -> pd.DataFrame:
    train = read_liar_tsv(cfg["raw_data"]["train"], "train")
    valid = read_liar_tsv(cfg["raw_data"]["valid"], "valid")
    test = read_liar_tsv(cfg["raw_data"]["test"], "test")
    return pd.concat([train, valid, test], ignore_index=True)


def save_parquet(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)

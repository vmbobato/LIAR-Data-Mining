#!/usr/bin/env python
"""Fine-tune a transformer classifier for low-truth LIAR detection.

This script trains a text classifier (BERT/RoBERTa/etc.) on LIAR statements and
reports performance under:
- fixed split (train -> test)
- speaker-disjoint split (GroupShuffleSplit)

Outputs:
- `data/processed/transformer_finetune_metrics_<tag>.json`
- `data/processed/transformer_finetune_predictions_fixed_split_<tag>.csv`
- `data/processed/transformer_finetune_predictions_speaker_disjoint_<tag>.csv`
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liar_mining.config import ensure_dirs, load_config
from liar_mining.io import load_parquet
from liar_mining.modeling import compute_binary_metrics, dump_json


@dataclass
class TextDataset(torch.utils.data.Dataset):
    encodings: dict
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune transformer classifier for LIAR low-truth detection.")
    parser.add_argument("--config", default="configs/liar_research.yaml")
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--warmup-fraction", type=float, default=0.1)
    return parser.parse_args()


def sanitize_tag(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def tokenize_texts(tokenizer, texts: list[str], max_length: int) -> dict:
    return tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
    )


def train_and_predict(
    model_name: str,
    x_train: list[str],
    y_train: np.ndarray,
    x_test: list[str],
    y_test: np.ndarray,
    out_dir: Path,
    run_name: str,
    args: argparse.Namespace,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    enc_train = tokenize_texts(tokenizer, x_train, args.max_length)
    enc_test = tokenize_texts(tokenizer, x_test, args.max_length)

    ds_train = TextDataset(encodings=enc_train, labels=y_train)
    ds_test = TextDataset(encodings=enc_test, labels=y_test)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # Avoid deprecated warmup_ratio by deriving warmup_steps from dataset size.
    if args.warmup_steps is not None:
        warmup_steps = max(0, int(args.warmup_steps))
    else:
        updates_per_epoch = math.ceil(len(ds_train) / max(1, args.batch_size))
        total_updates = max(1, int(updates_per_epoch * args.epochs))
        warmup_steps = max(0, int(total_updates * args.warmup_fraction))

    train_args = TrainingArguments(
        output_dir=str(out_dir / f"tmp_{run_name}"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=50,
        save_strategy="no",
        eval_strategy="no",
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
    )

    trainer.train()
    pred = trainer.predict(ds_test)
    logits = pred.predictions

    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    y_hat = (probs >= 0.5).astype(int)
    metrics = compute_binary_metrics(y_test, y_hat, probs)
    return metrics, probs, y_hat


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    model_tag = sanitize_tag(args.model_name)

    df = load_parquet(processed_dir / "liar_base.parquet")
    df["statement"] = df["statement"].fillna("").astype(str)

    # Fixed split
    tr = df[df["split"] == "train"].copy()
    te = df[df["split"] == "test"].copy()

    fixed_metrics, fixed_prob, fixed_pred = train_and_predict(
        model_name=args.model_name,
        x_train=tr["statement"].tolist(),
        y_train=tr["is_low_truth"].to_numpy(),
        x_test=te["statement"].tolist(),
        y_test=te["is_low_truth"].to_numpy(),
        out_dir=processed_dir,
        run_name=f"fixed_{model_tag}",
        args=args,
    )

    fixed_out = te[["id", "speaker", "label", "is_low_truth"]].copy()
    fixed_out["prob"] = fixed_prob
    fixed_out["pred"] = fixed_pred
    fixed_out.to_csv(processed_dir / f"transformer_finetune_predictions_fixed_split_{model_tag}.csv", index=False)

    # Speaker-disjoint split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    idx_train, idx_test = next(gss.split(df, df["is_low_truth"], groups=df["speaker"]))
    dtr = df.iloc[idx_train].copy()
    dte = df.iloc[idx_test].copy()

    group_metrics, group_prob, group_pred = train_and_predict(
        model_name=args.model_name,
        x_train=dtr["statement"].tolist(),
        y_train=dtr["is_low_truth"].to_numpy(),
        x_test=dte["statement"].tolist(),
        y_test=dte["is_low_truth"].to_numpy(),
        out_dir=processed_dir,
        run_name=f"speaker_{model_tag}",
        args=args,
    )

    group_out = dte[["id", "speaker", "label", "is_low_truth"]].copy()
    group_out["prob"] = group_prob
    group_out["pred"] = group_pred
    group_out.to_csv(processed_dir / f"transformer_finetune_predictions_speaker_disjoint_{model_tag}.csv", index=False)

    metrics = {
        "fixed_split_train_test": fixed_metrics,
        "speaker_disjoint_split": group_metrics,
        "notes": {
            "model_name": args.model_name,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
        },
    }
    dump_json(metrics, processed_dir / f"transformer_finetune_metrics_{model_tag}.json")

    print(f"Saved transformer metrics for {args.model_name}")


if __name__ == "__main__":
    main()

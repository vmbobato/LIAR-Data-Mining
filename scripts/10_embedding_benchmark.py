#!/usr/bin/env python
"""Benchmark embedding backbones and classifiers for LIAR low-truth detection.

Evaluates combinations of:
- Embeddings: TF-IDF, MiniLM, BERT-base
- Classifiers: logistic_regression, linear_svm, random_forest, gradient_boosting

For each pair it runs `09_fusion_model.py` and aggregates metrics into one table.

Outputs:
- `data/processed/embedding_classifier_model_comparison.csv`
- `data/processed/fusion_model_metrics_<embedding>_<classifier>.json`
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd


EMBEDDING_SPECS = [
    {
        "name": "tfidf",
        "text_file": "text_embeddings_tfidf.parquet",
        "gen_cmd": ["scripts/02_text_embeddings.py", "--method", "tfidf"],
    },
    {
        "name": "minilm",
        "text_file": "text_embeddings_sentence_all-MiniLM-L6-v2.parquet",
        "gen_cmd": [
            "scripts/02_text_embeddings.py",
            "--method",
            "sentence",
            "--model",
            "all-MiniLM-L6-v2",
        ],
    },
    {
        "name": "bert_base",
        "text_file": "text_embeddings_sentence_bert-base-uncased.parquet",
        "gen_cmd": [
            "scripts/02_text_embeddings.py",
            "--method",
            "sentence",
            "--model",
            "bert-base-uncased",
        ],
    },
]

CLASSIFIERS = ["logistic_regression", "linear_svm", "random_forest", "gradient_boosting"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding + classifier benchmark.")
    parser.add_argument("--config", default="configs/liar_research.yaml")
    parser.add_argument("--skip-generate", action="store_true", help="Skip embedding generation; use existing files")
    parser.add_argument("--max-iter", type=int, default=300)
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> None:
    full_cmd = [sys.executable] + cmd
    print("$", " ".join(full_cmd))
    proc = subprocess.run(full_cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(full_cmd)}")


def collect_metrics(metrics_path: Path, embedding_name: str, classifier_name: str) -> list[dict]:
    payload = json.loads(metrics_path.read_text())
    rows = []
    for setting in ["fixed_split_train_test", "speaker_disjoint_split"]:
        for variant in ["text_only", "text_plus_graph"]:
            row = {
                "embedding": embedding_name,
                "classifier": classifier_name,
                "setting": setting,
                "variant": variant,
            }
            row.update(payload[setting][variant])
            rows.append(row)
    return rows


def main() -> None:
    args = parse_args()

    processed_dir = Path("data/processed")
    interim_dir = Path("data/interim")
    processed_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    for emb in EMBEDDING_SPECS:
        print("\n=== Embedding:", emb["name"], "===")
        text_path = interim_dir / emb["text_file"]
        if not args.skip_generate or not text_path.exists():
            run_cmd(emb["gen_cmd"])

        for clf in CLASSIFIERS:
            print("--- Classifier:", clf, "---")
            suffix = f"_{emb['name']}_{clf}"
            run_cmd(
                [
                    "scripts/09_fusion_model.py",
                    "--config",
                    args.config,
                    "--text-file",
                    emb["text_file"],
                    "--model",
                    clf,
                    "--max-iter",
                    str(args.max_iter),
                    "--output-suffix",
                    suffix,
                ]
            )

            metrics_path = processed_dir / with_suffix("fusion_model_metrics.json", suffix)
            all_rows.extend(collect_metrics(metrics_path, emb["name"], clf))

    comp_df = pd.DataFrame(all_rows)
    out_csv = processed_dir / "embedding_classifier_model_comparison.csv"
    comp_df.to_csv(out_csv, index=False)

    print(f"\nSaved benchmark comparison table: {out_csv}")
    view = comp_df.sort_values(["setting", "variant", "f1"], ascending=[True, True, False])
    print(view.to_string(index=False))


def with_suffix(name: str, suffix: str) -> str:
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        return f"{stem}{suffix}.{ext}"
    return f"{name}{suffix}"


if __name__ == "__main__":
    main()

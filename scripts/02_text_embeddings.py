#!/usr/bin/env python
"""Generate statement-level text embeddings for LIAR claims.

Supports either:
- TF-IDF sparse text features (saved as dense parquet table for simplicity), or
- SentenceTransformer dense embeddings.

Input:
- `data/processed/liar_base.parquet`

Outputs:
- `data/interim/text_embeddings_tfidf.parquet` or
  `data/interim/text_embeddings_sentence_<model>.parquet`
- `models/tfidf_vectorizer.joblib` (TF-IDF mode)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import joblib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liar_mining.config import ensure_dirs, load_config
from liar_mining.io import load_parquet, save_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text embeddings for LIAR statements.")
    parser.add_argument("--config", default="configs/liar_research.yaml")
    parser.add_argument("--method", choices=["tfidf", "sentence"], default="tfidf")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformer model name")
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--output-file", default=None, help="Optional custom output parquet filename")
    return parser.parse_args()


def run_tfidf(df: pd.DataFrame, cfg: dict, max_features_override: int | None) -> tuple[pd.DataFrame, object]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    max_features = max_features_override or cfg["text"]["max_features"]
    vectorizer = TfidfVectorizer(
        min_df=cfg["text"]["min_df"],
        max_features=max_features,
        ngram_range=(cfg["text"]["ngram_min"], cfg["text"]["ngram_max"]),
        lowercase=True,
    )
    x = vectorizer.fit_transform(df["statement"].fillna(""))
    arr = x.astype(np.float32).toarray()
    cols = [f"text_emb_{i}" for i in range(arr.shape[1])]
    emb_df = pd.DataFrame(arr, columns=cols)
    emb_df.insert(0, "id", df["id"].values)
    return emb_df, vectorizer


def run_sentence_transformer(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    vectors = model.encode(df["statement"].fillna("").tolist(), show_progress_bar=True)
    arr = np.asarray(vectors, dtype=np.float32)
    cols = [f"text_emb_{i}" for i in range(arr.shape[1])]
    emb_df = pd.DataFrame(arr, columns=cols)
    emb_df.insert(0, "id", df["id"].values)
    return emb_df


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_name).strip("_")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    base_path = Path(cfg["paths"]["processed_dir"]) / "liar_base.parquet"
    base_df = load_parquet(base_path)

    if args.method == "tfidf":
        emb_df, vectorizer = run_tfidf(base_df, cfg, args.max_features)
        model_path = Path(cfg["paths"]["models_dir"]) / "tfidf_vectorizer.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, model_path)
        out_name = args.output_file or "text_embeddings_tfidf.parquet"
        print(f"Saved TF-IDF vectorizer: {model_path}")
    else:
        emb_df = run_sentence_transformer(base_df, args.model)
        model_tag = sanitize_model_name(args.model)
        out_name = args.output_file or f"text_embeddings_sentence_{model_tag}.parquet"

    out_path = Path(cfg["paths"]["interim_dir"]) / out_name
    save_parquet(emb_df, out_path)
    print(f"Saved text embeddings: {out_path}")
    print(f"Embedding shape: {emb_df.shape}")


if __name__ == "__main__":
    main()

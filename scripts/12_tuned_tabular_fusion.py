#!/usr/bin/env python
"""Train a tuned tabular model on fused text+graph+metadata features.

Uses HistGradientBoosting with RandomizedSearchCV over fused features to test
whether tuned tabular learning improves low-truth detection.

Outputs:
- `data/processed/tuned_tabular_fusion_metrics_<tag>.json`
- `data/processed/tuned_tabular_fusion_predictions_fixed_split_<tag>.csv`
- `data/processed/tuned_tabular_fusion_predictions_speaker_disjoint_<tag>.csv`
- `data/processed/tuned_tabular_fusion_best_params_<tag>.json`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liar_mining.config import ensure_dirs, load_config
from liar_mining.io import load_parquet
from liar_mining.modeling import compute_binary_metrics, dump_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune tabular fusion model for LIAR low-truth detection.")
    parser.add_argument("--config", default="configs/liar_research.yaml")
    parser.add_argument("--text-file", default="text_embeddings_sentence_bert-base-uncased.parquet")
    parser.add_argument("--text-svd-dim", type=int, default=256)
    parser.add_argument("--n-iter", type=int, default=18)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sanitize_tag(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "total_prior_statements",
        "statement_length_words",
        "statement_length_chars",
    ]
    cat_cols = ["party_grouped", "state_clean"]

    out = df.copy()
    for c in num_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    for c in cat_cols:
        out[c] = out[c].fillna("unknown").astype(str)

    cat_df = pd.get_dummies(out[cat_cols], drop_first=False)
    out = pd.concat([out, cat_df], axis=1)
    return out


def build_matrix(
    full_df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    text_cols: list[str],
    graph_cols: list[str],
    meta_cols: list[str],
    text_svd_dim: int,
):
    dtr = full_df.iloc[train_idx].copy()
    dte = full_df.iloc[test_idx].copy()

    xtr_text = dtr[text_cols].to_numpy(dtype=np.float32)
    xte_text = dte[text_cols].to_numpy(dtype=np.float32)

    if xtr_text.shape[1] > text_svd_dim:
        svd = TruncatedSVD(n_components=text_svd_dim, random_state=42)
        xtr_text = svd.fit_transform(xtr_text)
        xte_text = svd.transform(xte_text)

    xtr = np.hstack([
        xtr_text,
        dtr[graph_cols].to_numpy(dtype=np.float32),
        dtr[meta_cols].to_numpy(dtype=np.float32),
    ])
    xte = np.hstack([
        xte_text,
        dte[graph_cols].to_numpy(dtype=np.float32),
        dte[meta_cols].to_numpy(dtype=np.float32),
    ])

    ytr = dtr["is_low_truth"].to_numpy()
    yte = dte["is_low_truth"].to_numpy()
    return dtr, dte, xtr, xte, ytr, yte


def tune_and_eval(xtr, ytr, xte, yte, n_iter: int, cv: int, seed: int):
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                HistGradientBoostingClassifier(random_state=seed),
            ),
        ]
    )

    param_dist = {
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "clf__max_depth": [4, 6, 8, 10, None],
        "clf__max_iter": [200, 300, 400, 500],
        "clf__min_samples_leaf": [20, 40, 60, 100],
        "clf__l2_regularization": [0.0, 0.001, 0.01, 0.1],
    }

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="f1",
        cv=cv,
        random_state=seed,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(xtr, ytr)

    best = search.best_estimator_
    prob = best.predict_proba(xte)[:, 1]
    pred = (prob >= 0.5).astype(int)
    metrics = compute_binary_metrics(yte, pred, prob)
    return metrics, pred, prob, search.best_params_


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    processed = Path(cfg["paths"]["processed_dir"])
    interim = Path(cfg["paths"]["interim_dir"])
    tag = sanitize_tag(Path(args.text_file).stem)

    base = load_parquet(processed / "liar_base.parquet")
    text = load_parquet(interim / args.text_file)
    graph = load_parquet(interim / "graph_embeddings.parquet")

    df = base.merge(text, on="id", how="inner").merge(graph, on="id", how="inner")
    df = prepare_features(df)

    text_cols = [c for c in df.columns if c.startswith("text_emb_")]
    graph_cols = [c for c in df.columns if c.startswith("graph_emb_")]
    meta_cols = [c for c in df.columns if c.startswith("party_grouped_") or c.startswith("state_clean_")] + [
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "total_prior_statements",
        "statement_length_words",
        "statement_length_chars",
    ]

    fixed_tr_idx = np.where(df["split"].values == "train")[0]
    fixed_te_idx = np.where(df["split"].values == "test")[0]

    dtr, dte, xtr, xte, ytr, yte = build_matrix(
        df, fixed_tr_idx, fixed_te_idx, text_cols, graph_cols, meta_cols, args.text_svd_dim
    )
    fixed_metrics, fixed_pred, fixed_prob, fixed_best = tune_and_eval(xtr, ytr, xte, yte, args.n_iter, args.cv, args.seed)

    fixed_out = dte[["id", "speaker", "label", "is_low_truth"]].copy()
    fixed_out["pred"] = fixed_pred
    fixed_out["prob"] = fixed_prob
    fixed_out.to_csv(processed / f"tuned_tabular_fusion_predictions_fixed_split_{tag}.csv", index=False)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.seed)
    idx_train, idx_test = next(gss.split(df, df["is_low_truth"], groups=df["speaker"]))

    gdtr, gdte, gxtr, gxte, gytr, gyte = build_matrix(
        df, idx_train, idx_test, text_cols, graph_cols, meta_cols, args.text_svd_dim
    )
    group_metrics, group_pred, group_prob, group_best = tune_and_eval(gxtr, gytr, gxte, gyte, args.n_iter, args.cv, args.seed)

    group_out = gdte[["id", "speaker", "label", "is_low_truth"]].copy()
    group_out["pred"] = group_pred
    group_out["prob"] = group_prob
    group_out.to_csv(processed / f"tuned_tabular_fusion_predictions_speaker_disjoint_{tag}.csv", index=False)

    metrics = {
        "fixed_split_train_test": fixed_metrics,
        "speaker_disjoint_split": group_metrics,
        "notes": {
            "text_file": args.text_file,
            "text_svd_dim": args.text_svd_dim,
            "n_iter": args.n_iter,
            "cv": args.cv,
        },
    }
    dump_json(metrics, processed / f"tuned_tabular_fusion_metrics_{tag}.json")
    dump_json(
        {
            "fixed_split_best_params": fixed_best,
            "speaker_disjoint_best_params": group_best,
        },
        processed / f"tuned_tabular_fusion_best_params_{tag}.json",
    )

    print("Saved tuned tabular fusion outputs.")


if __name__ == "__main__":
    main()

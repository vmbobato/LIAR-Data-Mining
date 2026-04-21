#!/usr/bin/env python
"""Build a probability ensemble with threshold optimization.

Combines probabilities from multiple model outputs and optimizes:
- model weights (random search on simplex)
- classification threshold (grid)

This is intended to improve low-truth detection while preserving speaker-
disjoint robustness.

Outputs:
- `data/processed/ensemble_threshold_metrics.json`
- `data/processed/ensemble_threshold_weights.csv`
- `data/processed/ensemble_threshold_predictions.csv`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liar_mining.modeling import compute_binary_metrics, dump_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probability ensemble with threshold optimization.")
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--num-models", type=int, default=6, help="Top N candidate models by F1 to include")
    parser.add_argument("--weight-search-iters", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def pick_score_column(df: pd.DataFrame) -> str:
    for c in ["text_plus_graph_score", "prob", "score", "text_plus_graph_prob"]:
        if c in df.columns:
            return c
    raise ValueError("No probability/score column found.")


def load_candidate_predictions(processed_dir: Path, pattern: str) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for p in sorted(processed_dir.glob(pattern)):
        name = p.stem
        df = pd.read_csv(p)
        score_col = pick_score_column(df)
        keep = ["id", "speaker", "label", "is_low_truth", score_col]
        sub = df[keep].copy().rename(columns={score_col: f"score__{name}"})
        out.append((name, sub))
    return out


def merge_scores(frames: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No candidate prediction files found.")
    merged = frames[0][1]
    for _, df in frames[1:]:
        merged = merged.merge(df[["id"] + [c for c in df.columns if c.startswith("score__")]], on="id", how="inner")
    return merged


def optimize_weights_and_threshold(x_tune: np.ndarray, y_tune: np.ndarray, iters: int, seed: int):
    rng = np.random.default_rng(seed)
    n_models = x_tune.shape[1]

    best = {"f1": -1.0, "weights": None, "threshold": 0.5}
    thresholds = np.linspace(0.2, 0.8, 25)

    for _ in range(iters):
        raw = rng.random(n_models)
        weights = raw / raw.sum()
        prob = x_tune @ weights
        for th in thresholds:
            pred = (prob >= th).astype(int)
            m = compute_binary_metrics(y_tune, pred, prob)
            if m["f1"] > best["f1"]:
                best = {"f1": m["f1"], "weights": weights, "threshold": float(th)}
    return best


def evaluate_setting(setting_name: str, merged: pd.DataFrame, num_models: int, iters: int, seed: int):
    score_cols = [c for c in merged.columns if c.startswith("score__")]

    # rank candidate single models by F1 at threshold 0.5
    rows = []
    y = merged["is_low_truth"].to_numpy()
    for c in score_cols:
        prob = merged[c].to_numpy()
        pred = (prob >= 0.5).astype(int)
        m = compute_binary_metrics(y, pred, prob)
        rows.append((c, m["f1"]))
    rows.sort(key=lambda x: x[1], reverse=True)
    selected = [c for c, _ in rows[: min(num_models, len(rows))]]

    mdf = merged[["id", "speaker", "label", "is_low_truth"] + selected].copy()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    idx_tune, idx_eval = next(gss.split(mdf, mdf["is_low_truth"], groups=mdf["speaker"]))
    tune_df = mdf.iloc[idx_tune].copy()
    eval_df = mdf.iloc[idx_eval].copy()

    x_tune = tune_df[selected].to_numpy(dtype=np.float64)
    y_tune = tune_df["is_low_truth"].to_numpy()
    x_eval = eval_df[selected].to_numpy(dtype=np.float64)
    y_eval = eval_df["is_low_truth"].to_numpy()

    best = optimize_weights_and_threshold(x_tune, y_tune, iters=iters, seed=seed)
    p_eval = x_eval @ best["weights"]
    y_hat_eval = (p_eval >= best["threshold"]).astype(int)
    eval_metrics = compute_binary_metrics(y_eval, y_hat_eval, p_eval)

    pred_out = eval_df[["id", "speaker", "label", "is_low_truth"]].copy()
    pred_out["setting"] = setting_name
    pred_out["ensemble_prob"] = p_eval
    pred_out["ensemble_pred"] = y_hat_eval

    weight_rows = []
    for c, w in zip(selected, best["weights"]):
        weight_rows.append(
            {
                "setting": setting_name,
                "model_score_col": c,
                "weight": float(w),
            }
        )

    return {
        "setting": setting_name,
        "selected_models": selected,
        "threshold": float(best["threshold"]),
        "tune_f1": float(best["f1"]),
        "eval_metrics": eval_metrics,
    }, pd.DataFrame(weight_rows), pred_out


def main() -> None:
    args = parse_args()
    processed_dir = Path(args.processed_dir)

    fixed_frames = load_candidate_predictions(processed_dir, "fusion_model_predictions_fixed_split_*.csv")
    fixed_frames += load_candidate_predictions(processed_dir, "transformer_finetune_predictions_fixed_split_*.csv")
    fixed_frames += load_candidate_predictions(processed_dir, "tuned_tabular_fusion_predictions_fixed_split_*.csv")

    group_frames = load_candidate_predictions(processed_dir, "fusion_model_predictions_speaker_disjoint_*.csv")
    group_frames += load_candidate_predictions(processed_dir, "transformer_finetune_predictions_speaker_disjoint_*.csv")
    group_frames += load_candidate_predictions(processed_dir, "tuned_tabular_fusion_predictions_speaker_disjoint_*.csv")

    fixed_merged = merge_scores(fixed_frames)
    group_merged = merge_scores(group_frames)

    fixed_info, fixed_w, fixed_pred = evaluate_setting(
        "fixed_split_train_test", fixed_merged, args.num_models, args.weight_search_iters, args.seed
    )
    group_info, group_w, group_pred = evaluate_setting(
        "speaker_disjoint_split", group_merged, args.num_models, args.weight_search_iters, args.seed
    )

    metrics = {
        "fixed_split_train_test": fixed_info,
        "speaker_disjoint_split": group_info,
    }
    dump_json(metrics, processed_dir / "ensemble_threshold_metrics.json")

    weights_df = pd.concat([fixed_w, group_w], ignore_index=True)
    weights_df.to_csv(processed_dir / "ensemble_threshold_weights.csv", index=False)

    preds_df = pd.concat([fixed_pred, group_pred], ignore_index=True)
    preds_df.to_csv(processed_dir / "ensemble_threshold_predictions.csv", index=False)

    print("Saved ensemble threshold outputs.")


if __name__ == "__main__":
    main()

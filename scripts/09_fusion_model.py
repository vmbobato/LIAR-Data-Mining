#!/usr/bin/env python
"""Compare text-only versus text+graph models for low-truth detection.

Trains and evaluates configurable classifiers under two settings:
- fixed LIAR split (train->test)
- speaker-disjoint split (GroupShuffleSplit by speaker)

Inputs:
- `data/processed/liar_base.parquet`
- `data/interim/text_embeddings_*.parquet`
- `data/interim/graph_embeddings.parquet`

Outputs (suffix-aware):
- `data/processed/fusion_model_metrics{suffix}.json`
- `data/processed/fusion_model_feature_table{suffix}.parquet`
- `data/processed/fusion_model_predictions_*{suffix}.csv`
- `data/processed/fusion_model_confusion_matrices{suffix}.csv`
- `data/processed/fusion_model_curve_points{suffix}.csv`
- `data/processed/fusion_model_threshold_sweep{suffix}.csv`
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from liar_mining.config import ensure_dirs, load_config
from liar_mining.io import load_parquet, save_parquet
from liar_mining.modeling import compute_binary_metrics, dump_json


MODEL_CHOICES = ["logistic_regression", "linear_svm", "random_forest", "gradient_boosting"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate baseline vs fused models for low-truth detection.")
    parser.add_argument("--config", default="configs/liar_research.yaml")
    parser.add_argument("--text-file", default="text_embeddings_tfidf.parquet")
    parser.add_argument("--text-svd-dim", type=int, default=200)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--model", choices=MODEL_CHOICES, default="logistic_regression")
    parser.add_argument("--output-suffix", default="", help="Optional suffix (example: _tfidf_lr)")
    return parser.parse_args()


def with_suffix(name: str, suffix: str) -> str:
    if not suffix:
        return name
    if "." in name:
        stem, ext = name.rsplit(".", 1)
        return f"{stem}{suffix}.{ext}"
    return f"{name}{suffix}"


def make_model(model_name: str, max_iter: int):
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=max_iter, class_weight="balanced", solver="lbfgs")
    if model_name == "linear_svm":
        return LinearSVC(max_iter=max_iter, class_weight="balanced")
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
    if model_name == "gradient_boosting":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=8,
            max_iter=400,
            random_state=42,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    smin, smax = float(np.min(scores)), float(np.max(scores))
    if smax - smin < 1e-12:
        return np.full_like(scores, 0.5, dtype=np.float64)
    return (scores - smin) / (smax - smin)


def fit_eval(
    model_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_iter: int,
) -> dict:
    clf = make_model(model_name, max_iter)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)

    try:
        clf.fit(x_train, y_train, sample_weight=sample_weight)
    except TypeError:
        clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(x_test)[:, 1]
    elif hasattr(clf, "decision_function"):
        y_score = normalize_scores(clf.decision_function(x_test))
    else:
        y_score = y_pred.astype(float)

    metrics = compute_binary_metrics(y_test, y_pred, y_score)
    return {
        "metrics": metrics,
        "y_pred": y_pred,
        "y_score": y_score,
    }


def maybe_reduce(x_train: np.ndarray, x_test: np.ndarray, n_dim: int) -> tuple[np.ndarray, np.ndarray]:
    if x_train.shape[1] <= n_dim:
        return x_train, x_test
    svd = TruncatedSVD(n_components=n_dim, random_state=42)
    return svd.fit_transform(x_train), svd.transform(x_test)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    out_suffix = args.output_suffix.strip()

    base = load_parquet(Path(cfg["paths"]["processed_dir"]) / "liar_base.parquet")
    text = load_parquet(Path(cfg["paths"]["interim_dir"]) / args.text_file)
    graph = load_parquet(Path(cfg["paths"]["interim_dir"]) / "graph_embeddings.parquet")

    df = base[["id", "split", "speaker", "is_low_truth", "label", "statement"]].merge(text, on="id", how="inner")
    df = df.merge(graph, on="id", how="inner")

    text_cols = [c for c in df.columns if c.startswith("text_emb_")]
    graph_cols = [c for c in df.columns if c.startswith("graph_emb_")]

    train = df[df["split"] == "train"]
    test = df[df["split"] == "test"]

    x_train_text = train[text_cols].to_numpy(dtype=np.float32)
    x_test_text = test[text_cols].to_numpy(dtype=np.float32)
    x_train_text_r, x_test_text_r = maybe_reduce(x_train_text, x_test_text, args.text_svd_dim)

    x_train_fused = np.hstack([x_train_text_r, train[graph_cols].to_numpy(dtype=np.float32)])
    x_test_fused = np.hstack([x_test_text_r, test[graph_cols].to_numpy(dtype=np.float32)])

    y_train = train["is_low_truth"].to_numpy()
    y_test = test["is_low_truth"].to_numpy()

    fixed_text = fit_eval(args.model, x_train_text_r, y_train, x_test_text_r, y_test, args.max_iter)
    fixed_fused = fit_eval(args.model, x_train_fused, y_train, x_test_fused, y_test, args.max_iter)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    idx_train, idx_test = next(gss.split(df, df["is_low_truth"], groups=df["speaker"]))
    dtr = df.iloc[idx_train]
    dte = df.iloc[idx_test]

    xtr_text = dtr[text_cols].to_numpy(dtype=np.float32)
    xte_text = dte[text_cols].to_numpy(dtype=np.float32)
    xtr_text_r, xte_text_r = maybe_reduce(xtr_text, xte_text, args.text_svd_dim)

    xtr_fused = np.hstack([xtr_text_r, dtr[graph_cols].to_numpy(dtype=np.float32)])
    xte_fused = np.hstack([xte_text_r, dte[graph_cols].to_numpy(dtype=np.float32)])

    ytr = dtr["is_low_truth"].to_numpy()
    yte = dte["is_low_truth"].to_numpy()

    group_text = fit_eval(args.model, xtr_text_r, ytr, xte_text_r, yte, args.max_iter)
    group_fused = fit_eval(args.model, xtr_fused, ytr, xte_fused, yte, args.max_iter)

    metrics = {
        "fixed_split_train_test": {
            "text_only": fixed_text["metrics"],
            "text_plus_graph": fixed_fused["metrics"],
        },
        "speaker_disjoint_split": {
            "text_only": group_text["metrics"],
            "text_plus_graph": group_fused["metrics"],
        },
        "notes": {
            "target": "is_low_truth",
            "text_file": args.text_file,
            "speaker_disjoint_test_size": 0.2,
            "classifier": args.model,
        },
    }

    processed_dir = Path(cfg["paths"]["processed_dir"])

    metrics_path = processed_dir / with_suffix("fusion_model_metrics.json", out_suffix)
    dump_json(metrics, metrics_path)

    fixed_pred = test[["id", "speaker", "label", "is_low_truth"]].copy()
    fixed_pred["setting"] = "fixed_split_train_test"
    fixed_pred["text_only_pred"] = fixed_text["y_pred"]
    fixed_pred["text_only_score"] = fixed_text["y_score"]
    fixed_pred["text_plus_graph_pred"] = fixed_fused["y_pred"]
    fixed_pred["text_plus_graph_score"] = fixed_fused["y_score"]
    fixed_pred.to_csv(processed_dir / with_suffix("fusion_model_predictions_fixed_split.csv", out_suffix), index=False)

    group_pred = dte[["id", "speaker", "label", "is_low_truth"]].copy()
    group_pred["setting"] = "speaker_disjoint_split"
    group_pred["text_only_pred"] = group_text["y_pred"]
    group_pred["text_only_score"] = group_text["y_score"]
    group_pred["text_plus_graph_pred"] = group_fused["y_pred"]
    group_pred["text_plus_graph_score"] = group_fused["y_score"]
    group_pred.to_csv(processed_dir / with_suffix("fusion_model_predictions_speaker_disjoint.csv", out_suffix), index=False)

    conf_rows = []
    conf_specs = [
        ("fixed_split_train_test", y_test, fixed_text["y_pred"], "text_only"),
        ("fixed_split_train_test", y_test, fixed_fused["y_pred"], "text_plus_graph"),
        ("speaker_disjoint_split", yte, group_text["y_pred"], "text_only"),
        ("speaker_disjoint_split", yte, group_fused["y_pred"], "text_plus_graph"),
    ]
    for setting, y_true, y_pred, model_variant in conf_specs:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        conf_rows.append(
            {
                "setting": setting,
                "model": model_variant,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )
    pd.DataFrame(conf_rows).to_csv(processed_dir / with_suffix("fusion_model_confusion_matrices.csv", out_suffix), index=False)

    curve_rows = []
    curve_specs = [
        ("fixed_split_train_test", y_test, fixed_text["y_score"], "text_only"),
        ("fixed_split_train_test", y_test, fixed_fused["y_score"], "text_plus_graph"),
        ("speaker_disjoint_split", yte, group_text["y_score"], "text_only"),
        ("speaker_disjoint_split", yte, group_fused["y_score"], "text_plus_graph"),
    ]
    for setting, y_true, y_score, model_variant in curve_specs:
        p, r, pr_t = precision_recall_curve(y_true, y_score)
        fpr, tpr, roc_t = roc_curve(y_true, y_score)
        for i in range(len(p)):
            curve_rows.append(
                {
                    "setting": setting,
                    "model": model_variant,
                    "curve": "pr",
                    "x": float(r[i]),
                    "y": float(p[i]),
                    "threshold": float(pr_t[i - 1]) if i > 0 and i - 1 < len(pr_t) else np.nan,
                }
            )
        for i in range(len(fpr)):
            curve_rows.append(
                {
                    "setting": setting,
                    "model": model_variant,
                    "curve": "roc",
                    "x": float(fpr[i]),
                    "y": float(tpr[i]),
                    "threshold": float(roc_t[i]) if i < len(roc_t) else np.nan,
                }
            )
    pd.DataFrame(curve_rows).to_csv(processed_dir / with_suffix("fusion_model_curve_points.csv", out_suffix), index=False)

    th_rows = []
    for setting, y_true, y_score, model_variant in curve_specs:
        quantiles = np.quantile(y_score, np.linspace(0.05, 0.95, 17))
        thresholds = np.unique(np.round(quantiles, 6))
        for th in thresholds:
            y_hat = (y_score >= th).astype(int)
            m = compute_binary_metrics(y_true, y_hat, y_score)
            th_rows.append(
                {
                    "setting": setting,
                    "model": model_variant,
                    "threshold": float(th),
                    "accuracy": m["accuracy"],
                    "precision": m["precision"],
                    "recall": m["recall"],
                    "f1": m["f1"],
                    "roc_auc": m["roc_auc"],
                }
            )
    pd.DataFrame(th_rows).to_csv(processed_dir / with_suffix("fusion_model_threshold_sweep.csv", out_suffix), index=False)

    feature_table = df[["id", "split", "speaker", "label", "is_low_truth"] + text_cols[: min(20, len(text_cols))] + graph_cols[: min(20, len(graph_cols))]]
    feature_path = processed_dir / with_suffix("fusion_model_feature_table.parquet", out_suffix)
    save_parquet(feature_table, feature_path)

    print(f"Saved metrics: {metrics_path}")
    print(f"Saved feature table (sample columns): {feature_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import re
from typing import Dict

import numpy as np
import pandas as pd

US_STATES = {
    "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut",
    "delaware", "florida", "georgia", "hawaii", "idaho", "illinois", "indiana", "iowa",
    "kansas", "kentucky", "louisiana", "maine", "maryland", "massachusetts", "michigan",
    "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire",
    "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio",
    "oklahoma", "oregon", "pennsylvania", "rhode island", "south carolina", "south dakota",
    "tennessee", "texas", "utah", "vermont", "virginia", "washington", "west virginia",
    "wisconsin", "wyoming", "district of columbia",
}


def _clean_text(x: object) -> str:
    if pd.isna(x):
        return ""
    text = str(x).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_state(state: object) -> str:
    s = _clean_text(state).lower()
    if not s:
        return "Unknown"
    return s.title() if s in US_STATES else "Other"


def simplify_party(party: object) -> str:
    p = _clean_text(party).lower()
    if not p:
        return "unknown"
    if "democrat" in p:
        return "democrat"
    if "republican" in p or p == "gop":
        return "republican"
    third_party_tokens = ["libertarian", "green", "independent", "constitution", "reform"]
    if any(tok in p for tok in third_party_tokens):
        return "third-party"
    non_party_tokens = ["journalist", "news", "organization", "columnist", "blogger"]
    if any(tok in p for tok in non_party_tokens):
        return "non-party-role"
    return "unknown"


def add_engineered_columns(df: pd.DataFrame, labels_cfg: Dict) -> pd.DataFrame:
    out = df.copy()

    out["statement"] = out["statement"].fillna("").astype(str)
    out["subject"] = out["subject"].fillna("").astype(str)
    out["speaker"] = out["speaker"].fillna("unknown").astype(str)
    out["state_clean"] = out["state"].map(normalize_state)
    out["party_grouped"] = out["party"].map(simplify_party)

    out["statement_length_chars"] = out["statement"].str.len()
    out["statement_length_words"] = out["statement"].str.split().str.len().fillna(0).astype(int)

    prior_cols = [
        "barely_true_counts", "false_counts", "half_true_counts",
        "mostly_true_counts", "pants_on_fire_counts",
    ]
    for c in prior_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)

    out["total_prior_statements"] = out[prior_cols].sum(axis=1)

    low_truth_set = set(labels_cfg["low_truth"])
    out["is_low_truth"] = out["label"].isin(low_truth_set).astype(int)

    split_order = {"train": 0, "valid": 1, "test": 2}
    out["split_index"] = out["split"].map(split_order).fillna(99).astype(int)
    out["year_proxy"] = out["split_index"]

    return out


def parse_subject_list(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["id", "label", "is_low_truth", "subject", "speaker", "party_grouped"]].copy()
    out["subject"] = out["subject"].fillna("")
    out["subject"] = out["subject"].str.split(",")
    out = out.explode("subject")
    out["subject"] = out["subject"].astype(str).str.strip().str.lower()
    out = out[out["subject"] != ""]
    return out


def safe_merge(left: pd.DataFrame, right: pd.DataFrame, on: str, how: str = "left") -> pd.DataFrame:
    merged = left.merge(right, on=on, how=how)
    if merged.shape[0] != left.shape[0]:
        raise ValueError("Merge changed row count. Check duplicate keys in right dataframe.")
    return merged

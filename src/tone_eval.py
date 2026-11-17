"""Shared helpers for the transformer notebooks: load the splits and score predictions.

Labels follow the dataset: 0 = dovish, 1 = hawkish, 2 = neutral.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"

LABELS = ["dovish", "hawkish", "neutral"]


def load_splits():
    """Kept sentences (no ambiguous or header rows) with every split column, indexed by row_id."""
    return pd.read_csv(PROCESSED_DIR / "splits.csv").set_index("row_id")


def score(y_true, y_pred):
    return {
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", labels=[0, 1, 2], zero_division=0),
    }


def score_table(pred_df, name):
    """Overall and minutes-only scores for a frame with `label`, `pred` and `doc_type` columns."""
    rows = []
    for subset, part in [("all", pred_df), ("minutes", pred_df[pred_df["doc_type"] == "minutes"])]:
        rows.append({"model": name, "subset": subset, **score(part["label"], part["pred"])})
    return pd.DataFrame(rows)

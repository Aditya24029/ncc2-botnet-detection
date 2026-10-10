"""Shared config, splitting and evaluation used by every notebook.

Keeping this in one place guarantees that FT-Transformer, TabTransformer and
SAINT are trained and tested on exactly the same rows with the same metrics.
"""
import json
import os

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

SEED = 42
TARGET = "ActivityLabel"

RAW_DIR = "../data/raw"
PROCESSED_DIR = "../data/processed"
MODELS_DIR = "../models"
RESULTS_DIR = "../results"

CATEGORICAL_COLS = ["Proto", "Sport", "Dir", "Dport", "State"]
CONTINUOUS_COLS = [
    "Dur", "sTos", "dTos", "TotPkts", "TotBytes", "SrcBytes",
    "Hour", "Minute", "Second",
]


def three_way_split(df, target=TARGET, seed=SEED):
    """70 / 15 / 15 stratified split.

    Validation is used for checkpoint selection, test is touched once at the
    end. Same arithmetic as the SAINT notebooks, so all models share the
    identical test rows.
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=seed, stratify=df[target]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=seed, stratify=temp_df[target]
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_botnet": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_botnet": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_botnet": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def report_and_save(model_name, sensor, y_true, y_pred, results_dir=RESULTS_DIR):
    """Print metrics + classification report and save them as JSON."""
    m = compute_metrics(y_true, y_pred)
    print("=" * 60)
    print(f"{model_name} | {sensor}")
    print("=" * 60)
    for k in ["accuracy", "precision_botnet", "recall_botnet", "f1_botnet", "f1_macro", "f1_weighted"]:
        print(f"{k:<17}: {m[k]:.4f}")
    print()
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:", m["confusion_matrix"])

    os.makedirs(results_dir, exist_ok=True)
    out = {"model": model_name, "sensor": sensor, **m}
    fname = f"{model_name}_{sensor}".lower().replace(" ", "_").replace("-", "")
    with open(os.path.join(results_dir, fname + ".json"), "w") as f:
        json.dump(out, f, indent=2)
    return m

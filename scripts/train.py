"""Reproducible training + MLflow experiment tracking.

Usage:
    python scripts/train.py --data data/transactions.csv --version 1.0.0

CSV must contain the model input columns and the binary target `is_fraud`.
The script never applies SMOTE to validation/test data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES = [
    "transaction_id", "amount", "transaction_hour", "merchant_category",
    "foreign_transaction", "location_mismatch", "device_trust_score",
    "velocity_last_24h", "cardholder_age",
]
TARGET = "is_fraud"


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["merchant_category"])
        ],
        remainder="passthrough",
    )
    classifier = XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.05,
        max_depth=3,
        n_estimators=80,
        subsample=0.8,
        tree_method="hist",
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    return Pipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("classifier", classifier),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--version", default="1.0.0", help="Model version label")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = ROOT / data_path
    frame = pd.read_csv(data_path)

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    frame = frame[required].dropna().copy()
    frame[TARGET] = frame[TARGET].astype(int)
    if set(frame[TARGET].unique()) != {0, 1}:
        raise ValueError("is_fraud must contain both binary labels 0 and 1.")

    X, y = frame[FEATURES], frame[TARGET]
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.25,
        stratify=y_trainval, random_state=42
    )

    candidate = build_pipeline()
    candidate.fit(X_train, y_train)
    val_prob = candidate.predict_proba(X_val)[:, 1]
    threshold_grid = np.arange(0.10, 0.91, 0.05)
    threshold = max(
        threshold_grid,
        key=lambda t: f1_score(y_val, (val_prob >= t).astype(int), zero_division=0)
    )

    # Refit using train + validation; test remains untouched until final evaluation.
    final_model = build_pipeline()
    final_model.fit(X_trainval, y_trainval)
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    metrics = {
        "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, test_prob)),
        "test_pr_auc": float(average_precision_score(y_test, test_prob)),
        "selected_threshold": float(threshold),
    }

    model_dir = ROOT / "models"
    model_dir.mkdir(exist_ok=True)
    joblib.dump(final_model.named_steps["preprocessor"], model_dir / "preprocessor.pkl")
    final_model.named_steps["classifier"].save_model(str(model_dir / "xgb_classifier.json"))

    artifact_hashes = {}
    for artifact_name in ("preprocessor.pkl", "xgb_classifier.json"):
        artifact_bytes = (model_dir / artifact_name).read_bytes()
        artifact_hashes[artifact_name] = hashlib.sha256(artifact_bytes).hexdigest()

    metadata = {
        "model_name": "credit-card-fraud-detector",
        "model_version": args.version,
        "threshold": float(threshold),
        "features": FEATURES,
        "metrics": metrics,
        "artifact_sha256": artifact_hashes,
        "training_versions": {
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
    }
    (model_dir / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    mlflow.set_experiment("credit-card-fraud-detection")
    with mlflow.start_run(run_name=f"xgboost-{args.version}"):
        mlflow.log_param("model_version", args.version)
        mlflow.log_param("smote_random_state", 42)
        mlflow.log_param("threshold_selection", "validation_max_f1")
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(model_dir), artifact_path="model_artifacts")
        mlflow.sklearn.log_model(final_model, artifact_path="fraud_detection_pipeline")

    print(json.dumps(metrics, indent=2))
    print(f"Saved model artifacts in: {model_dir}")


if __name__ == "__main__":
    main()

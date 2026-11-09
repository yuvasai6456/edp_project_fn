import json
import logging
import os
from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
XGBOOST_PATH = MODEL_DIR / "xgb_classifier.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

logger = logging.getLogger(__name__)

# The current exported deployment artifacts were tested with threshold 0.50
# in the notebook's deployment cells. Override only after validating a new model.
DEFAULT_THRESHOLD = 0.50

def _read_metadata():
    if METADATA_PATH.exists():
        try:
            with METADATA_PATH.open(encoding="utf-8") as file:
                return json.load(file)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read model metadata; using defaults.")
    return {}

MODEL_METADATA = _read_metadata()
THRESHOLD = float(
    os.getenv("FRAUD_THRESHOLD", MODEL_METADATA.get("threshold", DEFAULT_THRESHOLD))
)
if not 0.0 < THRESHOLD < 1.0:
    raise ValueError("FRAUD_THRESHOLD must be between 0 and 1.")

for artifact in (PREPROCESSOR_PATH, XGBOOST_PATH):
    if not artifact.is_file():
        raise FileNotFoundError(f"Required model artifact is missing: {artifact.name}")

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    classifier = XGBClassifier()
    classifier.load_model(str(XGBOOST_PATH))
except Exception as exc:
    raise RuntimeError(
        "Could not load model artifacts. Check the pinned scikit-learn and "
        "XGBoost versions in requirements.txt."
    ) from exc

FEATURES = [
    "transaction_id",
    "amount",
    "transaction_hour",
    "merchant_category",
    "foreign_transaction",
    "location_mismatch",
    "device_trust_score",
    "velocity_last_24h",
    "cardholder_age",
]


def predict_fraud(transaction):
    """Return the fraud class and probability for one transaction."""
    missing = [name for name in FEATURES if name not in transaction]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")

    features = pd.DataFrame(
        [{name: transaction[name] for name in FEATURES}],
        columns=FEATURES,
    )
    transformed_features = preprocessor.transform(features)

    classes = list(classifier.classes_)
    if 1 not in classes:
        raise ValueError(f"Fraud class 1 is not present in model classes: {classes}")

    fraud_probability = float(
        classifier.predict_proba(transformed_features)[0][classes.index(1)]
    )
    prediction = "Fraud" if fraud_probability >= THRESHOLD else "Legitimate"

    return {
        "prediction": prediction,
        "fraud_probability": fraud_probability,
        "fraud_probability_pct": round(fraud_probability * 100, 2),
        "threshold": THRESHOLD,
        "model_version": _model_version(),
    }


def _model_version():
    return MODEL_METADATA.get("model_version", "baseline-1")

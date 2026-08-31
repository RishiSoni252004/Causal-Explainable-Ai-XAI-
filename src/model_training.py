"""
model_training.py — Train, evaluate, compare, and persist ML models.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models() -> dict:
    """Return a dict of model name → untrained estimator."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=42
        ),
    }


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(models: dict, X_train, X_test, y_train, y_test):
    """Train each model and return a results dict keyed by model name.

    Each entry contains:
        fitted_model, accuracy, precision, recall, f1, confusion_matrix
    """
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results[name] = {
            "fitted_model": model,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }
    return results


def results_to_dataframe(results: dict) -> pd.DataFrame:
    """Flatten results dict into a tidy comparison DataFrame."""
    rows = []
    for name, metrics in results.items():
        rows.append(
            {
                "Model": name,
                "Accuracy": round(metrics["accuracy"], 4),
                "Precision": round(metrics["precision"], 4),
                "Recall": round(metrics["recall"], 4),
                "F1 Score": round(metrics["f1"], 4),
            }
        )
    return pd.DataFrame(rows)


def select_best_model(results: dict) -> tuple[str, dict]:
    """Return (name, metrics_dict) of the model with the highest F1 score."""
    best_name = max(results, key=lambda n: results[n]["f1"])
    return best_name, results[best_name]


def save_model(model, scaler, label_encoder, feature_names, directory: str = "models"):
    """Persist the model, scaler, label encoder, and feature names."""
    os.makedirs(directory, exist_ok=True)
    joblib.dump(model, os.path.join(directory, "best_model.joblib"))
    joblib.dump(scaler, os.path.join(directory, "scaler.joblib"))
    joblib.dump(label_encoder, os.path.join(directory, "label_encoder.joblib"))
    joblib.dump(feature_names, os.path.join(directory, "feature_names.joblib"))

"""
model_training.py — Train, evaluate, compare, and persist ML models.

Supports:
    • Legacy demo student-performance model training
    • Modern pipeline architecture for uploaded datasets (Classification & Regression)
    • Preventing data leakage with sklearn Pipelines & ColumnTransformers
    • Automatic model comparison & best model selection
"""

import os
import datetime
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

__all__ = [
    "get_models",
    "train_and_evaluate",
    "results_to_dataframe",
    "select_best_model",
    "save_model",
    "get_classification_models",
    "get_regression_models",
    "build_column_transformer",
    "train_and_evaluate_pipeline",
    "pipeline_results_to_dataframe",
]


# ---------------------------------------------------------------------------
# Legacy Model Definitions & Training (Demo Dataset Support)
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


# ===========================================================================
# Advanced Pipeline Architecture for Uploaded Datasets (Classification & Regression)
# ===========================================================================


def get_classification_models() -> dict:
    """Return dictionary of classification estimators with balanced class weights."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, class_weight="balanced", random_state=42
        ),
        "Decision Tree Classifier": DecisionTreeClassifier(
            class_weight="balanced", random_state=42
        ),
    }


def get_regression_models() -> dict:
    """Return dictionary of regression estimators."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "Decision Tree Regressor": DecisionTreeRegressor(
            random_state=42
        ),
    }


def build_column_transformer(df: pd.DataFrame, feature_cols: list[str]) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Construct an sklearn ColumnTransformer with SimpleImputer, StandardScaler and OneHotEncoder.
    
    Prevents data leakage by encapsulating all transformations inside the pipeline.
    """
    numeric_features = []
    categorical_features = []

    for c in feature_cols:
        col_series = df[c]
        if pd.api.types.is_numeric_dtype(col_series) and not pd.api.types.is_bool_dtype(col_series):
            numeric_features.append(c)
        else:
            categorical_features.append(c)

    transformers = []

    if numeric_features:
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipeline, numeric_features))

    if categorical_features:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("cat", cat_pipeline, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_features, categorical_features


def train_and_evaluate_pipeline(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    problem_type: str = "Classification",
    test_size: float = 0.2,
    random_state: int = 42,
    save_dir: str = "models",
) -> dict:
    """End-to-end training and evaluation pipeline for uploaded datasets.
    
    Handles:
        • Numerical & categorical preprocessing with ColumnTransformer
        • Missing value imputation
        • Data leakage prevention using sklearn Pipeline
        • Class imbalance handling with class_weight='balanced'
        • Evaluation with problem-specific metrics (Classification vs. Regression)
        • Automatic best model selection
        • Model & metadata persistence
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")
    
    missing_feats = [f for f in feature_cols if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Features not found in dataset: {missing_feats}")

    # Work on clean copy of features and target
    work_df = df[feature_cols + [target_col]].copy()

    # Drop rows where target is null
    work_df = work_df.dropna(subset=[target_col]).reset_index(drop=True)
    if len(work_df) < 5:
        raise ValueError("Insufficient rows for model training (fewer than 5 valid records).")

    X = work_df[feature_cols].copy()
    y_raw = work_df[target_col].copy()

    # Build ColumnTransformer based on training data features
    preprocessor, numeric_features, categorical_features = build_column_transformer(X, feature_cols)

    # Format categorical columns as object to prevent mixed type errors
    for c in categorical_features:
        X[c] = X[c].astype(object)

    target_encoder = None
    classes = None

    if problem_type == "Classification":
        target_encoder = LabelEncoder()
        y_encoded = target_encoder.fit_transform(y_raw.astype(str))
        classes = list(target_encoder.classes_)

        # Stratified train/test split if minimum class count allows
        class_counts = pd.Series(y_encoded).value_counts()
        can_stratify = class_counts.min() >= 2
        stratify_arg = y_encoded if can_stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=random_state, stratify=stratify_arg
        )

        models = get_classification_models()
        results = {}

        for name, estimator in models.items():
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            avg_method = "binary" if len(classes) == 2 else "weighted"

            acc = float(accuracy_score(y_test, y_pred))
            prec = float(precision_score(y_test, y_pred, average=avg_method, zero_division=0))
            rec = float(recall_score(y_test, y_pred, average=avg_method, zero_division=0))
            f1 = float(f1_score(y_test, y_pred, average=avg_method, zero_division=0))
            cm = confusion_matrix(y_test, y_pred)

            results[name] = {
                "pipeline": pipe,
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "confusion_matrix": cm,
            }

        # Select best model by F1 score
        best_name = max(results, key=lambda n: results[n]["f1"])
        best_metrics = results[best_name]
        best_pipeline = best_metrics["pipeline"]

    else:  # Regression
        y_numeric = pd.to_numeric(y_raw, errors="coerce")
        valid_y_mask = y_numeric.notna()
        X = X[valid_y_mask].reset_index(drop=True)
        y_numeric = y_numeric[valid_y_mask].reset_index(drop=True)

        if len(y_numeric) < 5:
            raise ValueError("Insufficient valid numerical targets for regression.")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_numeric, test_size=test_size, random_state=random_state
        )

        models = get_regression_models()
        results = {}

        for name, estimator in models.items():
            pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            mae = float(mean_absolute_error(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test, y_pred))

            results[name] = {
                "pipeline": pipe,
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "r2": r2,
                "y_pred": y_pred,
            }

        # Select best model by R2 score (or lowest RMSE if all R2 negative)
        best_name = max(results, key=lambda n: results[n]["r2"])
        best_metrics = results[best_name]
        best_pipeline = best_metrics["pipeline"]

    # Persist the best pipeline and metadata
    metadata = {
        "problem_type": problem_type,
        "best_model_name": best_name,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "classes": classes,
        "n_samples": len(work_df),
        "test_size": test_size,
        "created_at": datetime.datetime.now().isoformat(),
    }

    os.makedirs(save_dir, exist_ok=True)
    joblib.dump(best_pipeline, os.path.join(save_dir, "uploaded_pipeline.joblib"))
    joblib.dump(metadata, os.path.join(save_dir, "uploaded_metadata.joblib"))

    return {
        "problem_type": problem_type,
        "results": results,
        "best_name": best_name,
        "best_pipeline": best_pipeline,
        "best_metrics": best_metrics,
        "target_col": target_col,
        "feature_cols": feature_cols,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "classes": classes,
        "target_encoder": target_encoder,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "metadata": metadata,
    }


def pipeline_results_to_dataframe(results: dict, problem_type: str = "Classification") -> pd.DataFrame:
    """Format pipeline comparison results into a clean, typed DataFrame."""
    rows = []
    if problem_type == "Classification":
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
    else:  # Regression
        for name, metrics in results.items():
            rows.append(
                {
                    "Model": name,
                    "MAE": round(metrics["mae"], 4),
                    "MSE": round(metrics["mse"], 4),
                    "RMSE": round(metrics["rmse"], 4),
                    "R² Score": round(metrics["r2"], 4),
                }
            )
    return pd.DataFrame(rows)


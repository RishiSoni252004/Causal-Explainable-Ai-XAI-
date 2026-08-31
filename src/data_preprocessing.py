"""
data_preprocessing.py — Load, clean, and prepare the student dataset for modelling.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(path: str) -> pd.DataFrame:
    """Read the CSV dataset from *path* and return a DataFrame."""
    return pd.read_csv(path)


def preprocess(
    df: pd.DataFrame,
    target_col: str = "result",
    test_size: float = 0.2,
    random_state: int = 42,
):
    """End-to-end preprocessing pipeline.

    Steps
    -----
    1. Encode the target column (PASS/FAIL → 1/0).
    2. Separate features and target.
    3. Train/test split.
    4. Standardise numerical features (fit on train, transform both).

    Returns
    -------
    X_train, X_test, y_train, y_test : arrays
        Ready for model consumption.
    scaler : StandardScaler
        Fitted scaler (saved alongside the model for inference).
    label_encoder : LabelEncoder
        Fitted encoder for decoding predictions back to labels.
    feature_names : list[str]
        Ordered list of feature column names.
    """
    df = df.copy()

    # Encode target
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])  # FAIL=0, PASS=1

    feature_names = [c for c in df.columns if c != target_col]
    X = df[feature_names]
    y = df[target_col]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, le, feature_names

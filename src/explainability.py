"""
explainability.py — SHAP-based Explainable AI for the Causal XAI project.

Provides:
    • Global feature importance (from the model itself)
    • SHAP-based feature importance (model-agnostic)
    • Individual prediction explanations with human-readable narratives
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Global feature importance (model-native)
# ---------------------------------------------------------------------------

def get_global_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Extract feature importance from the trained model.

    For tree-based models (RandomForest) uses `feature_importances_`.
    For linear models (LogisticRegression) uses absolute coefficient values.

    Returns a DataFrame sorted by importance (descending).
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model does not expose feature importances or coefficients.")

    df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Normalise to [0, 1] for easier comparison
    df["Importance"] = df["Importance"] / df["Importance"].max()
    return df


def plot_global_importance(importance_df: pd.DataFrame):
    """Return a horizontal bar chart (matplotlib Figure) of feature importance."""
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    colors = plt.cm.RdPu(np.linspace(0.35, 0.85, len(importance_df)))[::-1]

    bars = ax.barh(
        importance_df["Feature"],
        importance_df["Importance"],
        color=colors,
        edgecolor="none",
        height=0.6,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Relative Importance", color="#ccc", fontsize=10)
    ax.tick_params(colors="#ccc")
    for spine in ax.spines.values():
        spine.set_color("#333")

    # Value labels
    for bar, val in zip(bars, importance_df["Importance"]):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            color="#ccc",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# SHAP explanations
# ---------------------------------------------------------------------------

def compute_shap_values(model, X_data: np.ndarray, feature_names: list[str]):
    """Compute SHAP values using the appropriate explainer.

    Uses TreeExplainer for tree-based models and LinearExplainer for
    linear models, falling back to KernelExplainer if needed.

    Returns
    -------
    shap_values : shap.Explanation
        SHAP explanation object.
    explainer : shap.Explainer
        The fitted explainer (reusable for individual predictions).
    """
    # Choose the right explainer
    try:
        if hasattr(model, "estimators_"):
            # Tree-based (RandomForest, GradientBoosting, …)
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, "coef_"):
            # Linear model
            explainer = shap.LinearExplainer(model, X_data)
        else:
            # Fallback — slower but model-agnostic
            explainer = shap.KernelExplainer(model.predict_proba, X_data[:100])
    except Exception:
        # Ultimate fallback
        explainer = shap.KernelExplainer(model.predict_proba, X_data[:100])

    raw_shap = explainer.shap_values(X_data)

    # For binary classifiers some explainers return a list [class_0, class_1];
    # we always want the values for the positive class (PASS = 1).
    if isinstance(raw_shap, list):
        shap_array = raw_shap[1]
    elif raw_shap.ndim == 3:
        shap_array = raw_shap[:, :, 1]
    else:
        shap_array = raw_shap

    explanation = shap.Explanation(
        values=shap_array,
        base_values=(
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        ),
        data=X_data,
        feature_names=feature_names,
    )

    return explanation, explainer


def plot_shap_summary(explanation):
    """Return a SHAP beeswarm / summary plot as a matplotlib Figure."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    shap.summary_plot(explanation, show=False, plot_size=None)

    # Style the current axes produced by SHAP
    current_fig = plt.gcf()
    current_fig.patch.set_facecolor("#0e1117")
    for a in current_fig.axes:
        a.set_facecolor("#0e1117")
        a.tick_params(colors="#ccc")
        a.xaxis.label.set_color("#ccc")
        a.yaxis.label.set_color("#ccc")
        for spine in a.spines.values():
            spine.set_color("#333")
    plt.tight_layout()
    return current_fig


def get_shap_importance(explanation, feature_names: list[str]) -> pd.DataFrame:
    """Derive mean |SHAP| importance per feature."""
    mean_abs = np.abs(explanation.values).mean(axis=0)
    df = pd.DataFrame(
        {"Feature": feature_names, "Mean |SHAP|": mean_abs}
    ).sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Individual prediction explanation
# ---------------------------------------------------------------------------

def explain_individual(
    model,
    explainer,
    scaler,
    label_encoder,
    feature_names: list[str],
    input_values: dict,
) -> dict:
    """Generate a complete explanation for a single student.

    Parameters
    ----------
    input_values : dict
        Raw (unscaled) feature values, e.g. {"study_hours": 6, ...}.

    Returns
    -------
    dict with keys:
        prediction_label, prediction_proba,
        shap_values (per-feature), top_factors, narrative
    """
    # Build input array in the correct feature order
    raw = np.array([[input_values[f] for f in feature_names]])
    scaled = scaler.transform(raw)

    # Predict
    pred_class = model.predict(scaled)[0]
    pred_proba = model.predict_proba(scaled)[0]
    label = label_encoder.inverse_transform([pred_class])[0]
    confidence = float(pred_proba[pred_class])

    # SHAP for this single instance
    single_shap = explainer.shap_values(scaled)
    if isinstance(single_shap, list):
        sv = single_shap[1][0]
    elif single_shap.ndim == 3:
        sv = single_shap[0, :, 1]
    else:
        sv = single_shap[0]

    # Build per-feature breakdown
    feature_effects = pd.DataFrame(
        {
            "Feature": feature_names,
            "Value": [input_values[f] for f in feature_names],
            "SHAP": sv,
            "Abs SHAP": np.abs(sv),
        }
    ).sort_values("Abs SHAP", ascending=False).reset_index(drop=True)

    # Top 3 factors
    top3 = feature_effects.head(3)

    # Build human-readable narrative
    narrative = _build_narrative(label, top3)

    return {
        "prediction_label": label,
        "prediction_proba": confidence,
        "feature_effects": feature_effects,
        "top_factors": top3,
        "narrative": narrative,
        "shap_values": sv,
        "base_value": (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        ),
        "scaled_input": scaled[0],
    }


def _build_narrative(label: str, top3: pd.DataFrame) -> str:
    """Create a human-readable explanation sentence."""
    reasons = []
    for _, row in top3.iterrows():
        name = row["Feature"].replace("_", " ")
        val = row["Value"]
        direction = "high" if row["SHAP"] > 0 else "low"
        reasons.append(f"{direction} {name} ({val})")

    reason_str = ", ".join(reasons[:-1]) + f", and {reasons[-1]}" if len(reasons) > 2 else " and ".join(reasons)

    if label == "PASS":
        return (
            f"The student is predicted to **PASS** mainly because of "
            f"{reason_str}."
        )
    else:
        return (
            f"The student is predicted to **FAIL** mainly because of "
            f"{reason_str}."
        )

"""
explainability.py — SHAP-based Explainable AI for the Causal XAI project.

Provides:
    • Global feature importance (from the model itself)
    • SHAP-based feature importance (model-agnostic, tree & linear explainers)
    • Pipeline-aware SHAP explanation for both uploaded datasets & demo data
    • Classification and Regression support
    • Individual prediction explanations with human-readable narratives
    • Feature contribution visualizations (diverging bar & waterfall)
    • Clear distinction between statistical SHAP attributions and causal effects
"""

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

__all__ = [
    "clean_feature_name",
    "get_global_feature_importance",
    "get_pipeline_global_importance",
    "plot_global_importance",
    "compute_shap_values",
    "build_pipeline_shap_explainer",
    "compute_pipeline_shap",
    "plot_shap_summary",
    "get_shap_importance",
    "get_pipeline_shap_importance",
    "explain_individual",
    "explain_pipeline_prediction",
    "plot_feature_contributions",
    "plot_shap_waterfall",
]


# ---------------------------------------------------------------------------
# Feature name cleaning utility
# ---------------------------------------------------------------------------

def clean_feature_name(name: str, categorical_features: list[str] | None = None) -> str:
    """Transform sklearn ColumnTransformer feature names into user-friendly names.

    E.g.:
        num__study_hours -> study hours
        cat__department_Engineering -> department: Engineering
        cat__gender_Male -> gender: Male
    """
    if name.startswith("num__"):
        return name[5:].replace("_", " ")
    elif name.startswith("cat__"):
        raw = name[5:]
        if categorical_features:
            for cat_col in sorted(categorical_features, key=len, reverse=True):
                if raw.startswith(f"{cat_col}_"):
                    val = raw[len(cat_col) + 1:]
                    return f"{cat_col.replace('_', ' ')}: {val}"
        if "_" in raw:
            parts = raw.split("_", 1)
            return f"{parts[0].replace('_', ' ')}: {parts[1]}"
        return raw.replace("_", " ")
    return name.replace("_", " ")


# ---------------------------------------------------------------------------
# Global feature importance (model-native) - Legacy & Pipeline
# ---------------------------------------------------------------------------

def get_global_feature_importance(model, feature_names: list[str]) -> pd.DataFrame:
    """Extract feature importance from a standalone trained model (Demo flow)."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).flatten()
    else:
        raise ValueError("Model does not expose feature importances or coefficients.")

    df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False).reset_index(drop=True)

    max_imp = df["Importance"].max()
    df["Importance"] = df["Importance"] / (max_imp if max_imp > 0 else 1.0)
    return df


def get_pipeline_global_importance(pipeline, categorical_features: list[str] | None = None) -> pd.DataFrame:
    """Extract feature importance from a fitted scikit-learn Pipeline with ColumnTransformer."""
    prep = pipeline.named_steps.get("preprocessor")
    model = pipeline.named_steps.get("model")

    if prep is None or model is None:
        raise ValueError("Pipeline must contain 'preprocessor' and 'model' steps.")

    feat_names_out = list(prep.get_feature_names_out())
    clean_names = [clean_feature_name(f, categorical_features) for f in feat_names_out]

    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.abs(model.coef_)
        if coef.ndim > 1:
            importances = np.mean(coef, axis=0)
        else:
            importances = coef.flatten()
    else:
        # Fallback uniform if model does not expose feature importances or coefficients
        importances = np.ones(len(clean_names), dtype=float) / max(len(clean_names), 1)

    df = pd.DataFrame(
        {
            "Feature": clean_names,
            "Raw_Feature": feat_names_out,
            "Importance": importances,
        }
    ).sort_values("Importance", ascending=False).reset_index(drop=True)

    max_imp = df["Importance"].max()
    df["Relative_Importance"] = df["Importance"] / (max_imp if max_imp > 0 else 1.0)
    return df


def plot_global_importance(importance_df: pd.DataFrame, feature_col: str = "Feature", val_col: str = "Importance"):
    """Return a horizontal bar chart (matplotlib Figure) of feature importance."""
    df = importance_df.head(15).copy()
    fig, ax = plt.subplots(figsize=(8, max(3.5, len(df) * 0.35)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    colors = plt.cm.RdPu(np.linspace(0.35, 0.85, len(df)))[::-1]

    bars = ax.barh(
        df[feature_col],
        df[val_col],
        color=colors,
        edgecolor="none",
        height=0.6,
    )
    ax.invert_yaxis()
    ax.set_xlabel("Relative Importance", color="#ccc", fontsize=10)
    ax.tick_params(colors="#ccc")
    for spine in ax.spines.values():
        spine.set_color("#333")

    for bar, val in zip(bars, df[val_col]):
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
# SHAP explanations — Legacy & Pipeline
# ---------------------------------------------------------------------------

def compute_shap_values(model, X_data: np.ndarray, feature_names: list[str]):
    """Compute SHAP values using the appropriate explainer (Legacy Demo)."""
    try:
        if hasattr(model, "estimators_") or hasattr(model, "tree_"):
            explainer = shap.TreeExplainer(model)
        elif hasattr(model, "coef_"):
            explainer = shap.LinearExplainer(model, X_data)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_data[:100])
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, X_data[:100])

    raw_shap = explainer.shap_values(X_data)

    if isinstance(raw_shap, list):
        shap_array = raw_shap[1] if len(raw_shap) > 1 else raw_shap[0]
    elif raw_shap.ndim == 3:
        shap_array = raw_shap[:, :, 1] if raw_shap.shape[2] > 1 else raw_shap[:, :, 0]
    else:
        shap_array = raw_shap

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1] if len(base_val) > 1 else base_val[0]

    explanation = shap.Explanation(
        values=shap_array,
        base_values=base_val,
        data=X_data,
        feature_names=feature_names,
    )

    return explanation, explainer


def build_pipeline_shap_explainer(pipeline, X_background: pd.DataFrame, problem_type: str = "Classification"):
    """Construct the optimal SHAP explainer for a fitted pipeline and background dataset."""
    prep = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Transform background samples (sample up to 100 rows for speed and stability)
    bg_df = X_background.sample(n=min(len(X_background), 100), random_state=42) if len(X_background) > 100 else X_background
    X_bg_trans = prep.transform(bg_df)

    # Choose explainer
    if hasattr(model, "tree_") or hasattr(model, "estimators_"):
        explainer = shap.TreeExplainer(model)
        explainer_type = "TreeExplainer"
    elif hasattr(model, "coef_"):
        explainer = shap.LinearExplainer(model, X_bg_trans)
        explainer_type = "LinearExplainer"
    else:
        # Fallback to KernelExplainer
        predict_fn = model.predict_proba if (problem_type == "Classification" and hasattr(model, "predict_proba")) else model.predict
        explainer = shap.KernelExplainer(predict_fn, X_bg_trans[:50])
        explainer_type = "KernelExplainer"

    return explainer, explainer_type


def compute_pipeline_shap(
    pipeline,
    X_data: pd.DataFrame,
    categorical_features: list[str] | None = None,
    problem_type: str = "Classification",
    max_samples: int = 100,
    class_idx: int = 1,
):
    """Compute SHAP values for an entire dataset through a fitted Pipeline."""
    prep = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # Sample if needed to keep evaluation fast & responsive
    eval_df = X_data.sample(n=min(len(X_data), max_samples), random_state=42) if len(X_data) > max_samples else X_data
    X_trans = prep.transform(eval_df)

    feat_names_out = list(prep.get_feature_names_out())
    clean_names = [clean_feature_name(f, categorical_features) for f in feat_names_out]

    # Build or instantiate explainer
    explainer, expl_type = build_pipeline_shap_explainer(pipeline, X_data, problem_type=problem_type)

    raw_shap = explainer.shap_values(X_trans)

    if problem_type == "Classification":
        if isinstance(raw_shap, list):
            target_idx = class_idx if class_idx < len(raw_shap) else (len(raw_shap) - 1)
            shap_array = raw_shap[target_idx]
        elif raw_shap.ndim == 3:
            target_idx = class_idx if class_idx < raw_shap.shape[2] else (raw_shap.shape[2] - 1)
            shap_array = raw_shap[:, :, target_idx]
        else:
            shap_array = raw_shap

        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            target_idx = class_idx if class_idx < len(base_val) else (len(base_val) - 1)
            base_val = float(base_val[target_idx])
        else:
            base_val = float(base_val)
    else:  # Regression
        shap_array = raw_shap[0] if (hasattr(raw_shap, "ndim") and raw_shap.ndim == 3) else raw_shap
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(np.ravel(base_val)[0])
        else:
            base_val = float(base_val)

    base_values_arr = np.full(len(X_trans), base_val)

    explanation = shap.Explanation(
        values=shap_array,
        base_values=base_values_arr,
        data=X_trans,
        feature_names=clean_names,
    )

    return explanation, explainer, expl_type


def plot_shap_summary(explanation):
    """Return a SHAP beeswarm / summary plot as a matplotlib Figure."""
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    shap.summary_plot(explanation, show=False, plot_size=None)

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
    total_shap = df["Mean |SHAP|"].sum()
    df["Importance %"] = (df["Mean |SHAP|"] / (total_shap if total_shap > 0 else 1.0)) * 100.0
    return df


def get_pipeline_shap_importance(explanation) -> pd.DataFrame:
    """Derive mean |SHAP| importance per feature from a pipeline SHAP explanation."""
    return get_shap_importance(explanation, explanation.feature_names)


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
    """Legacy individual explanation for the demo student model."""
    raw = pd.DataFrame([[input_values[f] for f in feature_names]], columns=feature_names)
    scaled = scaler.transform(raw)

    pred_class = model.predict(scaled)[0]
    pred_proba = model.predict_proba(scaled)[0]
    label = label_encoder.inverse_transform([pred_class])[0]
    confidence = float(pred_proba[pred_class])

    single_shap = explainer.shap_values(scaled)
    if isinstance(single_shap, list):
        sv = single_shap[1][0]
    elif single_shap.ndim == 3:
        sv = single_shap[0, :, 1]
    else:
        sv = single_shap[0]

    feature_effects = pd.DataFrame(
        {
            "Feature": feature_names,
            "Value": [input_values[f] for f in feature_names],
            "SHAP": sv,
            "Abs SHAP": np.abs(sv),
        }
    ).sort_values("Abs SHAP", ascending=False).reset_index(drop=True)

    top3 = feature_effects.head(3)
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
    """Create a human-readable explanation sentence for the demo dataset."""
    reasons = []
    for _, row in top3.iterrows():
        name = row["Feature"].replace("_", " ")
        val = row["Value"]
        direction = "high" if row["SHAP"] > 0 else "low"
        reasons.append(f"{direction} {name} ({val})")

    reason_str = ", ".join(reasons[:-1]) + f", and {reasons[-1]}" if len(reasons) > 2 else " and ".join(reasons)

    if label == "PASS":
        return f"The student is predicted to **PASS** mainly because of {reason_str}."
    else:
        return f"The student is predicted to **FAIL** mainly because of {reason_str}."


def explain_pipeline_prediction(
    pipeline,
    explainer,
    input_values: dict,
    feature_cols: list[str],
    categorical_features: list[str] | None = None,
    problem_type: str = "Classification",
    target_encoder=None,
    target_col: str = "Target",
) -> dict:
    """Generate a comprehensive, pipeline-aware SHAP explanation for a single prediction.

    Supports both Classification and Regression with arbitrary numeric and
    one-hot encoded categorical features.
    """
    prep = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    # 1. Prepare 1-row DataFrame in correct feature order
    single_df = pd.DataFrame([input_values])[feature_cols]

    # 2. Compute model prediction
    pred = pipeline.predict(single_df)[0]
    pred_proba = None
    confidence = None
    classes = None
    class_probs = {}

    if problem_type == "Classification":
        if hasattr(pipeline, "predict_proba"):
            try:
                proba = pipeline.predict_proba(single_df)[0]
                pred_proba = proba
                confidence = float(proba[pred]) if isinstance(pred, (int, np.integer)) and pred < len(proba) else float(max(proba))
                if target_encoder and hasattr(target_encoder, "classes_"):
                    classes = list(target_encoder.classes_)
                    class_probs = {cls_name: float(p) for cls_name, p in zip(classes, proba)}
                else:
                    class_probs = {f"Class {i}": float(p) for i, p in enumerate(proba)}
            except Exception:
                pass

        if target_encoder and hasattr(target_encoder, "inverse_transform"):
            try:
                pred_label = target_encoder.inverse_transform([pred])[0]
            except Exception:
                pred_label = str(pred)
        else:
            pred_label = str(pred)
        pred_value = None
    else:  # Regression
        pred_label = f"{float(pred):,.2f}"
        pred_value = float(pred)

    # 3. Transform inputs through ColumnTransformer
    X_single_trans = prep.transform(single_df)
    feat_names_out = list(prep.get_feature_names_out())
    clean_names = [clean_feature_name(f, categorical_features) for f in feat_names_out]

    # 4. Compute SHAP values for single instance
    raw_shap = explainer.shap_values(X_single_trans)

    if problem_type == "Classification":
        target_idx = int(pred) if (isinstance(pred, (int, np.integer)) and pred < 2) else 1
        if isinstance(raw_shap, list):
            sv = raw_shap[target_idx][0] if target_idx < len(raw_shap) else raw_shap[-1][0]
        elif raw_shap.ndim == 3:
            sv = raw_shap[0, :, target_idx] if target_idx < raw_shap.shape[2] else raw_shap[0, :, -1]
        elif raw_shap.ndim == 2:
            sv = raw_shap[0]
        else:
            sv = raw_shap

        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(base_val[target_idx]) if target_idx < len(base_val) else float(base_val[-1])
        else:
            base_val = float(base_val)
    else:  # Regression
        sv = raw_shap[0] if (hasattr(raw_shap, "ndim") and raw_shap.ndim == 2) else raw_shap
        base_val = explainer.expected_value
        if isinstance(base_val, (list, np.ndarray)):
            base_val = float(np.ravel(base_val)[0])
        else:
            base_val = float(base_val)

    # 5. Extract input values corresponding to transformed features
    input_display_vals = []
    for raw_f in feat_names_out:
        if raw_f.startswith("num__"):
            orig_col = raw_f[5:]
            input_display_vals.append(input_values.get(orig_col, "—"))
        elif raw_f.startswith("cat__"):
            cat_raw = raw_f[5:]
            matched_val = "—"
            if categorical_features:
                for c_col in categorical_features:
                    if cat_raw.startswith(f"{c_col}_"):
                        cat_level = cat_raw[len(c_col) + 1:]
                        user_val = input_values.get(c_col, None)
                        matched_val = f"Yes ({cat_level})" if str(user_val) == str(cat_level) else f"No ({user_val})"
                        break
            input_display_vals.append(matched_val)
        else:
            input_display_vals.append("—")

    # 6. Build feature effects table (ensuring Input_Value is string for pyarrow compatibility)
    feature_effects = pd.DataFrame(
        {
            "Feature": clean_names,
            "Input_Value": [str(v) if v is not None else "—" for v in input_display_vals],
            "SHAP": sv,
            "Abs_SHAP": np.abs(sv),
            "Raw_Feature": feat_names_out,
        }
    ).sort_values("Abs_SHAP", ascending=False).reset_index(drop=True)

    # Categorize positive and negative contributions
    positive_factors = feature_effects[feature_effects["SHAP"] > 0].sort_values("SHAP", ascending=False).reset_index(drop=True)
    negative_factors = feature_effects[feature_effects["SHAP"] < 0].sort_values("SHAP", ascending=True).reset_index(drop=True)

    # 7. Generate dynamic narrative
    narrative = _build_pipeline_narrative(
        problem_type=problem_type,
        pred_label=pred_label,
        pred_value=pred_value,
        confidence=confidence,
        base_val=base_val,
        target_col=target_col,
        positive_factors=positive_factors,
        negative_factors=negative_factors,
    )

    # Single explanation object for waterfall
    single_explanation = shap.Explanation(
        values=sv,
        base_values=base_val,
        data=X_single_trans[0],
        feature_names=clean_names,
    )

    return {
        "problem_type": problem_type,
        "prediction_label": pred_label,
        "prediction_value": pred_value,
        "prediction_proba": confidence,
        "class_probabilities": class_probs,
        "base_value": base_val,
        "net_shap_shift": float(np.sum(sv)),
        "feature_effects": feature_effects,
        "positive_factors": positive_factors,
        "negative_factors": negative_factors,
        "narrative": narrative,
        "single_explanation": single_explanation,
        "transformed_values": X_single_trans[0],
        "transformed_names": clean_names,
    }


def _build_pipeline_narrative(
    problem_type: str,
    pred_label: str,
    pred_value: float | None,
    confidence: float | None,
    base_val: float,
    target_col: str,
    positive_factors: pd.DataFrame,
    negative_factors: pd.DataFrame,
) -> str:
    """Synthesize a natural, human-readable narrative explaining why the model produced this prediction."""
    pos_top = positive_factors.head(3)
    neg_top = negative_factors.head(3)

    if problem_type == "Classification":
        conf_str = f" with **{confidence:.1%} confidence**" if confidence is not None else ""
        narrative = f"The model predicts **{pred_label}** for `{target_col}`{conf_str} (dataset baseline: **{base_val:.2f}**).\n\n"

        if not pos_top.empty:
            reasons = [f"**{r['Feature']}** (value: `{r['Input_Value']}`, contribution: **+{r['SHAP']:.3f}**)" for _, r in pos_top.iterrows()]
            narrative += f"• **Upward Drivers (Pushed toward {pred_label}):** {', '.join(reasons)}.\n\n"

        if not neg_top.empty:
            reasons_neg = [f"**{r['Feature']}** (value: `{r['Input_Value']}`, contribution: **{r['SHAP']:.3f}**)" for _, r in neg_top.iterrows()]
            narrative += f"• **Downward Counter-factors:** {', '.join(reasons_neg)}.\n\n"

        narrative += f"*Summary:* The model shifted its output from the average baseline of **{base_val:.2f}** due to the balance of the factors above."
    else:  # Regression
        diff = pred_value - base_val
        direction = "above" if diff >= 0 else "below"
        narrative = f"The model predicts `{target_col}` = **{pred_label}**, which is **{abs(diff):,.2f} {direction}** the dataset average baseline of **{base_val:,.2f}**.\n\n"

        if not pos_top.empty:
            reasons = [f"**{r['Feature']}** (value: `{r['Input_Value']}`, contribution: **+{r['SHAP']:,.2f}**)" for _, r in pos_top.iterrows()]
            narrative += f"• **Primary Upward Drivers:** {', '.join(reasons)}.\n\n"

        if not neg_top.empty:
            reasons_neg = [f"**{r['Feature']}** (value: `{r['Input_Value']}`, contribution: **{r['SHAP']:,.2f}**)" for _, r in neg_top.iterrows()]
            narrative += f"• **Primary Downward Pressures:** {', '.join(reasons_neg)}.\n\n"

        narrative += f"*Summary:* Combining all positive and negative contributions resulted in a net shift of **{diff:+,.2f}** from the expected value {base_val:,.2f} to arrive at **{pred_label}**."

    return narrative


def plot_feature_contributions(
    feature_effects: pd.DataFrame,
    base_value: float | None = None,
    pred_value: float | None = None,
    problem_type: str = "Classification",
    target_label: str = "",
    top_n: int = 12,
):
    """Generate a horizontal diverging bar chart contrasting positive and negative SHAP contributions."""
    df = feature_effects.copy().head(top_n)
    df = df.iloc[::-1].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.38)))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    colors = ["#10b981" if s > 0 else "#ef4444" for s in df["SHAP"]]
    bars = ax.barh(df["Feature"], df["SHAP"], color=colors, height=0.6, edgecolor="none")

    ax.axvline(0, color="#64748b", linestyle="--", linewidth=1, alpha=0.8)

    ax.tick_params(colors="#ccc", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#333")

    val_max = df["SHAP"].abs().max() if not df.empty else 1.0
    for bar, val in zip(bars, df["SHAP"]):
        offset = val_max * 0.02 * (1 if val >= 0 else -1)
        ha = "left" if val >= 0 else "right"
        fmt_val = f"{val:+.3f}" if problem_type == "Classification" else f"{val:+,.1f}"
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            fmt_val,
            va="center",
            ha=ha,
            color="#e2e8f0",
            fontsize=9,
            fontweight="bold",
        )

    title_suffix = f" for '{target_label}'" if target_label else ""
    ax.set_title(f"Feature Contributions (SHAP Values){title_suffix}", color="#f8fafc", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("SHAP Impact (Green = Pushes Upward, Red = Pulls Downward)", color="#cbd5e1", fontsize=9)
    plt.tight_layout()
    return fig


def plot_shap_waterfall(single_explanation):
    """Return a SHAP waterfall plot as a matplotlib Figure."""
    fig = plt.figure(figsize=(9, 5))
    try:
        shap.plots.waterfall(single_explanation, show=False)
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
    except Exception:
        plt.close(fig)
        return None

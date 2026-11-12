"""
counterfactual.py — Counterfactual Simulator for the Causal XAI project.

Provides:
    • Functions to compare original and modified observations across both
      legacy demo datasets and custom uploaded ML pipelines.
    • Support for both Classification and Regression models.
    • Identification and tracking of changed numerical and categorical features.
    • Detailed human-readable explanation generation.
    • Graphical comparison visualization.
    • Clear scientific disclaimers separating model sensitivity from causal guarantees.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    "simulate_counterfactual",
    "simulate_pipeline_counterfactual",
    "plot_counterfactual_comparison",
]


def simulate_counterfactual(
    model,
    scaler,
    label_encoder,
    feature_names: list[str],
    original_inputs: dict,
    modified_inputs: dict,
) -> dict:
    """Legacy counterfactual simulation for the student performance demo model."""
    raw_orig = pd.DataFrame([[original_inputs[f] for f in feature_names]], columns=feature_names)
    scaled_orig = scaler.transform(raw_orig)
    pred_orig_class = model.predict(scaled_orig)[0]
    pred_orig_proba = model.predict_proba(scaled_orig)[0]
    label_orig = label_encoder.inverse_transform([pred_orig_class])[0]
    conf_orig = float(pred_orig_proba[pred_orig_class])

    raw_mod = pd.DataFrame([[modified_inputs[f] for f in feature_names]], columns=feature_names)
    scaled_mod = scaler.transform(raw_mod)
    pred_mod_class = model.predict(scaled_mod)[0]
    pred_mod_proba = model.predict_proba(scaled_mod)[0]
    label_mod = label_encoder.inverse_transform([pred_mod_class])[0]
    conf_mod = float(pred_mod_proba[pred_mod_class])

    changed_features = []
    changed_details = []
    for f in feature_names:
        vo = original_inputs[f]
        vm = modified_inputs[f]
        if vo != vm:
            clean_f = f.replace("_", " ").title()
            changed_features.append(clean_f)
            diff_str = f"{vm - vo:+g}" if isinstance(vo, (int, float)) and isinstance(vm, (int, float)) else f"'{vo}' ➔ '{vm}'"
            changed_details.append({
                "Feature": clean_f,
                "Original Value": str(vo),
                "Modified Value": str(vm),
                "Change Description": f"{vo} ➔ {vm} ({diff_str})",
            })

    narrative = ""
    if len(changed_features) > 0:
        changes_str = ", ".join(changed_features[:-1]) + f", and {changed_features[-1]}" if len(changed_features) > 2 else " and ".join(changed_features)
        if label_orig != label_mod:
            narrative = f"Changing {changes_str} led the model to flip its prediction from **{label_orig}** ({conf_orig:.1%} confidence) to **{label_mod}** ({conf_mod:.1%} confidence)."
        else:
            diff_conf = conf_mod - conf_orig
            narrative = f"Changing {changes_str} altered model confidence by {diff_conf:+.1%}, but the model's prediction remained **{label_orig}** ({conf_mod:.1%} confidence)."
    else:
        narrative = "No input variables were changed between baseline and counterfactual."

    return {
        "problem_type": "Classification",
        "original_label": label_orig,
        "original_confidence": conf_orig,
        "original_class_probs": {"FAIL": float(pred_orig_proba[0]), "PASS": float(pred_orig_proba[1])},
        "modified_label": label_mod,
        "modified_confidence": conf_mod,
        "modified_class_probs": {"FAIL": float(pred_mod_proba[0]), "PASS": float(pred_mod_proba[1])},
        "changed_features": changed_features,
        "changed_details": changed_details,
        "changes_df": pd.DataFrame(changed_details),
        "narrative": narrative,
        "prediction_changed": label_orig != label_mod,
    }


def simulate_pipeline_counterfactual(
    pipeline,
    original_inputs: dict,
    modified_inputs: dict,
    feature_cols: list[str],
    problem_type: str = "Classification",
    target_encoder=None,
    target_col: str = "Target",
) -> dict:
    """Simulate a what-if counterfactual observation through a fitted scikit-learn Pipeline.

    Supports both Classification and Regression models with arbitrary numerical
    and categorical features.
    """
    # 1. Prepare 1-row DataFrames in correct column order
    df_orig = pd.DataFrame([original_inputs])[feature_cols]
    df_mod = pd.DataFrame([modified_inputs])[feature_cols]

    # 2. Evaluate original baseline prediction
    pred_orig_raw = pipeline.predict(df_orig)[0]
    conf_orig = None
    class_probs_orig = None
    val_orig = None

    if problem_type == "Classification":
        if hasattr(pipeline, "predict_proba"):
            try:
                proba_orig = pipeline.predict_proba(df_orig)[0]
                conf_orig = (
                    float(proba_orig[pred_orig_raw])
                    if isinstance(pred_orig_raw, (int, np.integer)) and pred_orig_raw < len(proba_orig)
                    else float(max(proba_orig))
                )
                if target_encoder and hasattr(target_encoder, "classes_"):
                    classes = list(target_encoder.classes_)
                    class_probs_orig = {str(c): float(p) for c, p in zip(classes, proba_orig)}
                else:
                    class_probs_orig = {f"Class {i}": float(p) for i, p in enumerate(proba_orig)}
            except Exception:
                pass

        if target_encoder and hasattr(target_encoder, "inverse_transform"):
            try:
                label_orig = str(target_encoder.inverse_transform([pred_orig_raw])[0])
            except Exception:
                label_orig = str(pred_orig_raw)
        else:
            label_orig = str(pred_orig_raw)
    else:  # Regression
        val_orig = float(pred_orig_raw)
        label_orig = f"{val_orig:,.2f}"

    # 3. Evaluate modified counterfactual prediction
    pred_mod_raw = pipeline.predict(df_mod)[0]
    conf_mod = None
    class_probs_mod = None
    val_mod = None

    if problem_type == "Classification":
        if hasattr(pipeline, "predict_proba"):
            try:
                proba_mod = pipeline.predict_proba(df_mod)[0]
                conf_mod = (
                    float(proba_mod[pred_mod_raw])
                    if isinstance(pred_mod_raw, (int, np.integer)) and pred_mod_raw < len(proba_mod)
                    else float(max(proba_mod))
                )
                if target_encoder and hasattr(target_encoder, "classes_"):
                    classes = list(target_encoder.classes_)
                    class_probs_mod = {str(c): float(p) for c, p in zip(classes, proba_mod)}
                else:
                    class_probs_mod = {f"Class {i}": float(p) for i, p in enumerate(proba_mod)}
            except Exception:
                pass

        if target_encoder and hasattr(target_encoder, "inverse_transform"):
            try:
                label_mod = str(target_encoder.inverse_transform([pred_mod_raw])[0])
            except Exception:
                label_mod = str(pred_mod_raw)
        else:
            label_mod = str(pred_mod_raw)
    else:  # Regression
        val_mod = float(pred_mod_raw)
        label_mod = f"{val_mod:,.2f}"

    # 4. Identify changed features and audit differences
    changed_features = []
    changed_details = []
    all_comparison_rows = []

    for f in feature_cols:
        vo = original_inputs.get(f)
        vm = modified_inputs.get(f)
        is_changed = str(vo) != str(vm)
        clean_name = f.replace("_", " ").title()

        if isinstance(vo, (int, float, np.number)) and isinstance(vm, (int, float, np.number)):
            diff_num = float(vm) - float(vo)
            shift_text = f"{diff_num:+g}" if diff_num != 0 else "0"
            change_desc = f"{vo} ➔ {vm} ({shift_text})" if is_changed else f"{vo} (Unchanged)"
        else:
            shift_text = "Category changed" if is_changed else "None"
            change_desc = f"'{vo}' ➔ '{vm}'" if is_changed else f"'{vo}' (Unchanged)"

        row_info = {
            "Feature": clean_name,
            "Original Value": str(vo),
            "Modified Value": str(vm),
            "Shift": shift_text,
            "Change Description": change_desc,
            "Status": "✏️ Modified" if is_changed else "Identical",
        }
        all_comparison_rows.append(row_info)

        if is_changed:
            changed_features.append(clean_name)
            changed_details.append(row_info)

    # 5. Build human-readable narrative & shift statistics
    prediction_changed = False
    diff_value = None
    pct_change = None
    diff_conf = None

    if problem_type == "Classification":
        prediction_changed = (label_orig != label_mod)
        if conf_orig is not None and conf_mod is not None:
            diff_conf = conf_mod - conf_orig

        if changed_features:
            changes_list_str = ", ".join(changed_features[:-1]) + f", and {changed_features[-1]}" if len(changed_features) > 2 else " and ".join(changed_features)
            if prediction_changed:
                conf_orig_txt = f" ({conf_orig:.1%} confidence)" if conf_orig is not None else ""
                conf_mod_txt = f" ({conf_mod:.1%} confidence)" if conf_mod is not None else ""
                narrative = (
                    f"Modifying **{changes_list_str}** caused the model to flip its prediction for `{target_col}` from "
                    f"**{label_orig}**{conf_orig_txt} to **{label_mod}**{conf_mod_txt}."
                )
            else:
                conf_shift_txt = f" by {diff_conf:+.1%}" if diff_conf is not None else ""
                conf_mod_txt = f" at {conf_mod:.1%} confidence" if conf_mod is not None else ""
                narrative = (
                    f"Modifying **{changes_list_str}** shifted the prediction confidence{conf_shift_txt}, "
                    f"but the predicted class remained **{label_orig}**{conf_mod_txt}."
                )
        else:
            narrative = "No input features were modified between the baseline and counterfactual scenario."

    else:  # Regression
        diff_value = val_mod - val_orig
        pct_change = (diff_value / abs(val_orig) * 100.0) if val_orig != 0 else 0.0
        prediction_changed = abs(diff_value) > 1e-6

        if changed_features:
            changes_list_str = ", ".join(changed_features[:-1]) + f", and {changed_features[-1]}" if len(changed_features) > 2 else " and ".join(changed_features)
            direction_str = "increased" if diff_value >= 0 else "decreased"
            narrative = (
                f"Modifying **{changes_list_str}** {direction_str} the model's predicted `{target_col}` from "
                f"**{label_orig}** to **{label_mod}** (absolute difference: **{diff_value:+,.2f}**, or **{pct_change:+.1f}%**)."
            )
        else:
            narrative = "No input features were modified between the baseline and counterfactual scenario."

    return {
        "problem_type": problem_type,
        "target_col": target_col,
        "original_label": label_orig,
        "original_value": val_orig,
        "original_confidence": conf_orig,
        "original_class_probs": class_probs_orig,
        "modified_label": label_mod,
        "modified_value": val_mod,
        "modified_confidence": conf_mod,
        "modified_class_probs": class_probs_mod,
        "prediction_changed": prediction_changed,
        "diff_value": diff_value,
        "pct_change": pct_change,
        "diff_confidence": diff_conf,
        "changed_features": changed_features,
        "changed_details": changed_details,
        "changes_df": pd.DataFrame(changed_details),
        "all_comparison_df": pd.DataFrame(all_comparison_rows),
        "narrative": narrative,
    }


def plot_counterfactual_comparison(sim_result: dict):
    """Render a clean, dark-themed visual comparison of baseline vs counterfactual predictions."""
    prob_type = sim_result["problem_type"]
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    if prob_type == "Classification" and sim_result.get("original_class_probs") and sim_result.get("modified_class_probs"):
        classes = list(sim_result["original_class_probs"].keys())
        y_pos = np.arange(len(classes))
        bar_height = 0.32

        orig_probs = [sim_result["original_class_probs"][c] * 100 for c in classes]
        mod_probs = [sim_result["modified_class_probs"][c] * 100 for c in classes]

        bars1 = ax.barh(y_pos - bar_height / 2, orig_probs, bar_height, label="Original Baseline", color="#6366f1", alpha=0.9)
        bars2 = ax.barh(y_pos + bar_height / 2, mod_probs, bar_height, label="Counterfactual (Modified)", color="#10b981", alpha=0.9)

        for b, p in zip(bars1, orig_probs):
            ax.text(b.get_width() + 1.2, b.get_y() + b.get_height() / 2, f"{p:.1f}%", va="center", color="#cbd5e1", fontsize=8.5)

        for b, p in zip(bars2, mod_probs):
            ax.text(b.get_width() + 1.2, b.get_y() + b.get_height() / 2, f"{p:.1f}%", va="center", color="#6ee7b7", fontsize=8.5, fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels([str(c) for c in classes], color="#cbd5e1", fontsize=10, fontweight="bold")
        ax.set_xlabel("Model Predicted Probability (%)", color="#cbd5e1", fontsize=9.5)
        ax.set_xlim(0, 115)
        ax.set_title("Classification Class Probability Shift", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)
        ax.legend(loc="lower right", facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=9)

    else:  # Regression or single output
        scenarios = ["Original Baseline", "Counterfactual"]
        val_orig = sim_result["original_value"] if sim_result.get("original_value") is not None else 0.0
        val_mod = sim_result["modified_value"] if sim_result.get("modified_value") is not None else 0.0
        values = [val_orig, val_mod]

        color_mod = "#10b981" if val_mod >= val_orig else "#f43f5e"
        colors = ["#6366f1", color_mod]

        bars = ax.barh(scenarios, values, color=colors, height=0.45, alpha=0.9)
        max_val = max(abs(v) for v in values) if values else 1.0
        for b, v in zip(bars, values):
            pos_x = b.get_width() + (max_val * 0.02 if v >= 0 else -max_val * 0.02)
            ha = "left" if v >= 0 else "right"
            ax.text(pos_x, b.get_y() + b.get_height() / 2, f"{v:,.2f}", va="center", ha=ha, color="#ffffff", fontweight="bold", fontsize=9.5)

        target_name = sim_result.get("target_col", "Target")
        ax.set_xlabel(f"Predicted Value ({target_name})", color="#cbd5e1", fontsize=9.5)
        lower_lim = min(0, min(values) * 1.15) if min(values) < 0 else 0
        upper_lim = max(values) * 1.2 if max(values) > 0 else 1.0
        ax.set_xlim(lower_lim, upper_lim)
        diff_val = val_mod - val_orig
        pct_val = sim_result.get("pct_change", 0.0)
        ax.set_title(f"Predicted Output Shift: {diff_val:+,.2f} ({pct_val:+.1f}%)", color="#f8fafc", fontsize=11, fontweight="bold", pad=10)

    ax.tick_params(colors="#ccc")
    for s in ax.spines.values():
        s.set_color("#333")

    plt.tight_layout()
    return fig

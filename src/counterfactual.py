"""
counterfactual.py — Counterfactual Simulator for the Causal XAI project.

Provides:
    • Functions to compare original and modified predictions.
    • Generation of human-readable explanations of the changes.
"""

import numpy as np
import pandas as pd

def simulate_counterfactual(
    model,
    scaler,
    label_encoder,
    feature_names: list[str],
    original_inputs: dict,
    modified_inputs: dict,
) -> dict:
    """
    Simulate the effect of changing input variables on the model's prediction.
    
    Returns a dictionary containing the original prediction, the modified prediction,
    and a comparison.
    """
    # Original prediction
    raw_orig = np.array([[original_inputs[f] for f in feature_names]])
    scaled_orig = scaler.transform(raw_orig)
    pred_orig_class = model.predict(scaled_orig)[0]
    pred_orig_proba = model.predict_proba(scaled_orig)[0]
    label_orig = label_encoder.inverse_transform([pred_orig_class])[0]
    conf_orig = float(pred_orig_proba[pred_orig_class])

    # Modified prediction
    raw_mod = np.array([[modified_inputs[f] for f in feature_names]])
    scaled_mod = scaler.transform(raw_mod)
    pred_mod_class = model.predict(scaled_mod)[0]
    pred_mod_proba = model.predict_proba(scaled_mod)[0]
    label_mod = label_encoder.inverse_transform([pred_mod_class])[0]
    conf_mod = float(pred_mod_proba[pred_mod_class])

    # Determine changed features
    changed_features = []
    for f in feature_names:
        if original_inputs[f] != modified_inputs[f]:
            changed_features.append(f.replace("_", " "))

    narrative = ""
    if len(changed_features) > 0:
        changes_str = ", ".join(changed_features[:-1]) + f", and {changed_features[-1]}" if len(changed_features) > 2 else " and ".join(changed_features)
        
        if label_orig != label_mod:
            narrative = f"Changing {changes_str} may lead the model to change its prediction from {label_orig} to {label_mod}."
        else:
            narrative = f"Changing {changes_str} affects the confidence, but the model's prediction remains {label_orig}."
    else:
        narrative = "No variables were changed."

    return {
        "original_label": label_orig,
        "original_confidence": conf_orig,
        "modified_label": label_mod,
        "modified_confidence": conf_mod,
        "changed_features": changed_features,
        "narrative": narrative
    }

"""
utils.py — Utility helpers for the Causal XAI project.

Provides:
    - Synthetic dataset generation with realistic causal relationships
    - General-purpose helper functions
"""

import os
import numpy as np
import pandas as pd


def generate_student_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic synthetic student-performance dataset.

    The data encodes plausible causal relationships:
        • Higher study_hours, attendance, previous_scores, assignments_completed,
          and class_participation all *increase* the probability of passing.
        • Moderate sleep_hours (~7-8 h) are beneficial; too little or too much
          sleep hurts performance.

    Parameters
    ----------
    n_samples : int
        Number of student records to create.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature columns and a binary ``result`` column
        (``"PASS"`` / ``"FAIL"``).
    """
    rng = np.random.default_rng(seed)

    # --- Feature generation (with mild inter-feature correlations) ----------
    study_hours = rng.normal(loc=5.0, scale=2.0, size=n_samples).clip(0, 12)
    attendance = rng.normal(loc=75, scale=15, size=n_samples).clip(0, 100)
    previous_scores = rng.normal(loc=60, scale=15, size=n_samples).clip(0, 100)
    sleep_hours = rng.normal(loc=7.0, scale=1.5, size=n_samples).clip(3, 12)
    assignments_completed = rng.integers(low=0, high=11, size=n_samples)
    class_participation = rng.integers(low=0, high=11, size=n_samples)

    # --- Latent score (higher → more likely to pass) -----------------------
    score = (
        0.25 * study_hours
        + 0.02 * attendance
        + 0.03 * previous_scores
        + 0.15 * _sleep_benefit(sleep_hours)
        + 0.20 * assignments_completed
        + 0.10 * class_participation
    )

    # Add noise so the boundary isn't perfectly deterministic
    noise = rng.normal(loc=0, scale=0.8, size=n_samples)
    score += noise

    # Convert to probability via logistic function, then to binary label
    threshold = np.median(score)
    prob_pass = 1 / (1 + np.exp(-(score - threshold)))
    result_binary = (rng.random(n_samples) < prob_pass).astype(int)

    result = np.where(result_binary == 1, "PASS", "FAIL")

    df = pd.DataFrame(
        {
            "study_hours": np.round(study_hours, 1),
            "attendance": np.round(attendance, 1),
            "previous_scores": np.round(previous_scores, 1),
            "sleep_hours": np.round(sleep_hours, 1),
            "assignments_completed": assignments_completed,
            "class_participation": class_participation,
            "result": result,
        }
    )
    return df


def _sleep_benefit(hours: np.ndarray) -> np.ndarray:
    """Map sleep hours to a benefit score (inverted-U shape).

    Peak benefit around 7.5 hours; too little or too much sleep
    reduces the score.
    """
    return -((hours - 7.5) ** 2) / 5 + 2


def save_dataset(df: pd.DataFrame, path: str) -> str:
    """Persist a DataFrame to CSV, creating directories if needed.

    Returns the absolute path of the saved file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return os.path.abspath(path)

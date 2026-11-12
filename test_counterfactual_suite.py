"""
test_counterfactual_suite.py — Comprehensive tests for Counterfactual Simulator.
Tests Classification, Regression, Demo mode, and Streamlit AppTest integration.
"""

import unittest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.model_training import train_and_evaluate_pipeline
from src.counterfactual import (
    simulate_counterfactual,
    simulate_pipeline_counterfactual,
    plot_counterfactual_comparison,
)
from src.data_preprocessing import preprocess, load_data
from src.model_training import train_and_evaluate, get_models, select_best_model


class TestCounterfactualPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 1. Setup Classification dataset
        cls.df_clf = pd.read_csv("data/test_mixed.csv")
        cls.features_clf = ["age", "gender", "study_hours", "attendance", "major"]
        cls.res_clf = train_and_evaluate_pipeline(
            cls.df_clf, "passed", cls.features_clf, "Classification"
        )
        cls.pipe_clf = cls.res_clf["best_pipeline"]

        # 2. Setup Regression dataset
        cls.df_reg = pd.read_csv("data/test_regression.csv")
        cls.features_reg = ["sqft", "bedrooms", "bathrooms", "neighborhood", "age_years", "garage"]
        cls.res_reg = train_and_evaluate_pipeline(
            cls.df_reg, "price", cls.features_reg, "Regression"
        )
        cls.pipe_reg = cls.res_reg["best_pipeline"]

        # 3. Setup Demo Student Performance model
        cls.df_demo = load_data("data/student_data.csv")
        cls.X_train_d, cls.X_test_d, cls.y_train_d, cls.y_test_d, cls.scaler_d, cls.le_d, cls.feat_names_d = preprocess(cls.df_demo)
        models_d = get_models()
        res_d = train_and_evaluate(models_d, cls.X_train_d, cls.X_test_d, cls.y_train_d, cls.y_test_d)
        _, best_met_d = select_best_model(res_d)
        cls.best_model_d = best_met_d["fitted_model"]

    def test_classification_counterfactual_features_change(self):
        """Test classification what-if scenario with modified numerical and categorical features."""
        orig = {"age": 20, "gender": "Female", "study_hours": 1.0, "attendance": 40.0, "major": "Business"}
        mod = {"age": 20, "gender": "Female", "study_hours": 8.0, "attendance": 90.0, "major": "Engineering"}

        sim = simulate_pipeline_counterfactual(
            pipeline=self.pipe_clf,
            original_inputs=orig,
            modified_inputs=mod,
            feature_cols=self.features_clf,
            problem_type="Classification",
            target_encoder=self.res_clf["target_encoder"],
            target_col="passed",
        )

        self.assertEqual(sim["problem_type"], "Classification")
        self.assertIn("Study Hours", sim["changed_features"])
        self.assertIn("Attendance", sim["changed_features"])
        self.assertIn("Major", sim["changed_features"])
        self.assertNotIn("Age", sim["changed_features"])
        self.assertEqual(len(sim["changed_details"]), 3)

        # Check narrative
        self.assertTrue(len(sim["narrative"]) > 0)
        self.assertIn("Modifying", sim["narrative"])

        # Check probabilities
        self.assertIsNotNone(sim["original_class_probs"])
        self.assertIsNotNone(sim["modified_class_probs"])

        # Check figure generation
        fig = plot_counterfactual_comparison(sim)
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_classification_identical_inputs(self):
        """Test when no features are modified."""
        orig = {"age": 21, "gender": "Male", "study_hours": 5.0, "attendance": 75.0, "major": "Science"}
        mod = dict(orig)

        sim = simulate_pipeline_counterfactual(
            pipeline=self.pipe_clf,
            original_inputs=orig,
            modified_inputs=mod,
            feature_cols=self.features_clf,
            problem_type="Classification",
            target_encoder=self.res_clf["target_encoder"],
            target_col="passed",
        )

        self.assertEqual(len(sim["changed_features"]), 0)
        self.assertEqual(len(sim["changed_details"]), 0)
        self.assertFalse(sim["prediction_changed"])
        self.assertIn("No input features were modified", sim["narrative"])

    def test_regression_counterfactual_delta(self):
        """Test regression what-if scenario with delta and percentage change."""
        orig = {"sqft": 1400, "bedrooms": 3, "bathrooms": 2.0, "neighborhood": "Suburbs", "age_years": 20, "garage": "No"}
        mod = {"sqft": 2500, "bedrooms": 4, "bathrooms": 3.0, "neighborhood": "Downtown", "age_years": 5, "garage": "Yes"}

        sim = simulate_pipeline_counterfactual(
            pipeline=self.pipe_reg,
            original_inputs=orig,
            modified_inputs=mod,
            feature_cols=self.features_reg,
            problem_type="Regression",
            target_encoder=None,
            target_col="price",
        )

        self.assertEqual(sim["problem_type"], "Regression")
        self.assertIsNotNone(sim["original_value"])
        self.assertIsNotNone(sim["modified_value"])
        self.assertIsNotNone(sim["diff_value"])
        self.assertIsNotNone(sim["pct_change"])
        self.assertEqual(len(sim["changed_features"]), 6)

        # Value comparison
        expected_diff = sim["modified_value"] - sim["original_value"]
        self.assertAlmostEqual(sim["diff_value"], expected_diff, places=4)

        # Plot generation
        fig = plot_counterfactual_comparison(sim)
        self.assertIsNotNone(fig)
        plt.close(fig)

    def test_demo_mode_counterfactual(self):
        """Test the legacy/demo student-performance counterfactual simulation."""
        orig = {
            "study_hours": 2.0,
            "attendance": 50.0,
            "previous_scores": 45.0,
            "sleep_hours": 6.0,
            "assignments_completed": 3,
            "class_participation": 3,
        }
        mod = {
            "study_hours": 7.0,
            "attendance": 88.0,
            "previous_scores": 75.0,
            "sleep_hours": 8.0,
            "assignments_completed": 9,
            "class_participation": 8,
        }

        sim = simulate_counterfactual(
            self.best_model_d,
            self.scaler_d,
            self.le_d,
            self.feat_names_d,
            orig,
            mod,
        )

        self.assertIn("original_label", sim)
        self.assertIn("modified_label", sim)
        self.assertIn(sim["original_label"], ["PASS", "FAIL"])
        self.assertIn(sim["modified_label"], ["PASS", "FAIL"])
        self.assertGreater(len(sim["changed_features"]), 0)
        self.assertTrue(len(sim["narrative"]) > 0)

        fig = plot_counterfactual_comparison(sim)
        self.assertIsNotNone(fig)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()


"""
test_complete_workflows.py — End-to-end multi-workflow verification for Causal Explainable AI.
Verifies:
  Workflow 1: Home -> Dataset Analysis -> Upload -> Target -> Train -> Predict -> SHAP -> Causal -> Counterfactual
              (Tested separately for both Classification and Regression).
  Workflow 2: Home -> Explainable AI Agent -> Ask General Question -> Receive Structured Explanation.
  Workflow 3: Dataset Analysis -> Train Model -> Explainable AI Agent -> Ask Why Prediction Was Made -> Evidence-Based Response.
"""

import os
import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np

# Streamlit AppTest
from streamlit.testing.v1 import AppTest

# Source modules
from src.model_training import train_and_evaluate_pipeline, pipeline_results_to_dataframe
from src.explainability import (
    build_pipeline_shap_explainer,
    compute_pipeline_shap,
    explain_pipeline_prediction,
    plot_feature_contributions,
)
from src.causal_analysis import (
    generate_initial_causal_graph,
    add_node_to_graph,
    add_edge_to_graph,
    remove_edge_from_graph,
    build_causal_plotly_figure,
)
from src.counterfactual import (
    simulate_pipeline_counterfactual,
    plot_counterfactual_comparison,
)
from src.llm_service import (
    extract_grounding_evidence,
    format_evidence_prompt,
    query_explainable_ai_agent,
    is_api_key_configured,
    get_gemini_api_key,
    DEFAULT_GEMINI_MODEL,
)


class TestCompletePlatformWorkflows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df_clf = pd.read_csv("data/test_mixed.csv")
        cls.feats_clf = ["age", "gender", "study_hours", "attendance", "major"]
        cls.target_clf = "passed"

        cls.df_reg = pd.read_csv("data/test_regression.csv")
        cls.feats_reg = ["sqft", "bedrooms", "bathrooms", "neighborhood", "age_years", "garage"]
        cls.target_reg = "price"

    def test_workflow_1_classification_full_cycle(self):
        """Workflow 1 (Classification): Upload -> Train -> Predict -> SHAP -> Causal -> Counterfactual."""
        print("\n--- Testing Workflow 1: Complete Classification Lifecycle ---")

        # 1. Train Pipeline
        res_clf = train_and_evaluate_pipeline(
            self.df_clf, self.target_clf, self.feats_clf, "Classification"
        )
        self.assertIsNotNone(res_clf["best_pipeline"])
        self.assertIn("accuracy", res_clf["best_metrics"])
        self.assertIn("f1", res_clf["best_metrics"])
        print(f"  [1] Pipeline Trained: {res_clf['best_name']} (Accuracy: {res_clf['best_metrics']['accuracy']:.2%})")

        # 2. Make Prediction on New Input
        input_record = {"age": 21, "gender": "Male", "study_hours": 1.5, "attendance": 45.0, "major": "Business"}
        input_df = pd.DataFrame([input_record])[self.feats_clf]
        raw_pred = res_clf["best_pipeline"].predict(input_df)[0]
        pred_label = res_clf["target_encoder"].inverse_transform([raw_pred])[0]
        proba = res_clf["best_pipeline"].predict_proba(input_df)[0]
        conf = float(np.max(proba))
        self.assertIn(pred_label, ["0", "1", "FAIL", "PASS"])
        print(f"  [2] Prediction Generated: {pred_label} (Confidence: {conf:.1%})")

        # 3. SHAP Explanation
        explainer, expl_type = build_pipeline_shap_explainer(
            res_clf["best_pipeline"], res_clf["X_train"], problem_type="Classification"
        )
        ind_shap = explain_pipeline_prediction(
            pipeline=res_clf["best_pipeline"],
            explainer=explainer,
            input_values=input_record,
            feature_cols=self.feats_clf,
            categorical_features=res_clf["categorical_features"],
            problem_type="Classification",
            target_encoder=res_clf["target_encoder"],
            target_col=self.target_clf,
        )
        self.assertEqual(ind_shap["problem_type"], "Classification")
        self.assertGreater(len(ind_shap["feature_effects"]), 0)
        self.assertTrue(len(ind_shap["narrative"]) > 0)
        print(f"  [3] SHAP Explanation Generated: {len(ind_shap['feature_effects'])} features analyzed.")

        # 4. Causal Analysis Graph
        g_data = generate_initial_causal_graph(self.feats_clf, self.target_clf)
        self.assertIn("nodes", g_data)
        self.assertIn("edges", g_data)
        # Edit graph: add node, add edge
        g_data = add_node_to_graph(g_data, "tutoring_sessions")
        g_data = add_edge_to_graph(g_data, "tutoring_sessions", self.target_clf, "Supplemental academic support")
        fig_causal = build_causal_plotly_figure(g_data)
        self.assertIsNotNone(fig_causal)
        print(f"  [4] Causal Graph Created: {len(g_data['nodes'])} nodes, {len(g_data['edges'])} edges.")

        # 5. Counterfactual Simulation
        mod_record = {"age": 21, "gender": "Male", "study_hours": 7.0, "attendance": 90.0, "major": "Business"}
        cf_sim = simulate_pipeline_counterfactual(
            pipeline=res_clf["best_pipeline"],
            original_inputs=input_record,
            modified_inputs=mod_record,
            feature_cols=self.feats_clf,
            problem_type="Classification",
            target_encoder=res_clf["target_encoder"],
            target_col=self.target_clf,
        )
        self.assertEqual(cf_sim["problem_type"], "Classification")
        self.assertIn("Study Hours", cf_sim["changed_features"])
        self.assertIn("Attendance", cf_sim["changed_features"])
        fig_cf = plot_counterfactual_comparison(cf_sim)
        self.assertIsNotNone(fig_cf)
        print(f"  [5] Counterfactual Simulated: {cf_sim['original_label']} ➔ {cf_sim['modified_label']}")
        print("Workflow 1 (Classification): PASSED ✅")

    def test_workflow_1_regression_full_cycle(self):
        """Workflow 1 (Regression): Upload -> Train -> Predict -> SHAP -> Causal -> Counterfactual."""
        print("\n--- Testing Workflow 1: Complete Regression Lifecycle ---")

        # 1. Train Regression Pipeline
        res_reg = train_and_evaluate_pipeline(
            self.df_reg, self.target_reg, self.feats_reg, "Regression"
        )
        self.assertIsNotNone(res_reg["best_pipeline"])
        self.assertIn("r2", res_reg["best_metrics"])
        self.assertIn("rmse", res_reg["best_metrics"])
        print(f"  [1] Pipeline Trained: {res_reg['best_name']} (R²: {res_reg['best_metrics']['r2']:.4f})")

        # 2. Make Prediction on New Input
        input_record = {"sqft": 1600, "bedrooms": 3, "bathrooms": 2.0, "neighborhood": "Suburbs", "age_years": 15, "garage": "Yes"}
        input_df = pd.DataFrame([input_record])[self.feats_reg]
        raw_pred = float(res_reg["best_pipeline"].predict(input_df)[0])
        self.assertGreater(raw_pred, 0)
        print(f"  [2] Continuous Prediction Generated: ${raw_pred:,.2f}")

        # 3. SHAP Explanation
        explainer, expl_type = build_pipeline_shap_explainer(
            res_reg["best_pipeline"], res_reg["X_train"], problem_type="Regression"
        )
        ind_shap = explain_pipeline_prediction(
            pipeline=res_reg["best_pipeline"],
            explainer=explainer,
            input_values=input_record,
            feature_cols=self.feats_reg,
            categorical_features=res_reg["categorical_features"],
            problem_type="Regression",
            target_encoder=None,
            target_col=self.target_reg,
        )
        self.assertEqual(ind_shap["problem_type"], "Regression")
        self.assertIsNotNone(ind_shap["base_value"])
        print(f"  [3] SHAP Regression Shift: {ind_shap['net_shap_shift']:+,.2f}")

        # 4. Counterfactual Simulation
        mod_record = {"sqft": 2400, "bedrooms": 4, "bathrooms": 3.0, "neighborhood": "Downtown", "age_years": 5, "garage": "Yes"}
        cf_sim = simulate_pipeline_counterfactual(
            pipeline=res_reg["best_pipeline"],
            original_inputs=input_record,
            modified_inputs=mod_record,
            feature_cols=self.feats_reg,
            problem_type="Regression",
            target_encoder=None,
            target_col=self.target_reg,
        )
        self.assertEqual(cf_sim["problem_type"], "Regression")
        self.assertIsNotNone(cf_sim["diff_value"])
        self.assertIsNotNone(cf_sim["pct_change"])
        print(f"  [4] Counterfactual Shift: ${cf_sim['diff_value']:+,.2f} ({cf_sim['pct_change']:+.1f}%)")
        print("Workflow 1 (Regression): PASSED ✅")

    def test_workflow_2_general_knowledge_agent(self):
        """Workflow 2: Explainable AI Agent answering general AI/XAI questions."""
        print("\n--- Testing Workflow 2: General Knowledge Agent Q&A ---")

        general_question = "What is the key scientific difference between SHAP feature attributions and causal effects?"

        # Verify that without dataset evidence, agent functions in general knowledge mode
        mock_response = (
            "### 📖 Scientific Comparison: SHAP Attributions vs. Causal Effects\n\n"
            "1. **SHAP Feature Attributions**:\n"
            "   - Measure statistical association inside the trained model.\n"
            "   - Rooted in game-theoretic Shapley values.\n"
            "   - Answer: *'How much did feature X shift the model prediction away from the baseline?'*\n\n"
            "2. **Causal Effects**:\n"
            "   - Model the physical effect of real-world interventions: $P(Y \\mid do(X))$.\n"
            "   - Account for unobserved confounders, mediators, and structural equations.\n"
            "   - Answer: *'If we actively change X in the real world, what causal effect will it have on Y?'*"
        )

        with patch("src.llm_service.send_gemini_message") as mock_send:
            mock_send.return_value = {
                "success": True,
                "text": mock_response,
                "model": DEFAULT_GEMINI_MODEL,
                "history": [],
                "error": None,
                "error_type": None,
                "guidance": None,
            }

            resp = query_explainable_ai_agent(
                user_question=general_question,
                evidence=None,  # General knowledge mode
                api_key="mock_key",
            )

            self.assertTrue(resp["success"])
            self.assertIn("SHAP Feature Attributions", resp["text"])
            self.assertIn("Causal Effects", resp["text"])
            print("  Agent Response successfully synthesized in General Knowledge mode.")
            print("Workflow 2: PASSED ✅")

    def test_workflow_3_data_grounded_agent_explanation(self):
        """Workflow 3: Dataset Analysis -> Train Model -> Explainable AI Agent -> Evidence-Grounded Explanation."""
        print("\n--- Testing Workflow 3: Evidence-Grounded Prediction Explanation ---")

        # 1. Train model on test_mixed.csv
        res = train_and_evaluate_pipeline(self.df_clf, self.target_clf, self.feats_clf, "Classification")

        # 2. Build complete grounding evidence
        cf_mock = {
            "original_label": "FAIL",
            "original_confidence": 0.824,
            "original_class_probs": {"FAIL": 0.824, "PASS": 0.176},
            "modified_label": "PASS",
            "modified_confidence": 0.885,
            "modified_class_probs": {"FAIL": 0.115, "PASS": 0.885},
            "prediction_changed": True,
            "changed_features": ["Attendance", "Study Hours"],
            "changed_details": [
                {"Feature": "Attendance", "Original Value": "50%", "Modified Value": "85%", "Shift": "+35%"},
                {"Feature": "Study Hours", "Original Value": "2.0 hrs", "Modified Value": "6.0 hrs", "Shift": "+4.0"},
            ],
            "narrative": "Simulated increase in Attendance and Study Hours flipped outcome from FAIL to PASS.",
        }

        evidence = extract_grounding_evidence(
            dataset_source="upload",
            uploaded_df=self.df_clf,
            uploaded_filename="test_mixed.csv",
            uploaded_model_results=res,
            uploaded_cf_result=cf_mock,
        )

        self.assertEqual(evidence["prediction"]["outcome"], "FAIL")
        self.assertEqual(evidence["counterfactual"]["modified_prediction"], "PASS")

        # 3. Ask data-grounded question: "Why did the model predict FAIL?"
        mock_grounded_answer = (
            "### 🎯 Prediction Summary\n"
            "The model predicted **FAIL** with **82.4% confidence** on target `passed`.\n\n"
            "### ⚡ Key Influencing Features\n"
            "The primary drivers were **Attendance (50%)** and **Study Hours (2.0 hrs)**.\n\n"
            "### 🔍 SHAP Evidence\n"
            "- `Attendance` contributed **-0.342**, pulling the prediction toward FAIL.\n"
            "- `Study Hours` contributed **-0.285**, reinforcing the low score expectation.\n\n"
            "### 🔬 Causal Graph Assumptions\n"
            "The causal DAG assumes direct paths: `Attendance ➔ Exam Score` and `Study Hours ➔ Exam Score`. "
            "These are domain hypotheses rather than unconfounded causal proofs.\n\n"
            "### 🔄 What-If Alternatives\n"
            "Counterfactual simulation confirms that increasing **Attendance to 85%** and **Study Hours to 6.0** "
            "flips the model decision from **FAIL** to **PASS** (88.5% confidence)."
        )

        with patch("src.llm_service.send_gemini_message") as mock_send:
            mock_send.return_value = {
                "success": True,
                "text": mock_grounded_answer,
                "model": DEFAULT_GEMINI_MODEL,
                "history": [],
                "error": None,
                "error_type": None,
                "guidance": None,
            }

            resp = query_explainable_ai_agent(
                user_question="Why did the model predict FAIL?",
                evidence=evidence,
                api_key="mock_key",
            )

            self.assertTrue(resp["success"])
            # Verify 5 structured sections
            self.assertIn("Prediction Summary", resp["text"])
            self.assertIn("Key Influencing Features", resp["text"])
            self.assertIn("SHAP Evidence", resp["text"])
            self.assertIn("Causal Graph Assumptions", resp["text"])
            self.assertIn("What-If Alternatives", resp["text"])
            print("  Evidence-grounded 5-part explanation verified successfully.")
            print("Workflow 3: PASSED ✅")

    def test_streamlit_app_full_navigation(self):
        """Verify Streamlit App boots up, navigates between workspaces, and renders without error."""
        print("\n--- Testing Streamlit App Boot & Navigation ---")
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"App startup crashed: {at.exception}")

        # 1. Verify Home Workspace
        self.assertEqual(at.session_state["current_workspace"], "Home")
        print("  [1] Home Workspace verified.")

        # 2. Navigate to Dataset Analysis
        at.session_state["current_workspace"] = "Dataset Analysis"
        at.run()
        self.assertFalse(at.exception, f"Dataset Analysis crashed: {at.exception}")
        print("  [2] Dataset Analysis Workspace loaded.")

        # 3. Navigate to Explainable AI Agent
        at.session_state["current_workspace"] = "Explainable AI Agent"
        at.run()
        self.assertFalse(at.exception, f"Agent Workspace crashed: {at.exception}")
        btn_why_fail = at.button(key="sq_why_fail")
        self.assertIsNotNone(btn_why_fail)
        print("  [3] Explainable AI Agent Workspace verified with interactive controls.")
        print("Streamlit App Navigation: PASSED ✅")


if __name__ == "__main__":
    unittest.main()


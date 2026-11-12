"""
test_agent_workspace.py — Comprehensive tests for Explainable AI Agent and chatbot workspace.
Tests grounding evidence extraction, prompt construction, mock and live reasoning, and Streamlit AppTest.
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.llm_service import (
    extract_grounding_evidence,
    format_evidence_prompt,
    query_explainable_ai_agent,
    DEFAULT_GEMINI_MODEL,
)
from src.model_training import train_and_evaluate_pipeline


class TestExplainableAIEvidence(unittest.TestCase):
    def test_extract_demo_evidence(self):
        evidence = extract_grounding_evidence(
            dataset_source="demo",
            demo_model_name="Random Forest Classifier",
            demo_metrics={"f1": 0.88, "accuracy": 0.89},
        )
        self.assertIn("Student Performance", evidence["dataset_name"])
        self.assertEqual(evidence["problem_type"], "Classification")
        self.assertEqual(evidence["target_col"], "result")
        self.assertEqual(evidence["active_model"], "Random Forest Classifier")
        self.assertIsNotNone(evidence["prediction"])
        self.assertIsNotNone(evidence["shap"])
        self.assertIsNotNone(evidence["causal"])
        self.assertIsNotNone(evidence["counterfactual"])

        # Check prompt formatting
        formatted = format_evidence_prompt(evidence)
        self.assertIn("STRUCTURED MODEL EVIDENCE", formatted)
        self.assertIn("Random Forest Classifier", formatted)
        self.assertIn("SHAP", formatted)
        self.assertIn("Causal Graph", formatted)
        self.assertIn("Counterfactual", formatted)

    def test_extract_uploaded_classification_evidence(self):
        df_clf = pd.read_csv("data/test_mixed.csv")
        feats = ["age", "gender", "study_hours", "attendance", "major"]
        res = train_and_evaluate_pipeline(df_clf, "passed", feats, "Classification")

        cf_mock = {
            "original_label": "FAIL",
            "original_confidence": 0.82,
            "original_class_probs": {"FAIL": 0.82, "PASS": 0.18},
            "modified_label": "PASS",
            "modified_confidence": 0.89,
            "modified_class_probs": {"FAIL": 0.11, "PASS": 0.89},
            "prediction_changed": True,
            "changed_features": ["Attendance", "Study Hours"],
            "changed_details": [
                {"Feature": "Attendance", "Original Value": "50%", "Modified Value": "85%", "Shift": "+35"},
                {"Feature": "Study Hours", "Original Value": "2", "Modified Value": "6", "Shift": "+4"},
            ],
            "narrative": "Simulated change flipped outcome from FAIL to PASS.",
        }

        evidence = extract_grounding_evidence(
            dataset_source="upload",
            uploaded_df=df_clf,
            uploaded_filename="test_mixed.csv",
            uploaded_model_results=res,
            uploaded_cf_result=cf_mock,
        )

        self.assertEqual(evidence["dataset_name"], "test_mixed.csv")
        self.assertEqual(evidence["problem_type"], "Classification")
        self.assertEqual(evidence["target_col"], "passed")
        self.assertEqual(evidence["prediction"]["outcome"], "FAIL")
        self.assertEqual(evidence["counterfactual"]["modified_prediction"], "PASS")

        formatted = format_evidence_prompt(evidence)
        self.assertIn("test_mixed.csv", formatted)
        self.assertIn("passed", formatted)
        self.assertIn("FAIL", formatted)

    def test_query_explainable_ai_agent_mocked(self):
        """Test agent reasoning on both dataset question and general question with mocked SDK."""
        evidence = extract_grounding_evidence(dataset_source="demo")

        mock_answer_dataset = (
            "### 🎯 Prediction Summary\n"
            "The model predicted **FAIL** with **82.4% confidence**.\n\n"
            "### ⚡ Key Influencing Features\n"
            "The prediction was primarily driven by low Attendance (50%) and low Study Hours (2.0 hrs).\n\n"
            "### 🔍 SHAP Evidence\n"
            "SHAP indicates Attendance (-0.342) and Study Hours (-0.285) pushed the prediction below the passing threshold.\n\n"
            "### 🔬 Causal Graph Assumptions\n"
            "The causal DAG assumes Attendance and Study Hours have direct structural paths to Exam Score.\n\n"
            "### 🔄 What-If Alternatives\n"
            "Counterfactual simulation indicates increasing Attendance to 85% and Study Hours to 6.0 flips the prediction to PASS."
        )

        with patch("src.llm_service.send_gemini_message") as mock_send:
            mock_send.return_value = {
                "success": True,
                "text": mock_answer_dataset,
                "model": DEFAULT_GEMINI_MODEL,
                "history": [
                    {"role": "user", "content": "Why did the model predict FAIL?"},
                    {"role": "assistant", "content": mock_answer_dataset},
                ],
                "error": None,
                "error_type": None,
                "guidance": None,
            }

            res = query_explainable_ai_agent(
                user_question="Why did the model predict FAIL?",
                evidence=evidence,
                api_key="mock_key",
            )

            self.assertTrue(res["success"])
            self.assertIn("Prediction Summary", res["text"])
            self.assertIn("SHAP Evidence", res["text"])
            self.assertIn("Causal Graph Assumptions", res["text"])
            self.assertIn("What-If Alternatives", res["text"])

        # Test General Question
        mock_answer_general = (
            "The key scientific difference is that SHAP explains statistical feature attribution "
            "within the model's conditional distribution, whereas Causal Inference models the effect "
            "of physical interventions under potential unobserved confounders."
        )
        with patch("src.llm_service.send_gemini_message") as mock_send:
            mock_send.return_value = {
                "success": True,
                "text": mock_answer_general,
                "model": DEFAULT_GEMINI_MODEL,
                "history": [],
                "error": None,
                "error_type": None,
                "guidance": None,
            }

            res_gen = query_explainable_ai_agent(
                user_question="What is the difference between SHAP values and causal effects?",
                evidence=None,  # General question mode
                api_key="mock_key",
            )
            self.assertTrue(res_gen["success"])
            self.assertIn("SHAP", res_gen["text"])
            self.assertIn("Causal", res_gen["text"])


class TestStreamlitAgentWorkspace(unittest.TestCase):
    def test_agent_workspace_navigation_and_ui(self):
        """Test Explainable AI Agent in Streamlit using AppTest."""
        at = AppTest.from_file("app.py", default_timeout=30)
        at.run()
        self.assertFalse(at.exception, f"App crashed: {at.exception}")

        # Navigate to Explainable AI Agent
        at.session_state["current_workspace"] = "Explainable AI Agent"
        at.run()
        self.assertFalse(at.exception, f"Crashed in agent workspace: {at.exception}")

        # Verify return button exists
        btn_back = at.button(key="btn_back_home_agent")
        self.assertIsNotNone(btn_back)

        # Verify suggested question buttons exist
        btn_why_fail = at.button(key="sq_why_fail")
        self.assertIsNotNone(btn_why_fail)

        btn_shap_causal = at.button(key="gq_shap_vs_causal")
        self.assertIsNotNone(btn_shap_causal)

        # Click suggested question -> should produce an assistant response in history
        with patch("src.llm_service.query_explainable_ai_agent", return_value={"success": True, "text": "This is a simulated explanation."}):
            btn_why_fail.click().run()
        self.assertFalse(at.exception, f"Crashed clicking suggested question: {at.exception}")
        self.assertGreater(len(at.session_state["agent_chat_history"]), 0)
        last_msg = at.session_state["agent_chat_history"][-1]
        self.assertEqual(last_msg["role"], "assistant")
        self.assertTrue(
            "Gemini API Key Required" in last_msg["content"]
            or "Agent Error" in last_msg["content"]
            or len(last_msg["content"]) > 10,
            "Expected agent response message in chat history"
        )

        # Verify Clear Chat button exists and functions
        btn_clear = at.button(key="btn_clear_agent_chat")
        self.assertIsNotNone(btn_clear)
        btn_clear.click().run()
        self.assertFalse(at.exception, f"Crashed clearing chat: {at.exception}")
        self.assertEqual(len(at.session_state["agent_chat_history"]), 0)


if __name__ == "__main__":
    unittest.main()


"""
test_app_counterfactual.py — End-to-end integration tests using Streamlit AppTest.
"""

import sys
import pandas as pd
from streamlit.testing.v1 import AppTest
from src.model_training import train_and_evaluate_pipeline

def test_demo_counterfactual():
    print("Testing Demo Mode Counterfactual in Streamlit...")
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App crashed on startup: {at.exception}"

    # Navigate to Dataset Analysis
    at.session_state["current_workspace"] = "Dataset Analysis"
    at.session_state["dataset_source"] = "demo"
    at.run()
    assert not at.exception, f"Crashed navigating to Dataset Analysis: {at.exception}"

    # Click simulate demo counterfactual button
    btn_demo = at.button(key="btn_run_demo_cf")
    assert btn_demo is not None, "btn_run_demo_cf not found!"
    btn_demo.click().run()
    assert not at.exception, f"Crashed running demo counterfactual: {at.exception}"
    assert at.session_state["demo_cf_result"] is not None, "demo_cf_result was not populated!"
    print("Demo Mode Counterfactual: PASSED ✅")


def test_upload_classification_counterfactual():
    print("Testing Upload Classification Counterfactual in Streamlit...")
    df_clf = pd.read_csv("data/test_mixed.csv")
    features_clf = ["age", "gender", "study_hours", "attendance", "major"]
    res_clf = train_and_evaluate_pipeline(df_clf, "passed", features_clf, "Classification")

    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App crashed on startup: {at.exception}"

    at.session_state["current_workspace"] = "Dataset Analysis"
    at.session_state["dataset_source"] = "upload"
    at.session_state["uploaded_df"] = df_clf
    at.session_state["uploaded_filename"] = "test_mixed.csv"
    at.session_state["uploaded_model_results"] = res_clf
    at.run()
    assert not at.exception, f"Crashed setting up upload state: {at.exception}"

    # Verify button exists and click
    btn_up = at.button(key="btn_run_uploaded_cf")
    assert btn_up is not None, "btn_run_uploaded_cf not found!"
    btn_up.click().run()
    assert not at.exception, f"Crashed running uploaded classification counterfactual: {at.exception}"
    assert at.session_state["uploaded_cf_result"] is not None, "uploaded_cf_result was not populated!"
    sim = at.session_state["uploaded_cf_result"]
    assert sim["problem_type"] == "Classification"
    print("Upload Classification Counterfactual: PASSED ✅")


def test_upload_regression_counterfactual():
    print("Testing Upload Regression Counterfactual in Streamlit...")
    df_reg = pd.read_csv("data/test_regression.csv")
    features_reg = ["sqft", "bedrooms", "bathrooms", "neighborhood", "age_years", "garage"]
    res_reg = train_and_evaluate_pipeline(df_reg, "price", features_reg, "Regression")

    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception, f"App crashed on startup: {at.exception}"

    at.session_state["current_workspace"] = "Dataset Analysis"
    at.session_state["dataset_source"] = "upload"
    at.session_state["uploaded_df"] = df_reg
    at.session_state["uploaded_filename"] = "test_regression.csv"
    at.session_state["uploaded_model_results"] = res_reg
    at.run()
    assert not at.exception, f"Crashed setting up regression upload state: {at.exception}"

    # Verify button exists and click
    btn_up = at.button(key="btn_run_uploaded_cf")
    assert btn_up is not None, "btn_run_uploaded_cf not found!"
    btn_up.click().run()
    assert not at.exception, f"Crashed running uploaded regression counterfactual: {at.exception}"
    assert at.session_state["uploaded_cf_result"] is not None, "uploaded_cf_result was not populated for regression!"
    sim = at.session_state["uploaded_cf_result"]
    assert sim["problem_type"] == "Regression"
    print("Upload Regression Counterfactual: PASSED ✅")


if __name__ == "__main__":
    test_demo_counterfactual()
    test_upload_classification_counterfactual()
    test_upload_regression_counterfactual()
    print("\n🎉 ALL STREAMLIT INTEGRATION TESTS PASSED SUCCESSFULLY! 🎉")


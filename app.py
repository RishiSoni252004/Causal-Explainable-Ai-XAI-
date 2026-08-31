"""
app.py — Streamlit dashboard for the Causal XAI project.

Displays:
    • Home Page (Workflow explanation)
    • Dataset Analysis
    • Machine Learning Model
    • Explainable AI
    • Causal Analysis
    • Counterfactual Simulator
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.utils import generate_student_data, save_dataset
from src.data_preprocessing import load_data, preprocess
from src.model_training import (
    get_models,
    train_and_evaluate,
    results_to_dataframe,
    select_best_model,
    save_model,
)
from src.explainability import (
    get_global_feature_importance,
    plot_global_importance,
    compute_shap_values,
    plot_shap_summary,
    get_shap_importance,
    explain_individual,
)
from src.causal_analysis import (
    build_causal_graph,
    plot_causal_graph,
    get_edge_descriptions,
    CORRELATION_VS_CAUSATION,
    TRUSTWORTHY_AI_CONTENT,
)
from src.counterfactual import simulate_counterfactual

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Causal XAI — Student Performance",
    page_icon="🎓",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        padding: 1.2rem 0 0.2rem;
    }
    .main-title h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 800;
    }
    .main-title p {
        color: #888;
        font-size: 1.05rem;
        margin-top: -0.4rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
    }
    .metric-card h3 {
        margin: 0;
        color: #444;
        font-size: 0.9rem;
    }
    .metric-card p {
        margin: 0.3rem 0 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: #333;
    }
    .workflow-arrow {
        text-align: center;
        font-size: 2rem;
        color: #667eea;
        margin: 10px 0;
    }
    .workflow-box {
        background: #1e1e1e;
        border: 2px solid #333;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        margin: 0 auto;
        width: 60%;
        color: #fff;
    }
    .workflow-box h3 {
        margin-top: 0;
        color: #667eea;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Dataset Analysis",
        "Machine Learning Model",
        "Explainable AI",
        "Causal Analysis",
        "Counterfactual Simulator"
    ]
)

# ---------------------------------------------------------------------------
# Data Loading (Shared)
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join("data", "student_data.csv")
MODELS_DIR = "models"

@st.cache_data
def load_and_prep_data():
    if not os.path.exists(DATA_PATH):
        df = generate_student_data(n_samples=1000)
        save_dataset(df, DATA_PATH)
    else:
        df = load_data(DATA_PATH)
    return df

df = load_and_prep_data()
X_train, X_test, y_train, y_test, scaler, le, feature_names = preprocess(df)

# Train best model (cached)
@st.cache_resource
def train_best_model(_X_train, _X_test, _y_train, _y_test):
    models = get_models()
    results = train_and_evaluate(models, _X_train, _X_test, _y_train, _y_test)
    best_name, best_metrics = select_best_model(results)
    save_model(best_metrics["fitted_model"], scaler, le, feature_names, directory=MODELS_DIR)
    return best_name, best_metrics, results

best_name, best_metrics, results = train_best_model(X_train, X_test, y_train, y_test)
best_model = best_metrics["fitted_model"]


# ---------------------------------------------------------------------------
# Page: Home
# ---------------------------------------------------------------------------
if page == "Home":
    st.markdown(
        """
        <div class="main-title">
            <h1>🎓 Causal Explainable AI for Trustworthy Intelligent Systems</h1>
            <p>Bridging the gap between predictive power and causal understanding</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("<h2 style='text-align: center;'>Complete Project Workflow</h2>", unsafe_allow_html=True)
    
    workflow_steps = [
        ("📊 Dataset", "Synthetic generation of student data with assumed causal relationships."),
        ("🤖 Machine Learning Model", "Training baseline predictive models (Logistic Regression & Random Forest)."),
        ("🔍 Explainable AI", "Using SHAP to understand which features drive the model's predictions overall and for individuals."),
        ("🔬 Causal Analysis", "Defining a causal graph and differentiating correlation from causation."),
        ("🔄 Counterfactual Simulation", "Testing 'what-if' scenarios to see how changing inputs affects predictions."),
        ("🛡️ More Trustworthy AI", "Combining predictive accuracy with explainability and causal reasoning for safer deployment.")
    ]

    for title, desc in workflow_steps:
        st.markdown(f"""
        <div class="workflow-box">
            <h3>{title}</h3>
            <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)
        if title != workflow_steps[-1][0]:
            st.markdown("<div class='workflow-arrow'>↓</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Dataset Analysis
# ---------------------------------------------------------------------------
elif page == "Dataset Analysis":
    st.header("📊 Dataset Overview")

    col_info1, col_info2, col_info3, col_info4 = st.columns(4)
    with col_info1:
        st.metric("Total Records", f"{len(df):,}")
    with col_info2:
        st.metric("Features", len(df.columns) - 1)
    with col_info3:
        pass_count = (df["result"] == "PASS").sum()
        st.metric("PASS", f"{pass_count}  ({pass_count/len(df)*100:.1f}%)")
    with col_info4:
        fail_count = (df["result"] == "FAIL").sum()
        st.metric("FAIL", f"{fail_count}  ({fail_count/len(df)*100:.1f}%)")

    tab_preview, tab_stats = st.tabs(["🔍 Preview", "📈 Statistics"])
    with tab_preview:
        st.dataframe(df.head(20), use_container_width=True)
    with tab_stats:
        st.dataframe(df.describe().round(2), use_container_width=True)

    st.divider()

    st.header("📉 Feature Distributions")
    feature_cols = [c for c in df.columns if c != "result"]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.patch.set_facecolor("#0e1117")

    for ax, col in zip(axes.ravel(), feature_cols):
        ax.set_facecolor("#0e1117")
        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.tick_params(colors="#ccc")
        ax.set_title(col.replace("_", " ").title(), color="#ccc", fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("")

        pass_data = df.loc[df["result"] == "PASS", col]
        fail_data = df.loc[df["result"] == "FAIL", col]
        ax.hist(pass_data, bins=25, alpha=0.65, label="PASS", color="#667eea")
        ax.hist(fail_data, bins=25, alpha=0.65, label="FAIL", color="#e74c6f")
        ax.legend(fontsize=8, facecolor="#0e1117", edgecolor="#333", labelcolor="#ccc")

    plt.tight_layout()
    st.pyplot(fig)


# ---------------------------------------------------------------------------
# Page: Machine Learning Model
# ---------------------------------------------------------------------------
elif page == "Machine Learning Model":
    st.header("🤖 Model Training & Comparison")

    comparison_df = results_to_dataframe(results)
    st.dataframe(
        comparison_df.style.highlight_max(
            subset=["Accuracy", "Precision", "Recall", "F1 Score"],
            color="#667eea",
        ),
        use_container_width=True,
    )

    st.divider()

    st.header(f"🏆 Best Model — {best_name}")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Accuracy",  f"{best_metrics['accuracy']:.2%}")
    col_b.metric("Precision", f"{best_metrics['precision']:.2%}")
    col_c.metric("Recall",    f"{best_metrics['recall']:.2%}")
    col_d.metric("F1 Score",  f"{best_metrics['f1']:.2%}")

    st.subheader("Confusion Matrix")
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    fig_cm.patch.set_facecolor("#0e1117")
    ax_cm.set_facecolor("#0e1117")

    sns.heatmap(
        best_metrics["confusion_matrix"],
        annot=True,
        fmt="d",
        cmap="RdPu",
        xticklabels=le.classes_,
        yticklabels=le.classes_,
        ax=ax_cm,
        cbar_kws={"shrink": 0.8},
    )
    ax_cm.set_xlabel("Predicted", color="#ccc")
    ax_cm.set_ylabel("Actual", color="#ccc")
    ax_cm.tick_params(colors="#ccc")
    st.pyplot(fig_cm)


# ---------------------------------------------------------------------------
# Page: Explainable AI
# ---------------------------------------------------------------------------
elif page == "Explainable AI":
    st.markdown(
        """
        <div style="text-align:center; padding:0.5rem 0;">
            <h1 style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 2rem; font-weight: 800;">🔍 Explainable AI</h1>
            <p style="color:#888; font-size:1rem; margin-top:-0.4rem;">
                Understand <em>why</em> the model makes its predictions
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.header("📊 Global Feature Importance")
    st.caption("Derived from the model's own coefficients / feature importances.")

    importance_df = get_global_feature_importance(best_model, feature_names)

    col_chart, col_table = st.columns([2, 1])
    with col_chart:
        fig_imp = plot_global_importance(importance_df)
        st.pyplot(fig_imp)
    with col_table:
        st.dataframe(importance_df, use_container_width=True, hide_index=True)

    st.divider()

    st.header("🧠 SHAP Feature Importance")
    st.caption(
        "SHAP (SHapley Additive exPlanations) measures each feature's "
        "contribution to every individual prediction."
    )

    with st.spinner("Computing SHAP values … this may take a moment."):
        shap_explanation, shap_explainer = compute_shap_values(
            best_model, X_test, feature_names
        )

    tab_summary, tab_importance = st.tabs(["🐝 SHAP Summary Plot", "📋 Mean |SHAP| Table"])
    with tab_summary:
        fig_shap = plot_shap_summary(shap_explanation)
        st.pyplot(fig_shap)
    with tab_importance:
        shap_imp_df = get_shap_importance(shap_explanation, feature_names)
        st.dataframe(shap_imp_df, use_container_width=True, hide_index=True)

    st.divider()

    st.header("🎯 Individual Prediction Explanation")
    st.caption("Enter a student's data to see the prediction and its explanation.")

    with st.form("prediction_form"):
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            inp_study = st.slider("Study Hours", 0.0, 12.0, 5.0, 0.5)
        with r1c2:
            inp_attend = st.slider("Attendance (%)", 0.0, 100.0, 75.0, 1.0)
        with r1c3:
            inp_prev = st.slider("Previous Scores", 0.0, 100.0, 60.0, 1.0)

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            inp_sleep = st.slider("Sleep Hours", 3.0, 12.0, 7.0, 0.5)
        with r2c2:
            inp_assign = st.slider("Assignments Completed", 0, 10, 5)
        with r2c3:
            inp_partic = st.slider("Class Participation", 0, 10, 5)

        submitted = st.form_submit_button("🔮 Predict & Explain", use_container_width=True)

    if submitted:
        user_input = {
            "study_hours": inp_study,
            "attendance": inp_attend,
            "previous_scores": inp_prev,
            "sleep_hours": inp_sleep,
            "assignments_completed": inp_assign,
            "class_participation": inp_partic,
        }

        with st.spinner("Generating explanation …"):
            result = explain_individual(
                best_model, shap_explainer, scaler, le, feature_names, user_input
            )

        st.markdown("---")
        pred_col, prob_col = st.columns(2)
        pred_color = "#667eea" if result["prediction_label"] == "PASS" else "#e74c6f"
        with pred_col:
            st.markdown(
                f"<div style='text-align:center; padding:1rem; "
                f"border-radius:12px; border:2px solid {pred_color};'>"
                f"<h3 style='margin:0; color:#ccc;'>Prediction</h3>"
                f"<p style='font-size:2.2rem; font-weight:800; margin:0.3rem 0 0; "
                f"color:{pred_color};'>{result['prediction_label']}</p></div>",
                unsafe_allow_html=True,
            )
        with prob_col:
            st.markdown(
                f"<div style='text-align:center; padding:1rem; "
                f"border-radius:12px; border:2px solid {pred_color};'>"
                f"<h3 style='margin:0; color:#ccc;'>Confidence</h3>"
                f"<p style='font-size:2.2rem; font-weight:800; margin:0.3rem 0 0; "
                f"color:{pred_color};'>{result['prediction_proba']:.1%}</p></div>",
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.info(result["narrative"])

        st.subheader("Top Contributing Factors")
        top_display = result["top_factors"][["Feature", "Value", "SHAP"]].copy()
        top_display["Direction"] = top_display["SHAP"].apply(
            lambda v: "⬆️ Pushes toward PASS" if v > 0 else "⬇️ Pushes toward FAIL"
        )
        top_display["SHAP"] = top_display["SHAP"].round(4)
        st.dataframe(top_display, use_container_width=True, hide_index=True)

        with st.expander("📋 Full Feature Breakdown"):
            full = result["feature_effects"][["Feature", "Value", "SHAP"]].copy()
            full["SHAP"] = full["SHAP"].round(4)
            st.dataframe(full, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Page: Causal Analysis
# ---------------------------------------------------------------------------
elif page == "Causal Analysis":
    st.markdown(
        """
        <div style="text-align:center; padding:0.5rem 0;">
            <h1 style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 2rem; font-weight: 800;">🔬 Causal Analysis</h1>
            <p style="color:#888; font-size:1rem; margin-top:-0.4rem;">
                Explore the <em>causal structure</em> behind student performance
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.header("🕸️ Causal Graph")
    st.warning(
        "⚠️ **Important disclaimer:** This causal graph represents *assumed* causal "
        "relationships based on domain knowledge and common sense. It is **not** proof "
        "of real-world causality. Establishing true causation requires controlled "
        "experiments or advanced causal inference methods."
    )

    causal_graph = build_causal_graph()
    fig_cg = plot_causal_graph(causal_graph)
    st.pyplot(fig_cg)

    st.divider()

    st.header("📝 Causal Relationship Descriptions")
    st.caption("Each arrow in the graph above represents one of the following assumed relationships.")
    
    edge_rows = get_edge_descriptions()
    edge_df = pd.DataFrame(edge_rows)
    st.dataframe(edge_df, use_container_width=True, hide_index=True)

    st.divider()

    st.header("📐 " + CORRELATION_VS_CAUSATION["title"])
    st.markdown(CORRELATION_VS_CAUSATION["content"])

    st.divider()

    st.header("🛡️ " + TRUSTWORTHY_AI_CONTENT["title"])
    st.markdown(TRUSTWORTHY_AI_CONTENT["content"])


# ---------------------------------------------------------------------------
# Page: Counterfactual Simulator
# ---------------------------------------------------------------------------
elif page == "Counterfactual Simulator":
    st.markdown(
        """
        <div style="text-align:center; padding:0.5rem 0;">
            <h1 style="background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
                       -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                       font-size: 2rem; font-weight: 800;">🔄 Counterfactual Simulator</h1>
            <p style="color:#888; font-size:1rem; margin-top:-0.4rem;">
                Test <em>what-if</em> scenarios to see how predictions change
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    
    st.warning(
        "⚠️ **Important Scientific Disclaimer:** Counterfactual simulations in this version "
        "show how changes in input variables affect the machine learning model's prediction. "
        "They do **not** independently prove real-world causality."
    )

    st.markdown("### 1. Set Baseline (Original)")
    col1, col2, col3 = st.columns(3)
    with col1:
        orig_study = st.number_input("Original Study Hours", min_value=0.0, max_value=12.0, value=2.0)
        orig_sleep = st.number_input("Original Sleep Hours", min_value=3.0, max_value=12.0, value=6.0)
    with col2:
        orig_attend = st.number_input("Original Attendance (%)", min_value=0.0, max_value=100.0, value=55.0)
        orig_assign = st.number_input("Original Assignments Completed", min_value=0, max_value=10, value=3)
    with col3:
        orig_prev = st.number_input("Original Previous Scores", min_value=0.0, max_value=100.0, value=45.0)
        orig_partic = st.number_input("Original Class Participation", min_value=0, max_value=10, value=3)

    st.divider()

    st.markdown("### 2. Set Counterfactual (Modified)")
    col4, col5, col6 = st.columns(3)
    with col4:
        mod_study = st.number_input("Modified Study Hours", min_value=0.0, max_value=12.0, value=6.0)
        mod_sleep = st.number_input("Modified Sleep Hours", min_value=3.0, max_value=12.0, value=8.0)
    with col5:
        mod_attend = st.number_input("Modified Attendance (%)", min_value=0.0, max_value=100.0, value=80.0)
        mod_assign = st.number_input("Modified Assignments Completed", min_value=0, max_value=10, value=8)
    with col6:
        mod_prev = st.number_input("Modified Previous Scores", min_value=0.0, max_value=100.0, value=70.0)
        mod_partic = st.number_input("Modified Class Participation", min_value=0, max_value=10, value=8)

    if st.button("🔄 Simulate Counterfactual", use_container_width=True):
        orig_inputs = {
            "study_hours": orig_study,
            "attendance": orig_attend,
            "previous_scores": orig_prev,
            "sleep_hours": orig_sleep,
            "assignments_completed": orig_assign,
            "class_participation": orig_partic,
        }
        mod_inputs = {
            "study_hours": mod_study,
            "attendance": mod_attend,
            "previous_scores": mod_prev,
            "sleep_hours": mod_sleep,
            "assignments_completed": mod_assign,
            "class_participation": mod_partic,
        }

        with st.spinner("Simulating..."):
            sim_result = simulate_counterfactual(
                best_model, scaler, le, feature_names, orig_inputs, mod_inputs
            )
        
        st.markdown("---")
        st.subheader("Comparison Result")

        r1, r2 = st.columns(2)
        
        orig_color = "#667eea" if sim_result["original_label"] == "PASS" else "#e74c6f"
        mod_color = "#667eea" if sim_result["modified_label"] == "PASS" else "#e74c6f"
        
        with r1:
            st.markdown(
                f"<div style='text-align:center; padding:1rem; border-radius:12px; border:2px solid #555;'>"
                f"<h4 style='margin:0; color:#ccc;'>Original Prediction</h4>"
                f"<p style='font-size:2rem; font-weight:800; margin:0.3rem 0; color:{orig_color};'>{sim_result['original_label']}</p>"
                f"<p style='margin:0; color:#aaa;'>Confidence: {sim_result['original_confidence']:.1%}</p></div>",
                unsafe_allow_html=True,
            )
            
        with r2:
            st.markdown(
                f"<div style='text-align:center; padding:1rem; border-radius:12px; border:2px dashed {mod_color};'>"
                f"<h4 style='margin:0; color:#ccc;'>Modified Prediction</h4>"
                f"<p style='font-size:2rem; font-weight:800; margin:0.3rem 0; color:{mod_color};'>{sim_result['modified_label']}</p>"
                f"<p style='margin:0; color:#aaa;'>Confidence: {sim_result['modified_confidence']:.1%}</p></div>",
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.info(sim_result["narrative"])

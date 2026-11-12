"""
app.py — Modern Platform Dashboard for Causal Explainable AI for Trustworthy Intelligent Systems.

Workspaces:
    • Home (Platform Overview & Workspace Switcher)
    • Dataset Analysis (EDA, ML Models, SHAP Explainability, Causal Analysis, Counterfactuals)
    • Explainable AI Agent (Evidence-grounded Conversational AI Assistant)
"""

import sys
import os
import importlib

# Ensure application directory is at the front of sys.path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import networkx as nx

from src.utils import generate_student_data, save_dataset
from src.data_preprocessing import load_data, preprocess

try:
    from src.model_training import (
        get_models,
        train_and_evaluate,
        results_to_dataframe,
        select_best_model,
        save_model,
        get_classification_models,
        get_regression_models,
        build_column_transformer,
        train_and_evaluate_pipeline,
        pipeline_results_to_dataframe,
    )
except ImportError:
    if "src.model_training" in sys.modules:
        importlib.reload(sys.modules["src.model_training"])
    from src.model_training import (
        get_models,
        train_and_evaluate,
        results_to_dataframe,
        select_best_model,
        save_model,
        get_classification_models,
        get_regression_models,
        build_column_transformer,
        train_and_evaluate_pipeline,
        pipeline_results_to_dataframe,
    )

try:
    from src.explainability import (
        clean_feature_name,
        get_global_feature_importance,
        get_pipeline_global_importance,
        plot_global_importance,
        compute_shap_values,
        build_pipeline_shap_explainer,
        compute_pipeline_shap,
        plot_shap_summary,
        get_shap_importance,
        get_pipeline_shap_importance,
        explain_individual,
        explain_pipeline_prediction,
        plot_feature_contributions,
        plot_shap_waterfall,
    )
except ImportError:
    if "src.explainability" in sys.modules:
        importlib.reload(sys.modules["src.explainability"])
    from src.explainability import (
        clean_feature_name,
        get_global_feature_importance,
        get_pipeline_global_importance,
        plot_global_importance,
        compute_shap_values,
        build_pipeline_shap_explainer,
        compute_pipeline_shap,
        plot_shap_summary,
        get_shap_importance,
        get_pipeline_shap_importance,
        explain_individual,
        explain_pipeline_prediction,
        plot_feature_contributions,
        plot_shap_waterfall,
    )

try:
    from src.causal_analysis import (
        build_causal_graph,
        plot_causal_graph,
        get_edge_descriptions,
        CORRELATION_VS_CAUSATION,
        TRUSTWORTHY_AI_CONTENT,
        SCIENTIFIC_DISCLAIMER_CONTENT,
        generate_initial_causal_graph,
        graph_to_dict,
        dict_to_graph,
        add_node_to_graph,
        remove_node_from_graph,
        add_edge_to_graph,
        remove_edge_from_graph,
        update_edge_in_graph,
        build_causal_plotly_figure,
        get_graph_edge_descriptions,
    )
except ImportError:
    if "src.causal_analysis" in sys.modules:
        importlib.reload(sys.modules["src.causal_analysis"])
    from src.causal_analysis import (
        build_causal_graph,
        plot_causal_graph,
        get_edge_descriptions,
        CORRELATION_VS_CAUSATION,
        TRUSTWORTHY_AI_CONTENT,
        SCIENTIFIC_DISCLAIMER_CONTENT,
        generate_initial_causal_graph,
        graph_to_dict,
        dict_to_graph,
        add_node_to_graph,
        remove_node_from_graph,
        add_edge_to_graph,
        remove_edge_from_graph,
        update_edge_in_graph,
        build_causal_plotly_figure,
        get_graph_edge_descriptions,
    )

try:
    from src.counterfactual import (
        simulate_counterfactual,
        simulate_pipeline_counterfactual,
        plot_counterfactual_comparison,
    )
except ImportError:
    if "src.counterfactual" in sys.modules:
        importlib.reload(sys.modules["src.counterfactual"])
    from src.counterfactual import (
        simulate_counterfactual,
        simulate_pipeline_counterfactual,
        plot_counterfactual_comparison,
    )

try:
    from src.llm_service import (
        query_explainable_ai_agent,
        extract_grounding_evidence,
        is_api_key_configured,
        get_gemini_api_key,
        DEFAULT_GEMINI_MODEL,
        SUPPORTED_FREE_MODELS,
    )
except ImportError:
    if "src.llm_service" in sys.modules:
        importlib.reload(sys.modules["src.llm_service"])
    from src.llm_service import (
        query_explainable_ai_agent,
        extract_grounding_evidence,
        is_api_key_configured,
        get_gemini_api_key,
        DEFAULT_GEMINI_MODEL,
        SUPPORTED_FREE_MODELS,
    )

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Causal XAI — Trustworthy Intelligent Systems",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS for Modern AI Platform Interface
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 2.2rem 1rem 1.5rem;
        max-width: 950px;
        margin: 0 auto;
    }
    .platform-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #818cf8;
        padding: 5px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        line-height: 1.2;
        letter-spacing: -0.03em;
        margin: 0 0 0.8rem;
        background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-title-accent {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.1rem;
        line-height: 1.6;
        margin: 0 auto 1.5rem;
        max-width: 780px;
    }

    /* Badges Bar */
    .status-badge-bar {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 12px;
        margin-bottom: 1.5rem;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 9999px;
        padding: 4px 12px;
        font-size: 0.78rem;
        color: #cbd5e1;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 8px #10b981;
    }

    /* Clickable Platform Cards */
    .platform-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.85) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 18px;
        padding: 2rem 1.8rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.35);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        margin-bottom: 0.75rem;
        height: 380px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    .platform-card::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        opacity: 0.85;
    }
    .agent-card-theme::before {
        background: linear-gradient(90deg, #ec4899, #8b5cf6);
    }
    .card-top-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1.2rem;
    }
    .card-icon-box {
        width: 52px;
        height: 52px;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.8rem;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.3);
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.2);
    }
    .agent-icon-box {
        background: rgba(236, 72, 153, 0.15);
        border: 1px solid rgba(236, 72, 153, 0.3);
        box-shadow: 0 4px 14px rgba(236, 72, 153, 0.2);
    }
    .card-category-badge {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 4px 10px;
        border-radius: 9999px;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #94a3b8;
    }
    .card-title-text {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 0.75rem 0;
        color: #f8fafc;
    }
    .card-desc-text {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.55;
        margin: 0 0 1.2rem 0;
    }
    .card-chips-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
    }
    .feature-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 8px;
        padding: 4px 9px;
        font-size: 0.76rem;
        color: #cbd5e1;
        font-weight: 500;
    }

    /* Workspace Header Bar */
    .workspace-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 1.2rem 1.6rem;
        margin-bottom: 1.6rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .workspace-banner h1 {
        font-size: 1.7rem;
        font-weight: 800;
        color: #f8fafc;
        margin: 0;
    }
    .workspace-banner p {
        color: #94a3b8;
        font-size: 0.95rem;
        margin: 0.25rem 0 0;
    }

    /* Feature Highlight Grid */
    .pillar-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: left;
        height: 100%;
    }
    .pillar-card h4 {
        margin: 0.4rem 0 0.3rem;
        color: #f8fafc;
        font-size: 1.02rem;
    }
    .pillar-card p {
        margin: 0;
        color: #94a3b8;
        font-size: 0.86rem;
        line-height: 1.45;
    }

    /* Dataset Upload Zone */
    .upload-zone {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.5), rgba(15, 23, 42, 0.7));
        border: 2px dashed rgba(99, 102, 241, 0.4);
        border-radius: 16px;
        padding: 2rem 1.5rem;
        text-align: center;
        margin-bottom: 1rem;
        transition: border-color 0.2s ease;
    }
    .upload-zone:hover {
        border-color: rgba(99, 102, 241, 0.75);
    }
    .upload-zone h3 {
        color: #f8fafc;
        font-size: 1.1rem;
        font-weight: 700;
        margin: 0.5rem 0 0.3rem;
    }
    .upload-zone p {
        color: #94a3b8;
        font-size: 0.87rem;
        margin: 0;
    }
    /* Source Selector Buttons */
    .source-btn-active {
        background: rgba(99, 102, 241, 0.2);
        border: 2px solid #6366f1;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: #c7d2fe;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .source-btn-inactive {
        background: rgba(30, 41, 59, 0.5);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        color: #94a3b8;
        font-weight: 500;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    /* Profile stat card */
    .prof-stat {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        text-align: center;
        height: 100%;
    }
    .prof-stat .label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.4rem;
    }
    .prof-stat .value {
        color: #f8fafc;
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .prof-stat .sub {
        color: #64748b;
        font-size: 0.75rem;
        margin-top: 0.1rem;
    }
    /* Quality indicator bar */
    .quality-bar-wrap {
        height: 7px;
        background: #334155;
        border-radius: 9999px;
        overflow: hidden;
        margin-top: 4px;
    }
    .quality-bar-fill {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.4s ease;
    }

    /* ── Dataset Inspection Styles ───────────────────────────────────────── */
    .inspect-section-header {
        font-size: 1.0rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.01em;
        margin: 1.4rem 0 0.6rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .col-type-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 9px;
        border-radius: 9999px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .badge-numeric  { background: rgba(99, 102, 241, 0.18); color: #818cf8; border: 1px solid rgba(99, 102, 241, 0.35); }
    .badge-category { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.35); }
    .badge-id       { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.35); }
    .badge-bool     { background: rgba(236, 72, 153, 0.15); color: #f472b6; border: 1px solid rgba(236, 72, 153, 0.35); }
    .badge-datetime { background: rgba(14, 165, 233, 0.15); color: #38bdf8; border: 1px solid rgba(14, 165, 233, 0.35); }
    .col-row {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 7px 12px;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        margin-bottom: 5px;
    }
    .col-row-name { color: #f8fafc; font-size: 0.88rem; font-weight: 600; flex: 1; }
    .col-row-meta { color: #64748b; font-size: 0.78rem; }
    .suggestion-box {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
        color: #c7d2fe;
    }
    .warn-box {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
        color: #fde68a;
    }
    .ok-box {
        background: rgba(16, 185, 129, 0.07);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
        color: #6ee7b7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Navigation State Management
# ---------------------------------------------------------------------------
WORKSPACE_HOME = "Home"
WORKSPACE_DATASET = "Dataset Analysis"
WORKSPACE_AGENT = "Explainable AI Agent"

WORKSPACES = [WORKSPACE_HOME, WORKSPACE_DATASET, WORKSPACE_AGENT]

if "current_workspace" not in st.session_state:
    st.session_state.current_workspace = WORKSPACE_HOME

# Dataset source state: "demo" uses built-in student data, "upload" uses user CSV
if "dataset_source" not in st.session_state:
    st.session_state.dataset_source = "demo"
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "uploaded_filename" not in st.session_state:
    st.session_state.uploaded_filename = None
# Dataset Inspection state
if "ds_target_col" not in st.session_state:
    st.session_state.ds_target_col = None
if "ds_feature_cols" not in st.session_state:
    st.session_state.ds_feature_cols = None
if "ds_problem_type" not in st.session_state:
    st.session_state.ds_problem_type = None
if "uploaded_model_results" not in st.session_state:
    st.session_state.uploaded_model_results = None
if "uploaded_shap_cache" not in st.session_state:
    st.session_state.uploaded_shap_cache = None
if "uploaded_causal_graph" not in st.session_state:
    st.session_state.uploaded_causal_graph = None
if "demo_causal_graph" not in st.session_state:
    st.session_state.demo_causal_graph = None
if "uploaded_cf_result" not in st.session_state:
    st.session_state.uploaded_cf_result = None
if "demo_cf_result" not in st.session_state:
    st.session_state.demo_cf_result = None
if "agent_chat_history" not in st.session_state:
    st.session_state.agent_chat_history = []
if "agent_custom_api_key" not in st.session_state:
    st.session_state.agent_custom_api_key = None
if "latest_model_prediction" not in st.session_state:
    st.session_state.latest_model_prediction = None
if "latest_shap_individual" not in st.session_state:
    st.session_state.latest_shap_individual = None

def navigate_to(workspace_name: str):
    st.session_state.current_workspace = workspace_name

# ---------------------------------------------------------------------------
# Sidebar Navigation & System Status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="padding: 0.6rem 0 1rem 0;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="width: 44px; height: 44px; border-radius: 10px; background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%); display: flex; align-items: center; justify-content: center; font-size: 1.6rem; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.35);">
                    🛡️
                </div>
                <div>
                    <h3 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #f8fafc; letter-spacing: -0.02em;">
                        Trustworthy XAI
                    </h3>
                    <p style="margin: 0; font-size: 0.76rem; color: #94a3b8; font-weight: 600;">
                        Causal Intelligence Suite
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 1px; background: #334155; margin: 0.5rem 0 1rem;'></div>", unsafe_allow_html=True)

    # Sidebar workspace switcher
    active_nav = st.radio(
        "Platform Workspaces",
        options=WORKSPACES,
        key="current_workspace",
        format_func=lambda x: {
            WORKSPACE_HOME: "🏠  Platform Home",
            WORKSPACE_DATASET: "📊  Dataset Analysis",
            WORKSPACE_AGENT: "🤖  Explainable AI Agent",
        }.get(x, x),
    )

    st.markdown("<div style='height: 1px; background: #334155; margin: 1.2rem 0 1rem;'></div>", unsafe_allow_html=True)

    # System Status Monitor
    st.markdown(
        """
        <div style="background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; border-radius: 10px; padding: 12px; margin-top: 0.5rem;">
            <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #818cf8; letter-spacing: 0.05em; margin-bottom: 8px;">
                ⚡ System Telemetry
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px; color: #94a3b8;">
                <span>Active Mode:</span>
                <span style="color: #f8fafc; font-weight: 600;">Dual-Workspace</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; margin-bottom: 4px; color: #94a3b8;">
                <span>Inference Model:</span>
                <span style="color: #f8fafc; font-weight: 600;">Random Forest</span>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #94a3b8;">
                <span>Causal DAG:</span>
                <span style="color: #10b981; font-weight: 600;">● Online</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Data Loading & ML Training (Shared Pipeline)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_data.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

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

@st.cache_resource
def train_best_model(_X_train, _X_test, _y_train, _y_test):
    models = get_models()
    results = train_and_evaluate(models, _X_train, _X_test, _y_train, _y_test)
    best_name, best_metrics = select_best_model(results)
    save_model(best_metrics["fitted_model"], scaler, le, feature_names, directory=MODELS_DIR)
    return best_name, best_metrics, results

best_name, best_metrics, results = train_best_model(X_train, X_test, y_train, y_test)
best_model = best_metrics["fitted_model"]


# ===========================================================================
# VIEW 1: HOME PAGE (Redesigned Modern AI Platform Interface)
# ===========================================================================
if st.session_state.current_workspace == WORKSPACE_HOME:
    # Hero Title & Platform Badge
    st.markdown(
        """
        <div class="hero-container">
            <div class="platform-pill">
                <span>🛡️</span>
                <span>Enterprise Trustworthy AI Suite</span>
            </div>
            <h1 class="hero-title">
                Causal Explainable Artificial Intelligence<br>
                <span class="hero-title-accent">For Trustworthy Intelligent Systems</span>
            </h1>
            <p class="hero-subtitle">
                Bridging the gap between predictive power, game-theoretic interpretability, and structural causal reasoning. 
                Select a dedicated workspace below to begin analysis or interact with explainable intelligence.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Status Indicators
    st.markdown(
        """
        <div class="status-badge-bar">
            <span class="status-pill"><span class="status-dot"></span> Production Engine Active</span>
            <span class="status-pill"><span class="status-dot"></span> 1,000 Records Benchmark</span>
            <span class="status-pill"><span class="status-dot"></span> Structural DAG Verified</span>
            <span class="status-pill"><span class="status-dot"></span> SHAP Kernel Online</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # Two Large Clickable Cards
    # -----------------------------------------------------------------------
    col_card_left, col_card_right = st.columns(2, gap="large")

    with col_card_left:
        st.markdown(
            """
            <div class="platform-card">
                <div>
                    <div class="card-top-row">
                        <div class="card-icon-box">
                            <span>📊</span>
                        </div>
                        <span class="card-category-badge">Workspace 01 • Core Analytics</span>
                    </div>
                    <h2 class="card-title-text">Dataset Analysis</h2>
                    <p class="card-desc-text">
                        Upload datasets, train ML models, generate predictions, explore SHAP explanations, causal graphs and counterfactual simulations.
                    </p>
                </div>
                <div>
                    <div class="card-chips-wrap">
                        <span class="feature-chip">📈 Data Exploration</span>
                        <span class="feature-chip">🤖 Model Training</span>
                        <span class="feature-chip">🔍 SHAP Attribution</span>
                        <span class="feature-chip">🔬 Causal Graph</span>
                        <span class="feature-chip">🔄 Counterfactuals</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "🚀 Enter Dataset Analysis Workspace →",
            key="btn_home_to_dataset",
            use_container_width=True,
            type="primary",
            on_click=navigate_to,
            args=(WORKSPACE_DATASET,),
        )

    with col_card_right:
        st.markdown(
            """
            <div class="platform-card agent-card-theme">
                <div>
                    <div class="card-top-row">
                        <div class="card-icon-box agent-icon-box">
                            <span>🤖</span>
                        </div>
                        <span class="card-category-badge">Workspace 02 • Conversational AI</span>
                    </div>
                    <h2 class="card-title-text">Explainable AI Agent</h2>
                    <p class="card-desc-text">
                        Interact with an AI assistant that answers questions and explains its answers using AI-generated knowledge and available model evidence.
                    </p>
                </div>
                <div>
                    <div class="card-chips-wrap">
                        <span class="feature-chip">💬 Conversational Q&A</span>
                        <span class="feature-chip">🧠 Model Evidence</span>
                        <span class="feature-chip">🛡️ Trustworthy Auditing</span>
                        <span class="feature-chip">📋 Natural Language</span>
                        <span class="feature-chip">⚡ Reasoning Chain</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "💬 Enter Explainable AI Agent Workspace →",
            key="btn_home_to_agent",
            use_container_width=True,
            type="secondary",
            on_click=navigate_to,
            args=(WORKSPACE_AGENT,),
        )

    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)
    st.divider()

    # -----------------------------------------------------------------------
    # Platform Capabilities & Architecture Section
    # -----------------------------------------------------------------------
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h3 style="font-size: 1.4rem; font-weight: 700; color: #f8fafc; margin: 0 0 0.3rem;">
                Platform Architecture & Trust Pillars
            </h3>
            <p style="color: #94a3b8; font-size: 0.95rem; margin: 0;">
                How our integrated framework ensures transparency, accountability, and real causal insights
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pillar_cols = st.columns(4)
    with pillar_cols[0]:
        st.markdown(
            """
            <div class="pillar-card">
                <div style="font-size: 1.6rem;">📊</div>
                <h4>1. Dataset EDA</h4>
                <p>Curated feature synthesis with multi-variate statistical distributions and correlation profiling.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with pillar_cols[1]:
        st.markdown(
            """
            <div class="pillar-card">
                <div style="font-size: 1.6rem;">🤖</div>
                <h4>2. ML Modeling</h4>
                <p>Automated training & benchmarking of predictive baselines with F1 and precision-recall trade-offs.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with pillar_cols[2]:
        st.markdown(
            """
            <div class="pillar-card">
                <div style="font-size: 1.6rem;">🔍</div>
                <h4>3. SHAP XAI</h4>
                <p>Game-theoretic Shapley additive feature importance measuring exact individual and global contributions.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with pillar_cols[3]:
        st.markdown(
            """
            <div class="pillar-card">
                <div style="font-size: 1.6rem;">🔬</div>
                <h4>4. Causal DAGs</h4>
                <p>Structural causal graphs and counterfactual 'what-if' simulations separating correlation from true causation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ===========================================================================
# VIEW 2: DATASET ANALYSIS WORKSPACE
# ===========================================================================
elif st.session_state.current_workspace == WORKSPACE_DATASET:
    # Workspace Banner & Breadcrumb Navigation
    col_nav_left, col_nav_right = st.columns([1.8, 6.2])
    with col_nav_left:
        st.button(
            "🏠 ← Return to Platform Home",
            key="btn_back_home_dataset",
            use_container_width=True,
            on_click=navigate_to,
            args=(WORKSPACE_HOME,),
        )
    with col_nav_right:
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px;">
                <span style="font-size: 0.85rem; color: #94a3b8;">Current Location:</span>
                <span style="font-size: 0.85rem; color: #cbd5e1; font-weight: 600;">Home / Workspaces / Dataset Analysis</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="workspace-banner">
            <div>
                <h1>📊 Dataset Analysis Workspace</h1>
                <p>Upload datasets, train ML models, generate predictions, explore SHAP explanations, causal graphs and counterfactual simulations.</p>
            </div>
            <div style="text-align: right; display: flex; gap: 10px;">
                <span class="status-pill"><span class="status-dot"></span> Pipeline Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Workspace Sub-Modules via Tabs
    tab_overview, tab_inspect, tab_model, tab_shap, tab_causal, tab_counterfactual = st.tabs([
        "📊 Dataset Overview",
        "🔎 Dataset Inspection",
        "🤖 Machine Learning Model",
        "🔍 Explainable AI (SHAP)",
        "🔬 Causal Analysis",
        "🔄 Counterfactual Simulator",
    ])

    # -----------------------------------------------------------------------
    # Tab 1: Dataset Overview  — with CSV Upload + Profiling
    # -----------------------------------------------------------------------
    with tab_overview:
        # ── Data Source Selector ──────────────────────────────────────────
        st.markdown(
            """
            <div style="margin-bottom: 1rem;">
                <span style="font-size: 1rem; font-weight: 700; color: #f8fafc;">📂 Data Source</span>
                <span style="font-size: 0.82rem; color: #64748b; margin-left: 8px;">
                    Choose the demo dataset or upload your own CSV file
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        src_col1, src_col2 = st.columns(2)
        with src_col1:
            demo_active = st.session_state.dataset_source == "demo"
            st.button(
                "🎓 Use Demo Dataset  (Student Performance)" if demo_active
                else "🎓 Switch to Demo Dataset",
                key="btn_use_demo",
                use_container_width=True,
                type="primary" if demo_active else "secondary",
                disabled=demo_active,
                on_click=lambda: (
                    st.session_state.__setitem__("dataset_source", "demo"),
                    st.session_state.__setitem__("uploaded_df", None),
                    st.session_state.__setitem__("uploaded_filename", None),
                    st.session_state.__setitem__("uploaded_model_results", None),
                    st.session_state.__setitem__("uploaded_shap_cache", None),
                    st.session_state.__setitem__("uploaded_causal_graph", None),
                ),
            )
        with src_col2:
            upload_active = st.session_state.dataset_source == "upload"
            st.button(
                "📤 Upload CSV File  (Active)" if upload_active
                else "📤 Upload Your Own CSV File",
                key="btn_use_upload",
                use_container_width=True,
                type="primary" if upload_active else "secondary",
                disabled=upload_active,
                on_click=lambda: st.session_state.__setitem__("dataset_source", "upload"),
            )

        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        # ── Upload Widget (shown only when "upload" mode is active) ───────
        active_df = None
        active_name = "Demo: Student Performance"
        upload_error = None

        if st.session_state.dataset_source == "upload":
            st.markdown(
                """
                <div class="upload-zone">
                    <div style="font-size: 2rem;">📁</div>
                    <h3>Drop your CSV file here, or click Browse</h3>
                    <p>Accepts any well-formed CSV file. Max recommended size: 50 MB.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            uploaded_file = st.file_uploader(
                "Select a CSV file",
                type=["csv"],
                key="csv_uploader",
                help="Upload any CSV file. Column names are auto-detected from the first row.",
                label_visibility="collapsed",
            )

            if uploaded_file is not None:
                try:
                    # Try multiple encodings for robustness
                    try:
                        raw_df = pd.read_csv(uploaded_file, encoding="utf-8")
                    except UnicodeDecodeError:
                        uploaded_file.seek(0)
                        raw_df = pd.read_csv(uploaded_file, encoding="latin-1")

                    if raw_df.empty:
                        upload_error = "The uploaded CSV file is empty — no rows were detected."
                    elif len(raw_df.columns) < 2:
                        upload_error = "The CSV file must have at least 2 columns."
                    else:
                        st.session_state.uploaded_df = raw_df
                        st.session_state.uploaded_filename = uploaded_file.name

                except pd.errors.EmptyDataError:
                    upload_error = "The file appears to be empty or has no parseable data."
                except pd.errors.ParserError as e:
                    upload_error = f"CSV parsing error: {e}. Please check the file format and delimiter."
                except Exception as e:
                    upload_error = f"Unexpected error reading file: {e}"

            # Show error banner if something went wrong
            if upload_error:
                st.error(f"❌ **Invalid CSV:** {upload_error}")
                st.session_state.uploaded_df = None
                st.session_state.uploaded_filename = None

            # Resolve active dataframe
            if st.session_state.uploaded_df is not None:
                active_df = st.session_state.uploaded_df
                active_name = st.session_state.uploaded_filename
            else:
                # Upload mode but nothing uploaded yet — show placeholder
                if upload_error is None:
                    st.info("⬆️ Upload a CSV file above to begin analysis. The demo dataset is still available on the other tab.")
                active_df = None

        else:
            # Demo mode
            active_df = df
            active_name = "Student Performance Dataset (Demo)"

        # ── Remove / Replace Dataset Button ──────────────────────────────
        if st.session_state.dataset_source == "upload" and st.session_state.uploaded_df is not None:
            col_rep_left, col_rep_right = st.columns([3, 1])
            with col_rep_left:
                st.success(
                    f"✅ **Loaded:** `{st.session_state.uploaded_filename}` — "
                    f"{len(st.session_state.uploaded_df):,} rows × "
                    f"{len(st.session_state.uploaded_df.columns)} columns"
                )
            with col_rep_right:
                if st.button("🗑️ Remove Dataset", key="btn_remove_upload", use_container_width=True):
                    st.session_state.uploaded_df = None
                    st.session_state.uploaded_filename = None
                    st.session_state.uploaded_model_results = None
                    st.session_state.uploaded_shap_cache = None
                    st.session_state.uploaded_causal_graph = None
                    st.session_state.uploaded_cf_result = None
                    st.rerun()

        st.divider()

        # ── Main Profiling Panel ──────────────────────────────────────────
        if active_df is not None:
            n_rows, n_cols = active_df.shape
            n_missing = int(active_df.isnull().sum().sum())
            n_duplicates = int(active_df.duplicated().sum())
            completeness_pct = max(0.0, (1 - n_missing / max(n_rows * n_cols, 1)) * 100)
            uniqueness_pct   = max(0.0, (1 - n_duplicates / max(n_rows, 1)) * 100)

            # ── KPI Cards ────────────────────────────────────────────────
            st.markdown(
                f"""
                <div style="margin-bottom: 0.5rem;">
                    <span style="font-size: 1rem; font-weight: 700; color: #f8fafc;">
                        📊 {active_name}
                    </span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)

            completeness_color = "#10b981" if completeness_pct >= 95 else ("#f59e0b" if completeness_pct >= 80 else "#ef4444")
            uniqueness_color   = "#10b981" if uniqueness_pct  >= 95 else ("#f59e0b" if uniqueness_pct  >= 80 else "#ef4444")

            kpi1.metric("📋 Rows", f"{n_rows:,}")
            kpi2.metric("📐 Columns", str(n_cols))
            kpi3.metric("⚠️ Missing Values", str(n_missing), delta=None, help="Total null/NaN cells across all columns")
            kpi4.metric("🔁 Duplicate Rows", str(n_duplicates), delta=None)
            kpi5.metric(
                "✅ Completeness",
                f"{completeness_pct:.1f}%",
                delta=f"{n_missing} nulls" if n_missing > 0 else "No nulls",
                delta_color="inverse",
            )
            kpi6.metric(
                "🧬 Uniqueness",
                f"{uniqueness_pct:.1f}%",
                delta=f"{n_duplicates} dupes" if n_duplicates > 0 else "No duplicates",
                delta_color="inverse",
            )

            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

            # ── Tabbed Profiling Sections ─────────────────────────────────
            pt1, pt2, pt3, pt4, pt5 = st.tabs([
                "🔍 Data Preview",
                "🏷️ Schema & Types",
                "📈 Statistics",
                "🩺 Quality Report",
                "📉 Distributions",
            ])

            # -- Preview tab --
            with pt1:
                n_preview = st.select_slider(
                    "Rows to preview",
                    options=[10, 20, 50, 100, 200, 500],
                    value=20,
                    key="preview_rows",
                )
                st.dataframe(active_df.head(n_preview), use_container_width=True)
                st.caption(f"Showing first {n_preview} of {n_rows:,} rows.")

            # -- Schema tab --
            with pt2:
                schema_rows = []
                for col_name in active_df.columns:
                    col_series = active_df[col_name]
                    dtype_str  = str(col_series.dtype)
                    n_null     = int(col_series.isnull().sum())
                    n_unique   = int(col_series.nunique())
                    null_pct   = f"{n_null / n_rows * 100:.1f}%" if n_rows > 0 else "—"
                    sample_vals = col_series.dropna().unique()[:3]
                    sample_str  = ", ".join(str(v) for v in sample_vals)
                    schema_rows.append({
                        "Column":          col_name,
                        "Dtype":           dtype_str,
                        "Non-Null Count":  n_rows - n_null,
                        "Null Count":      n_null,
                        "Null %":          null_pct,
                        "Unique Values":   n_unique,
                        "Sample Values":   sample_str,
                    })
                schema_df = pd.DataFrame(schema_rows)
                st.dataframe(schema_df, use_container_width=True, hide_index=True)

            # -- Statistics tab --
            with pt3:
                numeric_cols = active_df.select_dtypes(include="number").columns.tolist()
                cat_cols     = active_df.select_dtypes(exclude="number").columns.tolist()

                if numeric_cols:
                    st.markdown("**Numeric Columns — Descriptive Statistics**")
                    st.dataframe(
                        active_df[numeric_cols].describe().round(4).T.rename_axis("Column").reset_index(),
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.info("No numeric columns detected.")

                if cat_cols:
                    st.markdown("**Categorical Columns — Value Counts (top 10 per column)**")
                    for c in cat_cols:
                        with st.expander(f"📌 `{c}` — {active_df[c].nunique()} unique values"):
                            vc = (
                                active_df[c]
                                .value_counts()
                                .head(10)
                                .rename_axis("Value")
                                .reset_index(name="Count")
                            )
                            vc["Percentage"] = (vc["Count"] / n_rows * 100).round(2).astype(str) + "%"
                            st.dataframe(vc, use_container_width=True, hide_index=True)

            # -- Quality Report tab --
            with pt4:
                st.markdown("#### 🩺 Data Quality Report")

                # Overall quality score (simple heuristic)
                quality_score = round((completeness_pct + uniqueness_pct) / 2, 1)
                q_color = "#10b981" if quality_score >= 90 else ("#f59e0b" if quality_score >= 70 else "#ef4444")
                q_label = "Excellent" if quality_score >= 90 else ("Good" if quality_score >= 70 else "Needs Attention")

                q_col1, q_col2 = st.columns([1, 2])
                with q_col1:
                    st.markdown(
                        f"""
                        <div style="background:#1e293b; border:1px solid #334155; border-radius:12px; padding:1.4rem; text-align:center;">
                            <div style="font-size:0.78rem; color:#94a3b8; font-weight:700; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:0.4rem;">
                                Overall Quality Score
                            </div>
                            <div style="font-size:2.8rem; font-weight:800; color:{q_color}; letter-spacing:-0.03em;">{quality_score}%</div>
                            <div style="font-size:0.9rem; color:{q_color}; font-weight:600; margin-top:0.2rem;">{q_label}</div>
                            <div class="quality-bar-wrap" style="margin-top:0.8rem;">
                                <div class="quality-bar-fill" style="width:{quality_score}%; background:{q_color};"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with q_col2:
                    # Per-column missing values breakdown
                    missing_by_col = active_df.isnull().sum()
                    missing_by_col = missing_by_col[missing_by_col > 0]

                    if missing_by_col.empty:
                        st.success("✅ **No missing values detected** — the dataset is complete.")
                    else:
                        st.warning(f"⚠️ **{len(missing_by_col)} column(s)** contain missing values:")
                        mv_df = pd.DataFrame({
                            "Column":    missing_by_col.index,
                            "Missing":   missing_by_col.values,
                            "Missing %": (missing_by_col.values / n_rows * 100).round(2),
                        }).sort_values("Missing %", ascending=False).reset_index(drop=True)
                        st.dataframe(mv_df, use_container_width=True, hide_index=True)

                st.markdown("")

                # Duplicate rows detail
                if n_duplicates > 0:
                    st.warning(f"⚠️ **{n_duplicates} duplicate row(s)** detected ({n_duplicates/n_rows*100:.1f}% of dataset).")
                    with st.expander(f"🔁 View {min(n_duplicates, 50)} duplicate rows"):
                        st.dataframe(
                            active_df[active_df.duplicated(keep=False)].head(50),
                            use_container_width=True,
                        )
                else:
                    st.success("✅ **No duplicate rows** — all records are unique.")

                # Column-level null heatmap (if any nulls)
                if n_missing > 0:
                    with st.expander("🗺️ Column Null Heatmap (sample of 100 rows)"):
                        sample_null = active_df.head(100).isnull().astype(int)
                        fig_null, ax_null = plt.subplots(figsize=(min(len(active_df.columns) * 0.8, 14), 4))
                        fig_null.patch.set_facecolor("#0e1117")
                        ax_null.set_facecolor("#0e1117")
                        sns.heatmap(
                            sample_null.T,
                            cmap=["#1e293b", "#ef4444"],
                            linewidths=0.3,
                            linecolor="#0e1117",
                            cbar=False,
                            ax=ax_null,
                            yticklabels=True,
                            xticklabels=False,
                        )
                        ax_null.tick_params(colors="#ccc", labelsize=9)
                        for spine in ax_null.spines.values():
                            spine.set_color("#333")
                        ax_null.set_title("Red = Missing | Dark = Present  (first 100 rows)", color="#94a3b8", fontsize=10)
                        plt.tight_layout()
                        st.pyplot(fig_null)

            # -- Distributions tab --
            with pt5:
                numeric_plot_cols = active_df.select_dtypes(include="number").columns.tolist()
                cat_plot_cols     = active_df.select_dtypes(exclude="number").columns.tolist()

                if not numeric_plot_cols and not cat_plot_cols:
                    st.info("No plottable columns detected.")
                else:
                    if numeric_plot_cols:
                        st.markdown("**📊 Numeric Feature Distributions**")
                        n_num = len(numeric_plot_cols)
                        ncols_fig = min(3, n_num)
                        nrows_fig = (n_num + ncols_fig - 1) // ncols_fig
                        fig_dist, axes_dist = plt.subplots(
                            nrows_fig, ncols_fig,
                            figsize=(14, 4 * nrows_fig),
                            squeeze=False,
                        )
                        fig_dist.patch.set_facecolor("#0e1117")
                        flat_axes = axes_dist.ravel()

                        for i, col_name in enumerate(numeric_plot_cols):
                            ax = flat_axes[i]
                            ax.set_facecolor("#0e1117")
                            for spine in ax.spines.values():
                                spine.set_color("#333")
                            ax.tick_params(colors="#ccc", labelsize=8)
                            ax.set_title(col_name.replace("_", " ").title(), color="#ccc", fontsize=10)
                            ax.set_xlabel("", color="#ccc")
                            ax.set_ylabel("Count", color="#888", fontsize=8)
                            col_data = active_df[col_name].dropna()
                            ax.hist(col_data, bins=30, color="#6366f1", alpha=0.78, edgecolor="none")
                            mean_val = col_data.mean()
                            ax.axvline(mean_val, color="#f472b6", linestyle="--", linewidth=1.2, label=f"Mean: {mean_val:.2f}")
                            ax.legend(fontsize=7, facecolor="#0e1117", edgecolor="#333", labelcolor="#ccc")

                        # Hide empty subplots
                        for j in range(len(numeric_plot_cols), len(flat_axes)):
                            flat_axes[j].set_visible(False)

                        plt.tight_layout()
                        st.pyplot(fig_dist)

                    if cat_plot_cols:
                        st.markdown("**📌 Categorical Feature Value Counts**")
                        for c in cat_plot_cols[:6]:  # limit to 6 cat columns
                            vc_data = active_df[c].value_counts().head(15)
                            if vc_data.empty:
                                continue
                            fig_cat, ax_cat = plt.subplots(figsize=(8, 3))
                            fig_cat.patch.set_facecolor("#0e1117")
                            ax_cat.set_facecolor("#0e1117")
                            ax_cat.barh(vc_data.index.astype(str)[::-1], vc_data.values[::-1], color="#818cf8", alpha=0.85)
                            ax_cat.set_title(f"{c} — Top Values", color="#ccc", fontsize=10)
                            ax_cat.tick_params(colors="#ccc", labelsize=8)
                            for spine in ax_cat.spines.values():
                                spine.set_color("#333")
                            ax_cat.set_xlabel("Count", color="#888", fontsize=8)
                            plt.tight_layout()
                            st.pyplot(fig_cat)

        elif st.session_state.dataset_source == "upload" and st.session_state.uploaded_df is None and upload_error is None:
            # Upload mode, file not yet chosen — nice placeholder
            st.markdown(
                """
                <div style="text-align: center; padding: 3rem 1rem; background: rgba(30,41,59,0.3); border: 1px dashed #334155; border-radius: 14px;">
                    <div style="font-size: 3rem;">📂</div>
                    <h3 style="color: #f8fafc; margin: 0.8rem 0 0.4rem;">No dataset loaded yet</h3>
                    <p style="color: #94a3b8; font-size: 0.95rem;">
                        Upload a CSV file above to view its profile and analysis.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -----------------------------------------------------------------------
    # Tab 2: Dataset Inspection — Column Classification, Target/Feature Setup
    # -----------------------------------------------------------------------
    with tab_inspect:

        # Determine which dataframe to inspect (same logic as tab_overview)
        if st.session_state.dataset_source == "upload" and st.session_state.uploaded_df is not None:
            insp_df   = st.session_state.uploaded_df
            insp_name = st.session_state.uploaded_filename
        elif st.session_state.dataset_source == "demo":
            insp_df   = df
            insp_name = "Student Performance Dataset (Demo)"
        else:
            insp_df   = None
            insp_name = None

        # Reset inspection widget states if dataset changed
        curr_ds_id = insp_name if insp_df is not None else None
        if st.session_state.get("last_inspected_dataset") != curr_ds_id:
            st.session_state.last_inspected_dataset = curr_ds_id
            for k in ["ds_target_sel", "ds_problem_type_sel", "ds_feature_sel", "inspect_cat_sel", "inspect_dist_sel", "inspect_corr_sel"]:
                if k in st.session_state:
                    del st.session_state[k]

        if insp_df is None:
            st.info("📂 Load a dataset first in the **Dataset Overview** tab, then return here.")
        else:
            n_insp_rows, n_insp_cols = insp_df.shape

            # ================================================================
            # STEP 1 — Automated Column Classification
            # ================================================================
            st.markdown(
                """
                <div class="inspect-section-header">
                    <span>🧬</span> Step 1 — Automated Column Classification
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption(f"Analysing **{n_insp_cols} columns** in `{insp_name}` ({n_insp_rows:,} rows)")

            # ── Helper: classify a single column ─────────────────────────
            def classify_column(series: pd.Series, n_rows: int) -> dict:
                """Return a dict with kind, reasons, and stats for a column."""
                n_unique   = series.nunique(dropna=True)
                n_null     = int(series.isnull().sum())
                dtype      = series.dtype
                name_lower = series.name.lower()

                kind    = "numeric"
                reasons = []

                # Datetime detection
                if pd.api.types.is_datetime64_any_dtype(dtype):
                    kind = "datetime"
                    reasons.append("datetime dtype")
                # Boolean detection
                elif pd.api.types.is_bool_dtype(dtype):
                    kind = "boolean"
                    reasons.append("bool dtype")
                elif pd.api.types.is_string_dtype(dtype) or dtype == object:
                    # Try parsing as datetime
                    try:
                        pd.to_datetime(series.dropna().head(50), infer_datetime_format=True)
                        kind = "datetime"
                        reasons.append("parseable as datetime")
                    except Exception:
                        pass

                    if kind != "datetime":
                        # Heuristic ID check: name hint + all-or-almost-unique
                        id_name_hints = ["id", "_id", "uuid", "key", "index", "code", "ref", "no", "num", "#"]
                        name_is_id = any(h in name_lower for h in id_name_hints)
                        cardinality_ratio = n_unique / max(n_rows, 1)

                        if (name_is_id or cardinality_ratio >= 0.95) and n_unique > 10:
                            kind = "identifier"
                            if name_is_id:
                                reasons.append(f"name suggests ID (`{series.name}`)")
                            if cardinality_ratio >= 0.95:
                                reasons.append(f"very high cardinality ({n_unique}/{n_rows} unique = {cardinality_ratio:.0%})")
                        else:
                            kind = "categorical"
                            reasons.append(f"{n_unique} unique string values")
                elif pd.api.types.is_integer_dtype(dtype):
                    # Binary integer (0/1) → boolean-like
                    id_name_hints = ["id", "_id", "uuid", "key", "index", "code", "ref", "no"]
                    name_is_id = any(h in name_lower for h in id_name_hints)
                    cardinality_ratio = n_unique / max(n_rows, 1)

                    if name_is_id and cardinality_ratio >= 0.9 and n_unique > 10:
                        kind = "identifier"
                        reasons.append(f"name suggests ID and high cardinality ({cardinality_ratio:.0%})")
                    elif n_unique == 2:
                        non_null = series.dropna().unique()
                        if set(int(v) for v in non_null if pd.notna(v)) <= {0, 1}:
                            kind = "boolean"
                            reasons.append("binary 0/1 integer")
                        else:
                            kind = "categorical"
                            reasons.append("only 2 unique integer values")
                    elif n_unique <= max(10, int(n_rows ** 0.4)):
                        kind = "categorical"
                        reasons.append(f"low-cardinality integer ({n_unique} unique ≤ threshold)")
                    else:
                        kind = "numeric"
                        reasons.append(f"continuous integer ({n_unique} unique values)")
                elif pd.api.types.is_float_dtype(dtype):
                    kind = "numeric"
                    reasons.append(f"float dtype ({n_unique} unique)")

                return {
                    "name":     series.name,
                    "kind":     kind,
                    "dtype":    str(dtype),
                    "n_unique": n_unique,
                    "n_null":   n_null,
                    "null_pct": n_null / max(n_rows, 1) * 100,
                    "reasons":  reasons,
                }

            # Classify all columns
            col_profiles = {c: classify_column(insp_df[c], n_insp_rows) for c in insp_df.columns}

            numeric_cols  = [c for c, p in col_profiles.items() if p["kind"] == "numeric"]
            cat_cols      = [c for c, p in col_profiles.items() if p["kind"] == "categorical"]
            bool_cols     = [c for c, p in col_profiles.items() if p["kind"] == "boolean"]
            id_cols       = [c for c, p in col_profiles.items() if p["kind"] == "identifier"]
            datetime_cols = [c for c, p in col_profiles.items() if p["kind"] == "datetime"]

            BADGE_HTML = {
                "numeric":    '<span class="col-type-badge badge-numeric">📊 Numeric</span>',
                "categorical":'<span class="col-type-badge badge-category">🏷️ Categorical</span>',
                "boolean":    '<span class="col-type-badge badge-bool">🔘 Boolean</span>',
                "identifier": '<span class="col-type-badge badge-id">🔑 Identifier</span>',
                "datetime":   '<span class="col-type-badge badge-datetime">📅 Datetime</span>',
            }

            # Summary pills
            summary_parts = []
            if numeric_cols:   summary_parts.append(f"📊 **{len(numeric_cols)} numeric**")
            if cat_cols:       summary_parts.append(f"🏷️ **{len(cat_cols)} categorical**")
            if bool_cols:      summary_parts.append(f"🔘 **{len(bool_cols)} boolean**")
            if id_cols:        summary_parts.append(f"🔑 **{len(id_cols)} identifier** (excluded from modelling)")
            if datetime_cols:  summary_parts.append(f"📅 **{len(datetime_cols)} datetime**")
            st.markdown("**Detected:** " + " · ".join(summary_parts))

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            # Column-by-column table
            schema_view = []
            for c, p in col_profiles.items():
                badge = BADGE_HTML.get(p["kind"], "")
                miss_str = f"{p['n_null']} ({p['null_pct']:.1f}%)" if p["n_null"] > 0 else "—"
                schema_view.append({
                    "Column":     c,
                    "Type":       p["kind"].title(),
                    "Dtype":      p["dtype"],
                    "Unique":     p["n_unique"],
                    "Missing":    miss_str,
                    "Reasoning":  "; ".join(p["reasons"]) if p["reasons"] else "inferred from dtype",
                })

            schema_view_df = pd.DataFrame(schema_view)

            def _style_kind(val):
                colour_map = {
                    "Numeric":    "#818cf8",
                    "Categorical":"#34d399",
                    "Boolean":    "#f472b6",
                    "Identifier": "#fbbf24",
                    "Datetime":   "#38bdf8",
                }
                col = colour_map.get(val, "#cbd5e1")
                return f"color: {col}; font-weight: 700"

            styled_schema = schema_view_df.style.map(_style_kind, subset=["Type"])
            st.dataframe(styled_schema, use_container_width=True, hide_index=True)

            if id_cols:
                st.markdown(
                    f'<div class="warn-box">🔑 <strong>Identifier columns detected:</strong> '
                    f'<code>{", ".join(id_cols)}</code> — these will be excluded from automatic feature suggestions. '
                    f'You can still manually include them if needed.</div>',
                    unsafe_allow_html=True,
                )

            # ================================================================
            # STEP 2 — Missing Values & Duplicates Quick Summary
            # ================================================================
            st.markdown(
                '<div class="inspect-section-header"><span>🩺</span> Step 2 — Data Quality Quick View</div>',
                unsafe_allow_html=True,
            )

            q2a, q2b = st.columns(2)

            # Missing values per column
            with q2a:
                missing_series = insp_df.isnull().sum()
                missing_cols   = missing_series[missing_series > 0]
                if missing_cols.empty:
                    st.markdown('<div class="ok-box">✅ <strong>No missing values</strong> across all columns.</div>', unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="warn-box">⚠️ <strong>{len(missing_cols)} column(s)</strong> have missing values '
                        f'(total: {missing_series.sum()} cells).</div>',
                        unsafe_allow_html=True,
                    )
                    mv_df = pd.DataFrame({
                        "Column":     missing_cols.index,
                        "Missing":    missing_cols.values,
                        "Missing %":  (missing_cols.values / n_insp_rows * 100).round(1),
                    }).sort_values("Missing %", ascending=False).reset_index(drop=True)
                    st.dataframe(mv_df, use_container_width=True, hide_index=True, height=180)

            # Duplicates
            with q2b:
                n_dupes = int(insp_df.duplicated().sum())
                if n_dupes == 0:
                    st.markdown('<div class="ok-box">✅ <strong>No duplicate rows</strong> detected.</div>', unsafe_allow_html=True)
                else:
                    dupe_pct = n_dupes / n_insp_rows * 100
                    st.markdown(
                        f'<div class="warn-box">⚠️ <strong>{n_dupes} duplicate row(s)</strong> '
                        f'({dupe_pct:.1f}% of dataset). Consider deduplication before training.</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander(f"🔁 View duplicate rows (up to 50)"):
                        st.dataframe(
                            insp_df[insp_df.duplicated(keep=False)].head(50),
                            use_container_width=True,
                        )

            # ================================================================
            # STEP 3 — Unique Values for Categorical Columns
            # ================================================================
            if cat_cols or bool_cols:
                st.markdown(
                    '<div class="inspect-section-header"><span>🏷️</span> Step 3 — Categorical Column Explorer</div>',
                    unsafe_allow_html=True,
                )
                all_cat_like = cat_cols + bool_cols
                cat_sel = st.selectbox(
                    "Select a categorical/boolean column to explore",
                    options=all_cat_like,
                    key="inspect_cat_sel",
                )
                if cat_sel:
                    cat_series = insp_df[cat_sel]
                    n_uniq = cat_series.nunique()
                    vc = cat_series.value_counts(dropna=False).reset_index()
                    vc.columns = ["Value", "Count"]
                    vc["Percentage"] = (vc["Count"] / n_insp_rows * 100).round(2).astype(str) + "%"
                    vc["Value"] = vc["Value"].astype(str)

                    vcol1, vcol2 = st.columns([1.4, 2.6])
                    with vcol1:
                        st.markdown(f"**`{cat_sel}`** — {n_uniq} unique values")
                        st.dataframe(vc.head(30), use_container_width=True, hide_index=True)

                    with vcol2:
                        fig_vc, ax_vc = plt.subplots(figsize=(7, min(0.45 * min(n_uniq, 20), 7) + 1.5))
                        fig_vc.patch.set_facecolor("#0e1117")
                        ax_vc.set_facecolor("#0e1117")
                        plot_vc = vc.head(20)
                        bars = ax_vc.barh(
                            plot_vc["Value"].iloc[::-1],
                            plot_vc["Count"].iloc[::-1],
                            color="#34d399", alpha=0.82,
                        )
                        ax_vc.set_title(f"{cat_sel} — Value Distribution", color="#ccc", fontsize=10, pad=8)
                        ax_vc.set_xlabel("Count", color="#888", fontsize=8)
                        ax_vc.tick_params(colors="#ccc", labelsize=8)
                        for spine in ax_vc.spines.values():
                            spine.set_color("#333")
                        for bar, row in zip(bars[::-1], plot_vc.itertuples()):
                            ax_vc.text(
                                bar.get_width() + max(plot_vc["Count"]) * 0.01,
                                bar.get_y() + bar.get_height() / 2,
                                row.Percentage,
                                va="center", color="#94a3b8", fontsize=7,
                            )
                        plt.tight_layout()
                        st.pyplot(fig_vc)
            else:
                st.markdown(
                    '<div class="inspect-section-header"><span>🏷️</span> Step 3 — Categorical Column Explorer</div>',
                    unsafe_allow_html=True,
                )
                st.info("No categorical columns detected in this dataset.")

            # ================================================================
            # STEP 4 — Feature Distributions
            # ================================================================
            st.markdown(
                '<div class="inspect-section-header"><span>📉</span> Step 4 — Feature Distributions</div>',
                unsafe_allow_html=True,
            )

            if numeric_cols:
                dist_col_sel = st.multiselect(
                    "Select numeric columns to plot",
                    options=numeric_cols,
                    default=numeric_cols[:min(6, len(numeric_cols))],
                    key="inspect_dist_sel",
                )
                if dist_col_sel:
                    n_dist = len(dist_col_sel)
                    ncols_d = min(3, n_dist)
                    nrows_d = (n_dist + ncols_d - 1) // ncols_d
                    fig_d, axs_d = plt.subplots(nrows_d, ncols_d, figsize=(14, 4 * nrows_d), squeeze=False)
                    fig_d.patch.set_facecolor("#0e1117")
                    flat_d = axs_d.ravel()

                    for i, cn in enumerate(dist_col_sel):
                        ax = flat_d[i]
                        ax.set_facecolor("#0e1117")
                        for s in ax.spines.values():
                            s.set_color("#333")
                        ax.tick_params(colors="#ccc", labelsize=8)
                        col_data = insp_df[cn].dropna()
                        ax.hist(col_data, bins=30, color="#6366f1", alpha=0.78, edgecolor="none")
                        mean_v = col_data.mean()
                        median_v = col_data.median()
                        ax.axvline(mean_v,   color="#f472b6", linestyle="--", linewidth=1.2, label=f"Mean: {mean_v:.2f}")
                        ax.axvline(median_v, color="#fbbf24", linestyle=":",  linewidth=1.2, label=f"Median: {median_v:.2f}")
                        ax.set_title(cn.replace("_", " ").title(), color="#ccc", fontsize=10)
                        ax.set_ylabel("Count", color="#888", fontsize=8)
                        ax.legend(fontsize=7, facecolor="#0e1117", edgecolor="#333", labelcolor="#ccc")

                    for j in range(len(dist_col_sel), len(flat_d)):
                        flat_d[j].set_visible(False)

                    plt.tight_layout()
                    st.pyplot(fig_d)
                else:
                    st.info("Select at least one numeric column above.")
            else:
                st.info("No numeric columns available for distribution plots.")

            # ================================================================
            # STEP 5 — Correlation Heatmap
            # ================================================================
            st.markdown(
                '<div class="inspect-section-header"><span>🌡️</span> Step 5 — Correlation Heatmap (Numeric Columns)</div>',
                unsafe_allow_html=True,
            )

            if len(numeric_cols) >= 2:
                corr_sel = st.multiselect(
                    "Select numeric columns for correlation matrix",
                    options=numeric_cols,
                    default=numeric_cols[:min(10, len(numeric_cols))],
                    key="inspect_corr_sel",
                )
                if len(corr_sel) >= 2:
                    corr_method = st.radio(
                        "Correlation method",
                        options=["pearson", "spearman", "kendall"],
                        horizontal=True,
                        key="inspect_corr_method",
                    )
                    corr_matrix = insp_df[corr_sel].corr(method=corr_method)

                    fig_corr, ax_corr = plt.subplots(figsize=(min(len(corr_sel) * 1.1 + 1, 14), min(len(corr_sel) * 1.0 + 1, 12)))
                    fig_corr.patch.set_facecolor("#0e1117")
                    ax_corr.set_facecolor("#0e1117")
                    mask = np.zeros_like(corr_matrix, dtype=bool)
                    mask[np.triu_indices_from(mask, k=1)] = True

                    sns.heatmap(
                        corr_matrix,
                        annot=True,
                        fmt=".2f",
                        cmap="RdBu_r",
                        center=0,
                        vmin=-1, vmax=1,
                        ax=ax_corr,
                        linewidths=0.4,
                        linecolor="#0e1117",
                        cbar_kws={"shrink": 0.75},
                        annot_kws={"size": 8},
                        square=True,
                    )
                    ax_corr.tick_params(colors="#ccc", labelsize=8)
                    ax_corr.set_xticklabels(ax_corr.get_xticklabels(), rotation=40, ha="right")
                    ax_corr.set_yticklabels(ax_corr.get_yticklabels(), rotation=0)
                    for spine in ax_corr.spines.values():
                        spine.set_color("#333")
                    ax_corr.set_title(
                        f"Correlation Matrix ({corr_method.title()}) — {len(corr_sel)} features",
                        color="#ccc", fontsize=11, pad=12,
                    )
                    plt.tight_layout()
                    st.pyplot(fig_corr)

                    # Highlight strong correlations
                    strong_pairs = []
                    for i_c, r in enumerate(corr_sel):
                        for j_c, c_col in enumerate(corr_sel):
                            if i_c < j_c:
                                v = corr_matrix.loc[r, c_col]
                                if abs(v) >= 0.7:
                                    strong_pairs.append((r, c_col, round(v, 3)))

                    if strong_pairs:
                        st.markdown("**⚡ Strong correlations detected (|r| ≥ 0.70):**")
                        for a, b, v in strong_pairs:
                            direction = "positive" if v > 0 else "negative"
                            emoji = "🔴" if abs(v) >= 0.9 else "🟠"
                            st.markdown(f"- {emoji} `{a}` ↔ `{b}`: **{v}** ({direction})")
                    else:
                        st.info("No strong correlations (|r| ≥ 0.70) detected among selected columns.")
                else:
                    st.info("Select at least 2 numeric columns for the correlation matrix.")
            else:
                st.info("At least 2 numeric columns are required to compute a correlation matrix.")

            # ================================================================
            # STEP 6 — Target Column & Feature Selection + Problem Type
            # ================================================================
            st.markdown(
                '<div class="inspect-section-header"><span>🎯</span> Step 6 — Target & Feature Setup</div>',
                unsafe_allow_html=True,
            )

            # Only columns that are NOT identifiers/datetimes make sense as target
            candidate_targets = [c for c in insp_df.columns if col_profiles[c]["kind"] not in ("identifier", "datetime")]
            # Non-ID, non-datetime columns that are also not pure dates
            candidate_features = [c for c in insp_df.columns if col_profiles[c]["kind"] not in ("identifier", "datetime")]

            if not candidate_targets:
                st.error("❌ No suitable target columns found. All columns appear to be identifiers or datetime fields.")
            else:
                # ── Auto-suggest target ─────────────────────────────────
                def _auto_suggest_target(col_names: list[str], profiles: dict) -> str | None:
                    """Return the most likely target column name via heuristics.
                    Priority:
                    1. Primary outcome hints (exact matches like target, label, result, passed, price, etc.)
                    2. Secondary hints (score, rate, value, amount)
                    3. Whole-word suffixes/prefixes (e.g. is_passed, target_val)
                    4. Convention fallback: the last candidate column.
                    """
                    primary_hints = [
                        "target", "label", "result", "outcome", "output",
                        "y", "dependent", "response", "passed", "pass", "fail",
                        "survived", "default", "churn", "fraud", "status",
                        "price", "salary"
                    ]
                    secondary_hints = ["value", "cost", "revenue", "rate", "amount", "score", "class"]

                    # Pass 1: Exact match with primary outcome hints
                    for c in col_names:
                        if c.lower() in primary_hints:
                            return c

                    # Pass 2: Suffix or prefix with primary hints
                    for hint in primary_hints:
                        for c in col_names:
                            if c.lower().endswith(f"_{hint}") or c.lower().startswith(f"{hint}_"):
                                return c

                    # Pass 3: Exact match with secondary hints
                    for c in col_names:
                        if c.lower() in secondary_hints:
                            return c

                    # Pass 4: Fall back to the last column (convention in ML benchmarks)
                    return col_names[-1] if col_names else None

                auto_target = _auto_suggest_target(candidate_targets, col_profiles)

                # Determine default index for selectbox
                target_default_idx = candidate_targets.index(auto_target) if auto_target in candidate_targets else 0

                tgt_col1, tgt_col2 = st.columns([1.3, 2.7])
                with tgt_col1:
                    selected_target = st.selectbox(
                        "🎯 Select Target Column",
                        options=candidate_targets,
                        index=target_default_idx,
                        key="ds_target_sel",
                        help="The column you want the model to predict.",
                    )

                with tgt_col2:
                    if auto_target:
                        st.markdown(
                            f'<div class="suggestion-box">💡 <strong>Auto-suggestion:</strong> '
                            f'<code>{auto_target}</code> was identified as the most likely target column '
                            f'(matched a known target-name pattern). You can override this.</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            '<div class="warn-box">ℹ️ No automatic target detected by name. '
                            'Please select the column you want to predict.</div>',
                            unsafe_allow_html=True,
                        )

                # ── Auto-suggest problem type ───────────────────────────
                def _infer_problem_type(series: pd.Series, profile: dict) -> tuple[str, str]:
                    """Return (suggested_type, reason) — 'classification' or 'regression'."""
                    kind   = profile["kind"]
                    n_uniq = profile["n_unique"]
                    n_rows = len(series.dropna())

                    if kind in ("categorical", "boolean"):
                        return "Classification", f"target is {kind} (non-numeric)"
                    if kind == "numeric":
                        if n_uniq <= 10:
                            return "Classification", f"numeric target has only {n_uniq} unique values (likely discrete classes)"
                        cardinality_ratio = n_uniq / max(n_rows, 1)
                        if cardinality_ratio >= 0.05:
                            return "Regression", f"numeric target has {n_uniq} unique values ({cardinality_ratio:.0%} cardinality) — likely continuous"
                        return "Classification", f"numeric target has {n_uniq} unique values — could be ordinal classes"
                    return "Classification", "default assumption"

                target_profile  = col_profiles[selected_target]
                target_series   = insp_df[selected_target]
                suggested_type, type_reason = _infer_problem_type(target_series, target_profile)

                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                prob_col1, prob_col2 = st.columns([1.3, 2.7])
                with prob_col1:
                    selected_problem = st.radio(
                        "🔧 Problem Type",
                        options=["Classification", "Regression"],
                        index=0 if suggested_type == "Classification" else 1,
                        key="ds_problem_type_sel",
                        help="Choose whether this is a classification (discrete labels) or regression (continuous value) problem.",
                    )
                with prob_col2:
                    type_color = "suggestion-box" if selected_problem == suggested_type else "warn-box"
                    type_icon  = "💡" if selected_problem == suggested_type else "⚠️"
                    override_note = "" if selected_problem == suggested_type else f" (you overrode the suggestion of <strong>{suggested_type}</strong>)"
                    st.markdown(
                        f'<div class="{type_color}">{type_icon} <strong>Auto-detected:</strong> '
                        f'<strong>{suggested_type}</strong> — {type_reason}.{override_note}</div>',
                        unsafe_allow_html=True,
                    )

                    # Show target column summary
                    t_kind = target_profile["kind"]
                    t_uniq = target_profile["n_unique"]
                    t_null = target_profile["n_null"]
                    t_badge = BADGE_HTML.get(t_kind, "")

                    st.markdown(
                        f"**Selected target:** `{selected_target}` &nbsp; {t_badge} &nbsp; "
                        f"— `{t_uniq}` unique values, `{t_null}` nulls",
                        unsafe_allow_html=True,
                    )

                    if selected_problem == "Classification":
                        top_classes = target_series.value_counts().head(10)
                        class_strs  = [f"`{v}` ({c})" for v, c in zip(top_classes.index, top_classes.values)]
                        st.markdown("**Class distribution:** " + " · ".join(class_strs))
                    else:
                        num_tgt = pd.to_numeric(target_series, errors="coerce").dropna()
                        if len(num_tgt) > 0:
                            st.markdown(
                                f"**Target range:** min `{num_tgt.min():.4g}`, "
                                f"mean `{num_tgt.mean():.4g}`, "
                                f"max `{num_tgt.max():.4g}`"
                            )

                # ── Feature Selection ───────────────────────────────────
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                st.markdown("**📌 Feature Column Selection**")

                # Suggested features = all candidate cols except target and id cols
                default_features = [
                    c for c in candidate_features
                    if c != selected_target and col_profiles[c]["kind"] not in ("identifier", "datetime")
                ]

                if id_cols or datetime_cols:
                    st.markdown(
                        f'<div class="suggestion-box">💡 <strong>Auto-excluded:</strong> '
                        f'Identifier columns (<code>{"</code>, <code>".join(id_cols) if id_cols else "none"}</code>) '
                        f'and datetime columns (<code>{"</code>, <code>".join(datetime_cols) if datetime_cols else "none"}</code>) '
                        f'are excluded from the default feature set. You can re-add them manually below.</div>',
                        unsafe_allow_html=True,
                    )

                # Available pool = all columns minus the target (user may add IDs/datetimes back)
                all_possible_features = [c for c in insp_df.columns if c != selected_target]

                selected_features = st.multiselect(
                    "Select input feature columns",
                    options=all_possible_features,
                    default=[c for c in default_features if c in all_possible_features],
                    key="ds_feature_sel",
                    help="Hold Ctrl/Cmd to select multiple. Identifier & datetime columns are excluded by default.",
                )

                # ── Validation Messages ─────────────────────────────────
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                validation_ok = True

                if not selected_features:
                    st.error("❌ **No features selected.** Please select at least one input feature column.")
                    validation_ok = False
                else:
                    # Warn if ID/datetime added back
                    id_in_features = [c for c in selected_features if col_profiles[c]["kind"] == "identifier"]
                    dt_in_features = [c for c in selected_features if col_profiles[c]["kind"] == "datetime"]
                    if id_in_features:
                        st.warning(
                            f"⚠️ Identifier column(s) included as features: "
                            f"`{'`, `'.join(id_in_features)}`. These may cause data leakage or poor generalisation."
                        )
                    if dt_in_features:
                        st.warning(
                            f"⚠️ Datetime column(s) included as features: "
                            f"`{'`, `'.join(dt_in_features)}`. Raw datetime values may not be directly usable by most models."
                        )

                    # Warn about high null columns in features
                    high_null_feats = [
                        c for c in selected_features
                        if col_profiles[c]["null_pct"] > 20
                    ]
                    if high_null_feats:
                        st.warning(
                            f"⚠️ High missing-value columns in features (>20% null): "
                            f"`{'`, `'.join(high_null_feats)}`. Consider imputation before training."
                        )

                    # Warn about target null rate
                    if target_profile["null_pct"] > 0:
                        st.warning(
                            f"⚠️ Target column `{selected_target}` has **{target_profile['n_null']} missing values** "
                            f"({target_profile['null_pct']:.1f}%). Rows with null targets are typically dropped during training."
                        )

                if validation_ok and selected_features:
                    # Save to session state
                    st.session_state.ds_target_col    = selected_target
                    st.session_state.ds_feature_cols  = selected_features
                    st.session_state.ds_problem_type  = selected_problem

                    st.markdown(
                        f'<div class="ok-box">✅ <strong>Configuration Ready</strong><br>'
                        f'<strong>Target:</strong> <code>{selected_target}</code> &nbsp;|&nbsp; '
                        f'<strong>Features:</strong> {len(selected_features)} columns selected &nbsp;|&nbsp; '
                        f'<strong>Problem:</strong> {selected_problem}</div>',
                        unsafe_allow_html=True,
                    )

                    with st.expander("📋 Selected Feature Summary"):
                        feat_summary = []
                        for fc in selected_features:
                            fp = col_profiles[fc]
                            feat_summary.append({
                                "Feature": fc,
                                "Type":    fp["kind"].title(),
                                "Dtype":   fp["dtype"],
                                "Unique":  fp["n_unique"],
                                "Null %":  f"{fp['null_pct']:.1f}%",
                            })
                        st.dataframe(pd.DataFrame(feat_summary), use_container_width=True, hide_index=True)

    # -----------------------------------------------------------------------
    # Tab 3: Machine Learning Model (Classification & Regression Engine)
    # -----------------------------------------------------------------------
    with tab_model:
        if st.session_state.dataset_source == "upload":
            if st.session_state.uploaded_df is None:
                st.info("📂 No uploaded dataset found. Please upload a CSV file in the **Dataset Overview** tab first.")
            else:
                up_df = st.session_state.uploaded_df
                up_filename = st.session_state.uploaded_filename or "Uploaded Dataset"

                target_col = st.session_state.get("ds_target_col")
                feature_cols = st.session_state.get("ds_feature_cols")
                problem_type = st.session_state.get("ds_problem_type", "Classification")

                if not target_col or not feature_cols:
                    st.warning("⚠️ Target column and features have not been configured yet. Please visit the **Dataset Inspection** tab to finalize your target and feature selection.")
                else:
                    st.markdown(
                        f"""
                        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem;">
                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                                <div>
                                    <h3 style="margin: 0; color: #f8fafc; font-size: 1.25rem;">
                                        🤖 ML Pipeline: <code>{up_filename}</code>
                                    </h3>
                                    <p style="margin: 0.2rem 0 0; color: #94a3b8; font-size: 0.88rem;">
                                        Mode: <strong style="color: #818cf8;">{problem_type}</strong> &nbsp;|&nbsp; 
                                        Target: <code>{target_col}</code> &nbsp;|&nbsp; 
                                        Features: <strong>{len(feature_cols)}</strong> columns
                                    </p>
                                </div>
                                <div>
                                    <span class="status-pill"><span class="status-dot"></span> Pipeline Configured</span>
                                </div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Training options
                    opt_col1, opt_col2 = st.columns([2, 1])
                    with opt_col1:
                        test_split_ratio = st.slider(
                            "Test Dataset Split Ratio",
                            min_value=0.10,
                            max_value=0.40,
                            value=0.20,
                            step=0.05,
                            help="Proportion of the dataset reserved for unbiased model evaluation (test partition).",
                            key="ml_test_split_ratio",
                        )

                    # Trigger Training or Display Results
                    train_needed = (
                        st.session_state.uploaded_model_results is None
                        or st.session_state.uploaded_model_results.get("target_col") != target_col
                        or set(st.session_state.uploaded_model_results.get("feature_cols", [])) != set(feature_cols)
                        or st.session_state.uploaded_model_results.get("problem_type") != problem_type
                    )

                    if train_needed:
                        st.info(f"💡 Ready to train 3 {problem_type.lower()} models on `{up_filename}`. Click the button below to execute.")
                        if st.button("🚀 Train Machine Learning Pipeline", key="btn_train_uploaded_pipeline", use_container_width=True, type="primary"):
                            with st.spinner(f"Training {problem_type} models with automated preprocessing pipelines..."):
                                try:
                                    if train_and_evaluate_pipeline is None:
                                        st.error("❌ The pipeline training function `train_and_evaluate_pipeline` could not be loaded from `src.model_training`. Please verify `src/model_training.py`.")
                                        st.stop()
                                    res_pipeline = train_and_evaluate_pipeline(
                                        df=up_df,
                                        target_col=target_col,
                                        feature_cols=feature_cols,
                                        problem_type=problem_type,
                                        test_size=test_split_ratio,
                                    )
                                    st.session_state.uploaded_model_results = res_pipeline
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error during model training: {e}")
                    else:
                        res = st.session_state.uploaded_model_results
                        best_name = res["best_name"]
                        best_metrics = res["best_metrics"]
                        best_pipeline = res["best_pipeline"]
                        classes = res.get("classes")
                        target_encoder = res.get("target_encoder")
                        n_train = len(res["X_train"])
                        n_test = len(res["X_test"])

                        # ── Model Comparison Table ───────────────────────
                        st.subheader(f"📊 Model Comparison ({problem_type})")
                        if pipeline_results_to_dataframe is not None:
                            comp_df = pipeline_results_to_dataframe(res["results"], problem_type=problem_type)
                        else:
                            comp_df = pd.DataFrame(res["results"])

                        if problem_type == "Classification":
                            st.dataframe(
                                comp_df.style.highlight_max(
                                    subset=["Accuracy", "Precision", "Recall", "F1 Score"],
                                    color="#667eea",
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:  # Regression
                            st.dataframe(
                                comp_df.style.highlight_max(
                                    subset=["R² Score"],
                                    color="#667eea",
                                ).highlight_min(
                                    subset=["RMSE", "MAE", "MSE"],
                                    color="#38bdf8",
                                ),
                                use_container_width=True,
                                hide_index=True,
                            )

                        st.divider()

                        # ── Best Model Metrics & Visualisation ────────────
                        st.subheader(f"🏆 Best Model: {best_name}")
                        st.caption(f"Fitted on {n_train:,} training records, evaluated on {n_test:,} test records without data leakage.")

                        if problem_type == "Classification":
                            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                            m_col1.metric("Accuracy", f"{best_metrics['accuracy']:.2%}")
                            m_col2.metric("Precision", f"{best_metrics['precision']:.2%}")
                            m_col3.metric("Recall", f"{best_metrics['recall']:.2%}")
                            m_col4.metric("F1 Score", f"{best_metrics['f1']:.2%}")

                            st.markdown("#### Confusion Matrix")
                            fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
                            fig_cm.patch.set_facecolor("#0e1117")
                            ax_cm.set_facecolor("#0e1117")

                            tick_labels = [str(c) for c in classes] if classes else "auto"
                            sns.heatmap(
                                best_metrics["confusion_matrix"],
                                annot=True,
                                fmt="d",
                                cmap="RdPu",
                                xticklabels=tick_labels,
                                yticklabels=tick_labels,
                                ax=ax_cm,
                                cbar_kws={"shrink": 0.8},
                            )
                            ax_cm.set_xlabel("Predicted", color="#ccc")
                            ax_cm.set_ylabel("Actual", color="#ccc")
                            ax_cm.tick_params(colors="#ccc")
                            for spine in ax_cm.spines.values():
                                spine.set_color("#333")
                            plt.tight_layout()
                            st.pyplot(fig_cm)

                        else:  # Regression
                            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                            m_col1.metric("R² Score", f"{best_metrics['r2']:.4f}")
                            m_col2.metric("RMSE", f"{best_metrics['rmse']:,.4f}")
                            m_col3.metric("MAE", f"{best_metrics['mae']:,.4f}")
                            m_col4.metric("MSE", f"{best_metrics['mse']:,.4f}")

                            st.markdown("#### Actual vs. Predicted Parity Plot")
                            fig_parity, ax_parity = plt.subplots(figsize=(6, 4.5))
                            fig_parity.patch.set_facecolor("#0e1117")
                            ax_parity.set_facecolor("#0e1117")

                            y_test_arr = np.array(res["y_test"])
                            y_pred_arr = np.array(best_metrics["y_pred"])

                            ax_parity.scatter(
                                y_test_arr, y_pred_arr,
                                color="#6366f1", alpha=0.75, edgecolors="none", s=45, label="Test Predictions"
                            )
                            min_val = min(y_test_arr.min(), y_pred_arr.min())
                            max_val = max(y_test_arr.max(), y_pred_arr.max())
                            ax_parity.plot(
                                [min_val, max_val], [min_val, max_val],
                                color="#f472b6", linestyle="--", linewidth=1.5, label="Perfect Prediction (y = x)"
                            )
                            ax_parity.set_xlabel("Actual Values", color="#ccc")
                            ax_parity.set_ylabel("Predicted Values", color="#ccc")
                            ax_parity.tick_params(colors="#ccc")
                            for spine in ax_parity.spines.values():
                                spine.set_color("#333")
                            ax_parity.legend(fontsize=8, facecolor="#0e1117", edgecolor="#333", labelcolor="#ccc")
                            plt.tight_layout()
                            st.pyplot(fig_parity)

                        st.divider()

                        # ── Pipeline Architecture & Re-Train ─────────────
                        with st.expander("🛡️ Pipeline Architecture & Leakage Prevention Details"):
                            st.markdown(
                                f"""
                                - **Data Leakage Safeguard:** Transformations (imputation, scaling, one-hot encoding) are encapsulated within an `sklearn.pipeline.Pipeline` and fit strictly on training folds.
                                - **Numerical Features ({len(res['numeric_features'])}):** Median Imputation $\\rightarrow$ StandardScaler.
                                - **Categorical Features ({len(res['categorical_features'])}):** Most Frequent Imputation $\\rightarrow$ OneHotEncoder (unknown categories ignored gracefully).
                                - **Model Persistence:** Best pipeline saved to `models/uploaded_pipeline.joblib` and metadata to `models/uploaded_metadata.joblib`.
                                """
                            )
                            if st.button("🔄 Re-Train Pipeline", key="btn_retrain_uploaded"):
                                st.session_state.uploaded_model_results = None
                                st.rerun()

                        st.divider()

                        # ── Interactive Predictions on New Inputs ────────
                        st.subheader(f"🔮 Make Predictions on New Data ({problem_type})")
                        st.caption("Provide feature values below to generate an instantaneous prediction using the best trained pipeline.")

                        with st.form("uploaded_prediction_form"):
                            n_feat_cols = min(3, len(feature_cols))
                            layout_cols = st.columns(n_feat_cols)
                            user_inputs = {}

                            for i, f_col in enumerate(feature_cols):
                                with layout_cols[i % n_feat_cols]:
                                    f_series = up_df[f_col]
                                    if f_col in res["numeric_features"]:
                                        f_numeric = pd.to_numeric(f_series, errors="coerce").dropna()
                                        f_min = float(f_numeric.min()) if not f_numeric.empty else 0.0
                                        f_max = float(f_numeric.max()) if not f_numeric.empty else 100.0
                                        f_mean = float(f_numeric.mean()) if not f_numeric.empty else (f_min + f_max) / 2
                                        f_step = round((f_max - f_min) / 100.0, 2) if f_max > f_min else 0.1
                                        user_inputs[f_col] = st.number_input(
                                            f_col.replace("_", " ").title(),
                                            value=round(f_mean, 2),
                                            step=f_step if f_step > 0 else 0.1,
                                            key=f"input_{f_col}",
                                        )
                                    else:
                                        opts = [str(v) for v in f_series.dropna().unique()]
                                        if not opts:
                                            opts = ["Standard"]
                                        user_inputs[f_col] = st.selectbox(
                                            f_col.replace("_", " ").title(),
                                            options=opts,
                                            key=f"input_{f_col}",
                                        )

                            submitted_pred = st.form_submit_button("🔮 Generate Prediction", use_container_width=True)

                        if submitted_pred:
                            input_df = pd.DataFrame([user_inputs])
                            raw_prediction = best_pipeline.predict(input_df)[0]

                            st.markdown("---")
                            st.subheader("🎯 Prediction Output")

                            if problem_type == "Classification":
                                if target_encoder is not None:
                                    predicted_label = target_encoder.inverse_transform([raw_prediction])[0]
                                else:
                                    predicted_label = str(raw_prediction)

                                pred_c1, pred_c2 = st.columns(2)
                                with pred_c1:
                                    st.markdown(
                                        f"""
                                        <div style="text-align: center; padding: 1.2rem; border-radius: 12px; background: #1e293b; border: 2px solid #6366f1;">
                                            <h4 style="margin: 0; color: #94a3b8; font-size: 0.95rem;">Predicted Class</h4>
                                            <p style="font-size: 2.2rem; font-weight: 800; margin: 0.3rem 0; color: #818cf8;">{predicted_label}</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                with pred_c2:
                                    if hasattr(best_pipeline, "predict_proba"):
                                        probabilities = best_pipeline.predict_proba(input_df)[0]
                                        confidence = float(np.max(probabilities))
                                        st.markdown(
                                            f"""
                                            <div style="text-align: center; padding: 1.2rem; border-radius: 12px; background: #1e293b; border: 2px solid #10b981;">
                                                <h4 style="margin: 0; color: #94a3b8; font-size: 0.95rem;">Prediction Confidence</h4>
                                                <p style="font-size: 2.2rem; font-weight: 800; margin: 0.3rem 0; color: #34d399;">{confidence:.1%}</p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                        proba_df = pd.DataFrame({
                                            "Class": [str(c) for c in classes],
                                            "Probability": probabilities
                                        })
                                        st.caption("Class Probability Breakdown:")
                                        st.dataframe(proba_df.style.format({"Probability": "{:.2%}"}), use_container_width=True, hide_index=True)
                                    else:
                                        st.markdown(
                                            f"""
                                            <div style="text-align: center; padding: 1.2rem; border-radius: 12px; background: #1e293b; border: 2px solid #334155;">
                                                <h4 style="margin: 0; color: #94a3b8; font-size: 0.95rem;">Model Type</h4>
                                                <p style="font-size: 1.5rem; font-weight: 700; margin: 0.3rem 0; color: #cbd5e1;">{best_name}</p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                st.session_state.latest_model_prediction = {
                                    "predicted_label": f"{pred_numeric:,.2f}",
                                    "confidence": None,
                                    "probabilities": None,
                                    "inputs": dict(user_inputs),
                                    "problem_type": "Regression",
                                    "target_col": target_col,
                                    "model_name": best_name,
                                }

                            else:  # Regression
                                pred_numeric = float(raw_prediction)
                                reg_c1, reg_c2 = st.columns(2)
                                with reg_c1:
                                    st.markdown(
                                        f"""
                                        <div style="text-align: center; padding: 1.2rem; border-radius: 12px; background: #1e293b; border: 2px solid #6366f1;">
                                            <h4 style="margin: 0; color: #94a3b8; font-size: 0.95rem;">Predicted Value ({target_col})</h4>
                                            <p style="font-size: 2.2rem; font-weight: 800; margin: 0.3rem 0; color: #818cf8;">{pred_numeric:,.4g}</p>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                                with reg_c2:
                                    y_all_numeric = pd.to_numeric(up_df[target_col], errors="coerce").dropna()
                                    if not y_all_numeric.empty:
                                        pct_rank = float((y_all_numeric <= pred_numeric).mean() * 100)
                                        y_min, y_mean, y_max = y_all_numeric.min(), y_all_numeric.mean(), y_all_numeric.max()
                                        st.markdown(
                                            f"""
                                            <div style="text-align: center; padding: 1.2rem; border-radius: 12px; background: #1e293b; border: 2px solid #334155;">
                                                <h4 style="margin: 0; color: #94a3b8; font-size: 0.95rem;">Relative Dataset Standing</h4>
                                                <p style="font-size: 2.2rem; font-weight: 800; margin: 0.3rem 0; color: #38bdf8;">~{pct_rank:.0f}th percentile</p>
                                                <p style="margin: 0; font-size: 0.76rem; color: #64748b;">Range: [{y_min:,.3g} to {y_max:,.3g}], Mean: {y_mean:,.3g}</p>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                st.session_state.latest_model_prediction = {
                                    "predicted_label": f"{pred_numeric:,.2f}",
                                    "confidence": None,
                                    "probabilities": None,
                                    "inputs": dict(user_inputs),
                                    "problem_type": "Regression",
                                    "target_col": target_col,
                                    "model_name": best_name,
                                }

        else:  # Demo Student Dataset Mode
            st.markdown(
                """
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 12px; padding: 1.2rem; margin-bottom: 1.2rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <h3 style="margin: 0; color: #f8fafc; font-size: 1.25rem;">
                                🤖 ML Pipeline: Student Performance Dataset (Demo)
                            </h3>
                            <p style="margin: 0.2rem 0 0; color: #94a3b8; font-size: 0.88rem;">
                                Mode: <strong style="color: #818cf8;">Classification</strong> &nbsp;|&nbsp; 
                                Target: <code>result</code> &nbsp;|&nbsp; 
                                Features: <strong>6</strong> academic indicators
                            </p>
                        </div>
                        <div>
                            <span class="status-pill"><span class="status-dot"></span> Benchmark Active</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.subheader("🤖 Model Training & Comparison")
            comparison_df = results_to_dataframe(results)
            st.dataframe(
                comparison_df.style.highlight_max(
                    subset=["Accuracy", "Precision", "Recall", "F1 Score"],
                    color="#667eea",
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.divider()

            st.subheader(f"🏆 Best Selected Model — {best_name}")

            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.metric("Accuracy", f"{best_metrics['accuracy']:.2%}")
            col_b.metric("Precision", f"{best_metrics['precision']:.2%}")
            col_c.metric("Recall", f"{best_metrics['recall']:.2%}")
            col_d.metric("F1 Score", f"{best_metrics['f1']:.2%}")

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
            for spine in ax_cm.spines.values():
                spine.set_color("#333")
            plt.tight_layout()
            st.pyplot(fig_cm)

            st.divider()

            # ── Interactive Prediction on New Inputs for Demo Data ────────
            st.subheader("🔮 Make Predictions on New Student Data")
            st.caption("Enter hypothetical student values to generate a model prediction and confidence score.")

            with st.form("demo_ml_prediction_form"):
                r1c1, r1c2, r1c3 = st.columns(3)
                with r1c1:
                    d_study = st.slider("Study Hours", 0.0, 12.0, 5.0, 0.5, key="demo_ml_study")
                with r1c2:
                    d_attend = st.slider("Attendance (%)", 0.0, 100.0, 75.0, 1.0, key="demo_ml_attend")
                with r1c3:
                    d_prev = st.slider("Previous Scores", 0.0, 100.0, 60.0, 1.0, key="demo_ml_prev")

                r2c1, r2c2, r2c3 = st.columns(3)
                with r2c1:
                    d_sleep = st.slider("Sleep Hours", 3.0, 12.0, 7.0, 0.5, key="demo_ml_sleep")
                with r2c2:
                    d_assign = st.slider("Assignments Completed", 0, 10, 5, key="demo_ml_assign")
                with r2c3:
                    d_partic = st.slider("Class Participation", 0, 10, 5, key="demo_ml_partic")

                d_submit = st.form_submit_button("🔮 Predict Outcome", use_container_width=True)

            if d_submit:
                raw_input = np.array([[d_study, d_attend, d_prev, d_sleep, d_assign, d_partic]])
                scaled_input = scaler.transform(raw_input)
                d_pred = best_model.predict(scaled_input)[0]
                d_proba = best_model.predict_proba(scaled_input)[0]
                d_label = le.inverse_transform([d_pred])[0]
                d_conf = float(d_proba[d_pred])

                st.markdown("---")
                st.subheader("🎯 Prediction Output")
                p1, p2 = st.columns(2)
                pred_color = "#667eea" if d_label == "PASS" else "#e74c6f"
                with p1:
                    st.markdown(
                        f"""
                        <div style="text-align: center; padding: 1.2rem; border-radius: 12px; background: #1e293b; border: 2px solid {pred_color};">
                            <h4 style="margin: 0; color: #94a3b8; font-size: 0.95rem;">Prediction</h4>
                            <p style="font-size: 2.2rem; font-weight: 800; margin: 0.3rem 0; color: {pred_color};">{d_label}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with p2:
                    st.markdown(
                        f"""
                        <div style="text-align: center; padding: 1.2rem; border-radius: 12px; background: #1e293b; border: 2px solid {pred_color};">
                            <h4 style="margin: 0; color: #94a3b8; font-size: 0.95rem;">Confidence</h4>
                            <p style="font-size: 2.2rem; font-weight: 800; margin: 0.3rem 0; color: {pred_color};">{d_conf:.1%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


    # -----------------------------------------------------------------------
    # Tab 3: Explainable AI
    # -----------------------------------------------------------------------
    with tab_shap:
        if st.session_state.dataset_source == "upload":
            up_filename = st.session_state.get("uploaded_filename") or "Uploaded Dataset"
            up_df = st.session_state.get("uploaded_df")

            if up_df is None:
                st.markdown(
                    f"""
                    <div style="background: #1e293b; border: 1px dashed #64748b; border-radius: 12px; padding: 2.5rem; text-align: center; margin: 1.5rem 0;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">📂</div>
                        <h3 style="color: #f8fafc; margin: 0 0 0.5rem 0;">No Dataset Uploaded Yet</h3>
                        <p style="color: #94a3b8; max-width: 580px; margin: 0 auto; font-size: 0.95rem;">
                            Please upload a CSV file in the <b>📊 Dataset Overview</b> tab to inspect data and train machine learning models.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif st.session_state.uploaded_model_results is None:
                st.markdown(
                    f"""
                    <div style="background: #1e293b; border: 1px dashed #64748b; border-radius: 12px; padding: 2.5rem; text-align: center; margin: 1.5rem 0;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">🤖</div>
                        <h3 style="color: #f8fafc; margin: 0 0 0.5rem 0;">Trained Model Required for SHAP Explanations</h3>
                        <p style="color: #94a3b8; max-width: 600px; margin: 0 auto 1.5rem auto; font-size: 0.95rem;">
                            SHAP calculates game-theoretic Shapley values by interpreting a fitted machine learning pipeline. Please navigate to the <b>🤖 Machine Learning Model</b> tab and train a model on <code>{up_filename}</code> first.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                up_res = st.session_state.uploaded_model_results
                up_best_name = up_res["best_name"]
                up_pipeline = up_res["best_pipeline"]
                up_prob_type = up_res["problem_type"]
                up_target_col = up_res["target_col"]
                up_feature_cols = up_res["feature_cols"]
                up_numeric_feats = up_res["numeric_features"]
                up_cat_feats = up_res["categorical_features"]
                up_classes = up_res.get("classes")
                up_target_enc = up_res.get("target_encoder")
                up_X_train = up_res["X_train"]
                up_X_test = up_res["X_test"]
                up_y_test = up_res["y_test"]

                # ── Header & Telemetry ────────────────────────────────────
                st.markdown(
                    f"""
                    <div style="text-align:center; padding:0.5rem 0;">
                        <h2 style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                                   font-size: 2rem; font-weight: 800;">🔍 Explainable AI (SHAP)</h2>
                        <p style="color:#94a3b8; font-size:1rem; margin-top:-0.4rem;">
                            Model-agnostic game-theoretic feature attributions for <b>{up_best_name}</b> on <code>{up_filename}</code>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f"""
                    <div style="display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin: 0.8rem 0 1.5rem 0;">
                        <span class="status-pill" style="border-color: #6366f1; color: #a5b4fc;">
                            🎯 Problem: <b>{up_prob_type}</b>
                        </span>
                        <span class="status-pill" style="border-color: #10b981; color: #6ee7b7;">
                            🏷️ Target: <b>{up_target_col}</b>
                        </span>
                        <span class="status-pill" style="border-color: #ec4899; color: #f472b6;">
                            🏆 Model: <b>{up_best_name}</b>
                        </span>
                        <span class="status-pill" style="border-color: #f59e0b; color: #fcd34d;">
                            📊 Features: <b>{len(up_feature_cols)} ({len(up_numeric_feats)} Num, {len(up_cat_feats)} Cat)</b>
                        </span>
                        <span class="status-pill" style="border-color: #38bdf8; color: #7dd3fc;">
                            🧪 Test Samples: <b>{len(up_X_test):,}</b>
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # ── Scientific Distinction Callout ────────────────────────
                st.warning(
                    "⚠️ **Scientific Advisory: SHAP Statistical Attributions vs. Causal Effects**\n\n"
                    "• **Associational Attribution:** SHAP (SHapley Additive exPlanations) values quantify how each input feature shifts the trained model's output score away from the dataset baseline expectation ($E[f(X)]$). It identifies **predictive correlations** learned by this specific statistical pipeline.\n\n"
                    "• **Not Real-World Causality:** A high positive SHAP score does **not** prove that changing this feature in reality will causally produce that outcome shift. Direct interventions and policy decisions require causal identification (e.g. Causal DAGs, do-calculus, or randomized controlled trials) to account for structural dependencies and unobserved confounding."
                )

                st.divider()

                # ── 1. Global Feature Importance ──────────────────────────
                st.subheader("📊 1. Model-Native Global Feature Importance")
                st.caption("Extracted directly from the model's coefficients or tree splits (normalized [0, 1]).")

                imp_pipe_df = get_pipeline_global_importance(up_pipeline, up_cat_feats)
                gcol_chart, gcol_table = st.columns([2, 1])
                with gcol_chart:
                    fig_pipe_imp = plot_global_importance(imp_pipe_df, feature_col="Feature", val_col="Relative_Importance")
                    st.pyplot(fig_pipe_imp)
                with gcol_table:
                    st.dataframe(
                        imp_pipe_df[["Feature", "Relative_Importance"]].rename(columns={"Relative_Importance": "Importance"}),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.divider()

                # ── 2. SHAP Feature Importance ────────────────────────────
                st.subheader("🧠 2. SHAP Global Feature Attribution")
                st.caption(
                    "SHAP computes the marginal contribution of each feature across the test dataset "
                    "using game-theoretic Shapley values."
                )

                shap_cache_key = (up_best_name, up_target_col, tuple(sorted(up_feature_cols)), up_prob_type, len(up_X_test))
                if (
                    st.session_state.uploaded_shap_cache is None
                    or st.session_state.uploaded_shap_cache.get("key") != shap_cache_key
                ):
                    with st.spinner("Computing global SHAP values across evaluation set..."):
                        try:
                            exp, expl, expl_t = compute_pipeline_shap(
                                up_pipeline,
                                up_X_test,
                                categorical_features=up_cat_feats,
                                problem_type=up_prob_type,
                            )
                            st.session_state.uploaded_shap_cache = {
                                "key": shap_cache_key,
                                "explanation": exp,
                                "explainer": expl,
                                "expl_type": expl_t,
                                "error": None,
                            }
                        except Exception as e:
                            st.session_state.uploaded_shap_cache = {
                                "key": shap_cache_key,
                                "explanation": None,
                                "explainer": None,
                                "expl_type": None,
                                "error": str(e),
                            }

                cached_shap = st.session_state.uploaded_shap_cache
                if cached_shap.get("error"):
                    st.info(
                        f"ℹ️ **SHAP Explainer Notice:** Could not automatically compute global SHAP values with standard explainers: "
                        f"`{cached_shap['error']}`. Model-native feature importances are available above."
                    )
                elif cached_shap.get("explanation") is not None:
                    exp = cached_shap["explanation"]
                    expl = cached_shap["explainer"]
                    expl_t = cached_shap["expl_type"]
                    base_val_fmt = f"{float(np.ravel(exp.base_values)[0]):,.3f}" if isinstance(exp.base_values, (int, float, np.number, np.ndarray)) else str(exp.base_values)

                    st.markdown(
                        f"""
                        <div style="margin-bottom: 0.8rem;">
                            <span class="status-pill" style="border-color: #10b981; color: #6ee7b7; font-size: 0.8rem;">
                                ⚡ Explainer Active: <b>{expl_t}</b>
                            </span>
                            <span style="font-size: 0.85rem; color: #94a3b8; margin-left: 10px;">
                                Baseline Expected Value: <code>{base_val_fmt}</code>
                            </span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    tab_beeswarm, tab_shap_table = st.tabs(["🐝 SHAP Summary Plot", "📋 Mean |SHAP| Table"])
                    with tab_beeswarm:
                        fig_shap = plot_shap_summary(exp)
                        st.pyplot(fig_shap)
                    with tab_shap_table:
                        shap_imp_df = get_pipeline_shap_importance(exp)
                        st.dataframe(shap_imp_df, use_container_width=True, hide_index=True)

                st.divider()

                # ── 3. Individual Prediction Explanations ──────────────────
                st.subheader("🎯 3. Individual Prediction Explanation & Feature Breakdown")
                st.caption("Inspect how features drive an individual instance's prediction upward or downward.")

                input_mode = st.radio(
                    "Prediction Input Method:",
                    ["🧪 Inspect Sample from Test Set", "✏️ Enter Custom Feature Values"],
                    horizontal=True,
                    key="shap_input_mode_selector",
                )

                user_input_record = None
                ground_truth_display = None

                if input_mode == "🧪 Inspect Sample from Test Set":
                    sample_idx = st.slider(
                        f"Select Test Record Index (0 to {len(up_X_test) - 1}):",
                        min_value=0,
                        max_value=len(up_X_test) - 1,
                        value=0,
                        key="shap_sample_idx_slider",
                    )
                    user_input_record = up_X_test.iloc[sample_idx].to_dict()

                    # Ground truth display
                    raw_gt = up_y_test.iloc[sample_idx] if hasattr(up_y_test, "iloc") else up_y_test[sample_idx]
                    if up_prob_type == "Classification" and up_target_enc:
                        gt_str = up_target_enc.inverse_transform([raw_gt])[0] if isinstance(raw_gt, (int, np.integer)) else str(raw_gt)
                    else:
                        gt_str = f"{float(raw_gt):,.2f}" if isinstance(raw_gt, (int, float, np.number)) else str(raw_gt)
                    ground_truth_display = gt_str

                    # Show preview of record
                    with st.expander("🔍 View Selected Test Sample Data", expanded=False):
                        st.json(user_input_record)

                    explain_triggered = True

                else:  # Custom feature values
                    with st.form("custom_shap_prediction_form"):
                        st.markdown("##### ✏️ Enter Feature Values")
                        cols = st.columns(3)
                        custom_inputs = {}

                        for idx, feat in enumerate(up_feature_cols):
                            col_widget = cols[idx % 3]
                            with col_widget:
                                if feat in up_numeric_feats:
                                    s_min = float(up_X_train[feat].min()) if not up_X_train[feat].empty else 0.0
                                    s_max = float(up_X_train[feat].max()) if not up_X_train[feat].empty else 100.0
                                    s_med = float(up_X_train[feat].median()) if not up_X_train[feat].empty else 50.0
                                    step_val = (s_max - s_min) / 100.0 if (s_max > s_min) else 1.0
                                    custom_inputs[feat] = st.number_input(
                                        f"{feat.replace('_', ' ').title()}",
                                        value=s_med,
                                        step=step_val,
                                        key=f"shap_inp_{feat}",
                                    )
                                else:  # Categorical
                                    options = sorted(list(up_X_train[feat].dropna().unique()))
                                    default_idx = 0 if options else 0
                                    custom_inputs[feat] = st.selectbox(
                                        f"{feat.replace('_', ' ').title()}",
                                        options=options,
                                        index=default_idx,
                                        key=f"shap_inp_{feat}",
                                    )

                        explain_triggered = st.form_submit_button("🔮 Predict & Explain Custom Input", type="primary", use_container_width=True)
                        if explain_triggered:
                            user_input_record = custom_inputs

                if explain_triggered and user_input_record:
                    # Get explainer
                    cur_explainer = cached_shap.get("explainer") if cached_shap else None
                    if cur_explainer is None:
                        cur_explainer, _ = build_pipeline_shap_explainer(up_pipeline, up_X_train, problem_type=up_prob_type)

                    with st.spinner("Generating SHAP feature contributions..."):
                        try:
                            ind_res = explain_pipeline_prediction(
                                pipeline=up_pipeline,
                                explainer=cur_explainer,
                                input_values=user_input_record,
                                feature_cols=up_feature_cols,
                                categorical_features=up_cat_feats,
                                problem_type=up_prob_type,
                                target_encoder=up_target_enc,
                                target_col=up_target_col,
                            )
                        except Exception as e:
                            st.error(f"❌ Failed to generate individual SHAP explanation: {e}")
                            ind_res = None

                    if ind_res is not None:
                        st.markdown("---")
                        st.subheader("🎯 Prediction Outcome & SHAP Attribution")

                        # Output metric cards
                        if up_prob_type == "Classification":
                            c1, c2, c3, c4 = st.columns(4)
                            pred_lbl = ind_res["prediction_label"]
                            c_color = "#10b981" if pred_lbl in ["1", "PASS", "True", "Yes", "positive"] else "#6366f1"
                            with c1:
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; padding: 1.1rem; border-radius: 12px; background: #1e293b; border: 2px solid {c_color};">
                                        <h4 style="margin: 0; color: #94a3b8; font-size: 0.85rem;">Predicted Class</h4>
                                        <p style="font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0; color: {c_color};">{pred_lbl}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            with c2:
                                prob_val = ind_res["prediction_proba"]
                                prob_str = f"{prob_val:.1%}" if prob_val is not None else "N/A"
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; padding: 1.1rem; border-radius: 12px; background: #1e293b; border: 2px solid #38bdf8;">
                                        <h4 style="margin: 0; color: #94a3b8; font-size: 0.85rem;">Confidence</h4>
                                        <p style="font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0; color: #38bdf8;">{prob_str}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            with c3:
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; padding: 1.1rem; border-radius: 12px; background: #1e293b; border: 2px solid #64748b;">
                                        <h4 style="margin: 0; color: #94a3b8; font-size: 0.85rem;">Baseline $E[f(X)]$</h4>
                                        <p style="font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0; color: #cbd5e1;">{ind_res['base_value']:.2f}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            with c4:
                                shift_val = ind_res["net_shap_shift"]
                                s_col = "#10b981" if shift_val >= 0 else "#ef4444"
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; padding: 1.1rem; border-radius: 12px; background: #1e293b; border: 2px solid {s_col};">
                                        <h4 style="margin: 0; color: #94a3b8; font-size: 0.85rem;">Net SHAP Shift</h4>
                                        <p style="font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0; color: {s_col};">{shift_val:+.3f}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                        else:  # Regression
                            c1, c2, c3, c4 = st.columns(4)
                            pred_v = ind_res["prediction_value"]
                            base_v = ind_res["base_value"]
                            net_diff = pred_v - base_v
                            diff_col = "#10b981" if net_diff >= 0 else "#ef4444"
                            with c1:
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; padding: 1.1rem; border-radius: 12px; background: #1e293b; border: 2px solid #6366f1;">
                                        <h4 style="margin: 0; color: #94a3b8; font-size: 0.85rem;">Predicted {up_target_col}</h4>
                                        <p style="font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0; color: #818cf8;">{pred_v:,.2f}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            with c2:
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; padding: 1.1rem; border-radius: 12px; background: #1e293b; border: 2px solid #64748b;">
                                        <h4 style="margin: 0; color: #94a3b8; font-size: 0.85rem;">Expected Baseline</h4>
                                        <p style="font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0; color: #cbd5e1;">{base_v:,.2f}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            with c3:
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; padding: 1.1rem; border-radius: 12px; background: #1e293b; border: 2px solid {diff_col};">
                                        <h4 style="margin: 0; color: #94a3b8; font-size: 0.85rem;">Baseline Delta</h4>
                                        <p style="font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0; color: {diff_col};">{net_diff:+,.2f}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )
                            with c4:
                                st.markdown(
                                    f"""
                                    <div style="text-align: center; padding: 1.1rem; border-radius: 12px; background: #1e293b; border: 2px solid #38bdf8;">
                                        <h4 style="margin: 0; color: #94a3b8; font-size: 0.85rem;">Net SHAP Sum</h4>
                                        <p style="font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0; color: #38bdf8;">{ind_res['net_shap_shift']:+,.2f}</p>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        if ground_truth_display is not None:
                            st.caption(f"📌 **Actual Ground Truth Value in Test Dataset:** `{ground_truth_display}`")

                        # ── Human-Readable Narrative ───────────────────────
                        st.markdown("##### 📝 Human-Readable Explanation")
                        st.info(ind_res["narrative"])

                        # ── Feature Contribution Visualizations ───────────
                        st.markdown("##### 📈 Feature Contribution Visualizations")
                        vis_tab1, vis_tab2 = st.tabs(["📊 Diverging Contribution Chart", "🌊 SHAP Waterfall Plot"])

                        with vis_tab1:
                            fig_contrib = plot_feature_contributions(
                                ind_res["feature_effects"],
                                base_value=ind_res["base_value"],
                                pred_value=ind_res["prediction_value"],
                                problem_type=up_prob_type,
                                target_label=up_target_col,
                            )
                            st.pyplot(fig_contrib)

                        with vis_tab2:
                            fig_waterfall = plot_shap_waterfall(ind_res["single_explanation"])
                            if fig_waterfall is not None:
                                st.pyplot(fig_waterfall)
                            else:
                                st.caption("Waterfall plot unavailable for this model configuration. See diverging contribution chart above.")

                        # ── Positive vs. Negative Breakdown Tables ────────
                        st.markdown("##### ⚖️ Feature Contribution Breakdown")
                        pos_col, neg_col = st.columns(2)

                        with pos_col:
                            st.markdown("###### ⬆️ Positive Drivers (Pushing Upward)")
                            pos_df = ind_res["positive_factors"][["Feature", "Input_Value", "SHAP"]].copy()
                            if not pos_df.empty:
                                pos_df["SHAP"] = pos_df["SHAP"].round(4)
                                st.dataframe(pos_df.rename(columns={"Input_Value": "Input Value"}), use_container_width=True, hide_index=True)
                            else:
                                st.caption("No positive feature contributions for this prediction.")

                        with neg_col:
                            st.markdown("###### ⬇️ Negative Drivers (Pulling Downward)")
                            neg_df = ind_res["negative_factors"][["Feature", "Input_Value", "SHAP"]].copy()
                            if not neg_df.empty:
                                neg_df["SHAP"] = neg_df["SHAP"].round(4)
                                st.dataframe(neg_df.rename(columns={"Input_Value": "Input Value"}), use_container_width=True, hide_index=True)
                            else:
                                st.caption("No negative feature contributions for this prediction.")

                        with st.expander("📋 Full Transformed Feature Breakdown Table"):
                            full_df = ind_res["feature_effects"][["Feature", "Input_Value", "SHAP", "Abs_SHAP", "Raw_Feature"]].copy()
                            full_df["SHAP"] = full_df["SHAP"].round(4)
                            full_df["Abs_SHAP"] = full_df["Abs_SHAP"].round(4)
                            st.dataframe(
                                full_df.rename(columns={"Input_Value": "Input Value", "Abs_SHAP": "Absolute |SHAP|", "Raw_Feature": "Transformed Step"}),
                                use_container_width=True,
                                hide_index=True,
                            )

        else:
            # ── DEMO FLOW (Student Performance Dataset) ───────────────────
            st.markdown(
                """
                <div style="text-align:center; padding:0.5rem 0;">
                    <h2 style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                               font-size: 2rem; font-weight: 800;">🔍 Explainable AI (SHAP)</h2>
                    <p style="color:#888; font-size:1rem; margin-top:-0.4rem;">
                        Understand <em>why</em> the model makes its predictions using SHAP
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Scientific Disclaimer Callout
            st.warning(
                "⚠️ **Scientific Advisory: SHAP Attribution vs. Causal Effects**\n\n"
                "• **Statistical Attribution:** SHAP values quantify how each feature mathematically contributes to shifting the model's score away from the dataset baseline. It reflects **predictive correlations** in the trained model.\n\n"
                "• **Not Real Causality:** A high positive SHAP score does **not** prove that changing that feature in the real world will causally alter the actual outcome. Interventions require causal inference (e.g. Causal DAGs and do-calculus)."
            )

            st.divider()

            st.subheader("📊 Global Feature Importance")
            st.caption("Derived directly from the model's coefficients or tree splits.")

            importance_df = get_global_feature_importance(best_model, feature_names)

            col_chart, col_table = st.columns([2, 1])
            with col_chart:
                fig_imp = plot_global_importance(importance_df)
                st.pyplot(fig_imp)
            with col_table:
                st.dataframe(importance_df, use_container_width=True, hide_index=True)

            st.divider()

            st.subheader("🧠 SHAP Feature Importance")
            st.caption(
                "SHAP (SHapley Additive exPlanations) measures each feature's "
                "contribution to every individual prediction."
            )

            with st.spinner("Computing SHAP values … this may take a moment."):
                shap_explanation, shap_explainer = compute_shap_values(
                    best_model, X_test, feature_names
                )

            tab_beeswarm, tab_shap_table = st.tabs(["🐝 SHAP Summary Plot", "📋 Mean |SHAP| Table"])
            with tab_beeswarm:
                fig_shap = plot_shap_summary(shap_explanation)
                st.pyplot(fig_shap)
            with tab_shap_table:
                shap_imp_df = get_shap_importance(shap_explanation, feature_names)
                st.dataframe(shap_imp_df, use_container_width=True, hide_index=True)

            st.divider()

            st.subheader("🎯 Individual Prediction Explanation")
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

                st.subheader("📊 Feature Contribution Chart")
                fig_demo_contrib = plot_feature_contributions(
                    result["feature_effects"],
                    base_value=result["base_value"],
                    problem_type="Classification",
                    target_label="Student Outcome",
                )
                st.pyplot(fig_demo_contrib)

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

    # -----------------------------------------------------------------------
    # Tab 4: Causal Analysis
    # -----------------------------------------------------------------------
    with tab_causal:
        is_upload_causal = st.session_state.dataset_source == "upload"

        if is_upload_causal and st.session_state.uploaded_df is None:
            st.markdown(
                f"""
                <div style="background: #1e293b; border: 1px dashed #64748b; border-radius: 12px; padding: 2.5rem; text-align: center; margin: 1.5rem 0;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">🔬</div>
                    <h3 style="color: #f8fafc; margin: 0 0 0.5rem 0;">No Dataset Uploaded Yet</h3>
                    <p style="color: #94a3b8; max-width: 580px; margin: 0 auto; font-size: 0.95rem;">
                        Please upload a CSV file in the <b>📊 Dataset Overview</b> tab or switch to the Demo Dataset to explore and edit causal graphs.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # ── 1. Setup Graph Context & State ────────────────────────────
            if is_upload_causal:
                up_filename = st.session_state.get("uploaded_filename") or "Uploaded Dataset"
                active_df = st.session_state.uploaded_df
                target_col = st.session_state.ds_target_col or active_df.columns[-1]
                feature_cols = st.session_state.ds_feature_cols or [c for c in active_df.columns if c != target_col]
                session_graph_key = "uploaded_causal_graph"
                graph_source_name = f"Uploaded Dataset: {up_filename}"

                # Initialize graph state if absent
                if st.session_state.uploaded_causal_graph is None:
                    st.session_state.uploaded_causal_graph = generate_initial_causal_graph(
                        feature_cols=feature_cols,
                        target_col=target_col,
                        df=active_df,
                    )
            else:
                active_df = df
                target_col = "result"
                feature_cols = [c for c in df.columns if c != "result"]
                session_graph_key = "demo_causal_graph"
                graph_source_name = "Demo Dataset: Student Academic Performance"

                # Initialize demo graph state if absent
                if st.session_state.demo_causal_graph is None:
                    st.session_state.demo_causal_graph = graph_to_dict(build_causal_graph())

            # Load active DiGraph from session state
            G_active = dict_to_graph(st.session_state[session_graph_key])
            is_dag = nx.is_directed_acyclic_graph(G_active)

            # ── 2. Header & Telemetry ─────────────────────────────────────
            st.markdown(
                f"""
                <div style="text-align:center; padding:0.5rem 0;">
                    <h2 style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
                               -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                               font-size: 2rem; font-weight: 800;">🔬 Causal Analysis Workspace</h2>
                    <p style="color:#94a3b8; font-size:1rem; margin-top:-0.4rem;">
                        Qualitative Structural Causal Models (DAGs), Directed Hypotheses & Domain Knowledge Mapping
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            dag_status_pill = (
                '<span class="status-pill" style="border-color: #10b981; color: #6ee7b7;">✅ Valid Acyclic DAG</span>'
                if is_dag else
                '<span class="status-pill" style="border-color: #ef4444; color: #fca5a5;">⚠️ Contains Feedback Cycles</span>'
            )

            st.markdown(
                f"""
                <div style="display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; margin: 0.8rem 0 1.5rem 0;">
                    <span class="status-pill" style="border-color: #6366f1; color: #a5b4fc;">
                        📂 Context: <b>{graph_source_name}</b>
                    </span>
                    <span class="status-pill" style="border-color: #ec4899; color: #f472b6;">
                        🏷️ Target Node: <b>{target_col}</b>
                    </span>
                    <span class="status-pill" style="border-color: #f59e0b; color: #fcd34d;">
                        ⭕ Nodes: <b>{len(G_active.nodes())}</b>
                    </span>
                    <span class="status-pill" style="border-color: #38bdf8; color: #7dd3fc;">
                        ➡️ Directed Edges: <b>{len(G_active.edges())}</b>
                    </span>
                    {dag_status_pill}
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── 3. Mandatory Scientific Disclaimer Callout ────────────────
            st.warning(
                "⚠️ **Important Scientific Disclaimer: Qualitative Causal Hypotheses (Not Formal Causal Inference)**\n\n"
                "• **Human Assumptions & Hypotheses:** Every directed arrow ($A \\rightarrow B$) in this graph represents an **unproven qualitative hypothesis or domain assumption**. Generating or drawing an edge does **not** prove real-world causality.\n\n"
                "• **Correlation $\\neq$ Causation:** Observational datasets record statistical associations ($P(Y|X)$). Causal effects require interventional distributions ($P(Y|\\text{do}(X))$), randomized controlled trials (RCTs), instrumental variables, or verified structural equations.\n\n"
                "• **Review & Modify:** Users are encouraged to scrutinize, add, edit, or delete nodes and directed edges below to align with true domain knowledge and explore potential unobserved confounders."
            )

            if not is_dag:
                try:
                    cycles = list(nx.simple_cycles(G_active))
                    cycle_strs = [" ➔ ".join(c + [c[0]]) for c in cycles[:3]]
                    st.error(f"⚠️ **Feedback Cycle Detected:** Standard Pearl Causal DAGs require acyclicity. Detected cycle(s): {', '.join(cycle_strs)}. Please edit or reverse edges to restore acyclic structure.")
                except Exception:
                    pass

            st.divider()

            # ── 4. Graph Visualizations (Plotly & NetworkX Matplotlib) ────
            st.subheader("🕸️ 1. Causal Directed Acyclic Graph (DAG)")
            st.caption("Visualize assumed causal dependencies with interactive node/edge tooltips or high-resolution layouts.")

            tab_plotly, tab_mpl = st.tabs(["🌐 Interactive Plotly Graph", "🖼️ Static NetworkX (Matplotlib)"])

            with tab_plotly:
                fig_plotly = build_causal_plotly_figure(G_active, title=f"Causal DAG: {graph_source_name}")
                st.plotly_chart(fig_plotly, use_container_width=True)
                st.caption("💡 *Tip: Hover over any node to view in-degree and out-degree. Hover over the purple diamond markers on edges to read the causal rationale and observed correlation.*")

            with tab_mpl:
                fig_mpl = plot_causal_graph(G_active)
                st.pyplot(fig_mpl)

            st.divider()

            # ── 5. Inspect Relationship Explanations ───────────────────────
            st.subheader("🔍 2. Inspect Relationship Explanations")
            st.caption("Select any directed link in the graph to inspect its hypothesized causal mechanism and observed statistical association.")

            edges_list = list(G_active.edges())
            if edges_list:
                edge_options = [f"{u}  ➔  {v}" for u, v in edges_list]
                selected_edge_str = st.selectbox(
                    "Select Directed Link to Inspect:",
                    options=edge_options,
                    key="sel_inspect_edge",
                )
                sel_u, sel_v = selected_edge_str.split("  ➔  ")
                sel_edge_data = G_active.edges[sel_u, sel_v]
                sel_rel_type = sel_edge_data.get("relationship_type", "Hypothesized Direct Cause")
                sel_desc = sel_edge_data.get("description", "No description provided.")
                sel_corr = sel_edge_data.get("correlation", None)

                if sel_corr is None and active_df is not None and sel_u in active_df.columns and sel_v in active_df.columns:
                    if pd.api.types.is_numeric_dtype(active_df[sel_u]) and pd.api.types.is_numeric_dtype(active_df[sel_v]):
                        try:
                            c = active_df[[sel_u, sel_v]].dropna().corr().iloc[0, 1]
                            sel_corr = float(c) if not np.isnan(c) else None
                        except Exception:
                            sel_corr = None

                insp_c1, insp_c2 = st.columns([1.2, 1.8])
                with insp_c1:
                    st.markdown(
                        f"""
                        <div style="background: #1e293b; border-radius: 10px; padding: 1.2rem; border-left: 4px solid #6366f1;">
                            <h4 style="margin: 0; color: #cbd5e1; font-size: 0.95rem;">Relationship Overview</h4>
                            <p style="margin: 0.4rem 0; font-size: 1.1rem; font-weight: 700; color: #f8fafc;">
                                {sel_u.replace('_', ' ').title()} ➔ {sel_v.replace('_', ' ').title()}
                            </p>
                            <span class="status-pill" style="border-color: #38bdf8; color: #7dd3fc; font-size: 0.78rem;">
                                {sel_rel_type}
                            </span>
                            <div style="margin-top: 0.8rem; font-size: 0.88rem; color: #94a3b8;">
                                <b>Observed Pearson Correlation:</b><br>
                                <span style="font-size: 1.1rem; font-weight: 700; color: {'#10b981' if sel_corr and sel_corr > 0 else '#f43f5e' if sel_corr and sel_corr < 0 else '#cbd5e1'};">
                                    {f'r = {sel_corr:+.2f}' if sel_corr is not None else 'N/A (Categorical / Latent)'}
                                </span>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with insp_c2:
                    st.markdown(
                        f"""
                        <div style="background: #1e293b; border-radius: 10px; padding: 1.2rem; border-left: 4px solid #10b981; height: 100%;">
                            <h4 style="margin: 0; color: #cbd5e1; font-size: 0.95rem;">Hypothesized Causal Rationale</h4>
                            <p style="margin: 0.4rem 0; font-size: 0.95rem; line-height: 1.5; color: #e2e8f0;">
                                {sel_desc}
                            </p>
                            <p style="margin-top: 0.8rem; font-size: 0.82rem; color: #64748b;">
                                🛡️ <i>Scientific Note: Observational correlation indicates how these variables vary together in the data. The arrow represents your explicit theoretical hypothesis about causal direction.</i>
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
            else:
                st.info("ℹ️ No directed edges exist in the current graph. Use the editor below to construct relationships.")

            st.divider()

            # ── 6. Interactive Graph Management Workspace ─────────────────
            st.subheader("🛠️ 3. Interactive Graph Editor & Relationship Management")
            st.caption("Customize your structural model: add or delete variables, create directed edges, update causal hypotheses, and save your graph.")

            ed_tab_nodes, ed_tab_edges, ed_tab_modify, ed_tab_save = st.tabs([
                "➕ Node Management",
                "🔗 Edge Management",
                "✏️ Edit Relationship",
                "💾 Save & Export Graph",
            ])

            # Tab 1: Node Management
            with ed_tab_nodes:
                col_n_add, col_n_del = st.columns(2)
                with col_n_add:
                    st.markdown("##### ➕ Add Node")
                    with st.form("form_add_node"):
                        new_node_name = st.text_input("New Node Name (e.g. tutoring_program, socioeconomic):", key="inp_add_node_name")
                        new_node_role = st.selectbox(
                            "Node Role:",
                            options=["feature", "target", "confounder"],
                            format_func=lambda r: "Feature / Covariate" if r == "feature" else "Target / Outcome" if r == "target" else "Confounder / Latent Factor",
                            key="sel_add_node_role",
                        )
                        btn_submit_add_node = st.form_submit_button("➕ Add Node to Graph", use_container_width=True)

                        if btn_submit_add_node and new_node_name:
                            G_updated, err = add_node_to_graph(G_active, new_node_name, role=new_node_role)
                            if err:
                                st.error(err)
                            else:
                                st.session_state[session_graph_key] = graph_to_dict(G_updated)
                                st.success(f"✅ Node '{new_node_name}' added successfully!")
                                st.rerun()

                with col_n_del:
                    st.markdown("##### 🗑️ Remove Node")
                    curr_nodes = sorted(list(G_active.nodes()))
                    if curr_nodes:
                        with st.form("form_del_node"):
                            node_to_del = st.selectbox("Select Node to Delete:", options=curr_nodes, key="sel_del_node")
                            btn_submit_del_node = st.form_submit_button("🗑️ Delete Node & Connected Edges", type="secondary", use_container_width=True)

                            if btn_submit_del_node:
                                G_updated, err = remove_node_from_graph(G_active, node_to_del)
                                if err:
                                    st.error(err)
                                else:
                                    st.session_state[session_graph_key] = graph_to_dict(G_updated)
                                    st.success(f"✅ Node '{node_to_del}' and associated edges removed.")
                                    st.rerun()
                    else:
                        st.caption("No nodes available to delete.")

            # Tab 2: Edge Management
            with ed_tab_edges:
                col_e_add, col_e_del = st.columns(2)
                with col_e_add:
                    st.markdown("##### 🔗 Add Directed Edge")
                    curr_nodes_e = sorted(list(G_active.nodes()))
                    if len(curr_nodes_e) >= 2:
                        with st.form("form_add_edge"):
                            src_node = st.selectbox("Source Node (Hypothesized Cause):", options=curr_nodes_e, key="sel_edge_src")
                            remaining_nodes = [n for n in curr_nodes_e if n != src_node]
                            tgt_node = st.selectbox("Target Node (Hypothesized Effect):", options=remaining_nodes, key="sel_edge_tgt")
                            rel_type_choice = st.selectbox(
                                "Relationship Status / Type:",
                                options=[
                                    "Hypothesized Direct Cause",
                                    "Domain Knowledge Assumption",
                                    "Hypothesized Association",
                                    "Latent Confounder Link",
                                ],
                                key="sel_edge_type",
                            )
                            desc_input = st.text_area(
                                "Hypothesized Mechanism / Theoretical Rationale:",
                                placeholder=f"Explain why {src_node} is hypothesized to cause changes in {tgt_node}...",
                                key="inp_edge_desc",
                            )
                            btn_submit_add_edge = st.form_submit_button("➕ Add Directed Edge", use_container_width=True)

                            if btn_submit_add_edge:
                                corr_val = None
                                if active_df is not None and src_node in active_df.columns and tgt_node in active_df.columns:
                                    if pd.api.types.is_numeric_dtype(active_df[src_node]) and pd.api.types.is_numeric_dtype(active_df[tgt_node]):
                                        try:
                                            c = active_df[[src_node, tgt_node]].dropna().corr().iloc[0, 1]
                                            corr_val = float(c) if not np.isnan(c) else None
                                        except Exception:
                                            corr_val = None

                                G_updated, warn = add_edge_to_graph(
                                    G_active,
                                    source=src_node,
                                    target=tgt_node,
                                    relationship_type=rel_type_choice,
                                    description=desc_input,
                                    correlation=corr_val,
                                )
                                st.session_state[session_graph_key] = graph_to_dict(G_updated)
                                if warn:
                                    st.warning(warn)
                                else:
                                    st.success(f"✅ Directed edge '{src_node} ➔ {tgt_node}' added successfully!")
                                st.rerun()
                    else:
                        st.caption("At least two nodes are required to create a directed edge.")

                with col_e_del:
                    st.markdown("##### 🗑️ Remove Directed Edge")
                    if edges_list:
                        with st.form("form_del_edge"):
                            edge_to_del_str = st.selectbox("Select Edge to Delete:", options=[f"{u} ➔ {v}" for u, v in edges_list], key="sel_del_edge")
                            btn_submit_del_edge = st.form_submit_button("🗑️ Delete Directed Edge", type="secondary", use_container_width=True)

                            if btn_submit_del_edge:
                                du, dv = edge_to_del_str.split(" ➔ ")
                                G_updated, err = remove_edge_from_graph(G_active, du, dv)
                                if err:
                                    st.error(err)
                                else:
                                    st.session_state[session_graph_key] = graph_to_dict(G_updated)
                                    st.success(f"✅ Edge '{du} ➔ {dv}' removed.")
                                    st.rerun()
                    else:
                        st.caption("No edges available to delete.")

            # Tab 3: Edit Relationship
            with ed_tab_modify:
                st.markdown("##### ✏️ Edit Relationship Details & Direction")
                if edges_list:
                    edit_edge_options = [f"{u} ➔ {v}" for u, v in edges_list]
                    chosen_edit_str = st.selectbox("Select Edge to Modify:", options=edit_edge_options, key="sel_mod_edge")
                    orig_u, orig_v = chosen_edit_str.split(" ➔ ")
                    curr_edge_data = G_active.edges[orig_u, orig_v]

                    with st.form("form_edit_edge"):
                        mod_rel_type = st.selectbox(
                            "Update Relationship Type:",
                            options=[
                                "Hypothesized Direct Cause",
                                "Domain Knowledge Assumption",
                                "Hypothesized Association",
                                "Latent Confounder Link",
                            ],
                            index=["Hypothesized Direct Cause", "Domain Knowledge Assumption", "Hypothesized Association", "Latent Confounder Link"].index(
                                curr_edge_data.get("relationship_type", "Hypothesized Direct Cause")
                            ) if curr_edge_data.get("relationship_type") in ["Hypothesized Direct Cause", "Domain Knowledge Assumption", "Hypothesized Association", "Latent Confounder Link"] else 0,
                            key="sel_mod_type",
                        )
                        mod_desc = st.text_area(
                            "Update Causal Mechanism / Explanation:",
                            value=curr_edge_data.get("description", ""),
                            key="inp_mod_desc",
                        )
                        flip_direction = st.checkbox(f"🔄 Invert Causal Direction (Change to {orig_v} ➔ {orig_u})", key="chk_flip_dir")
                        btn_submit_edit = st.form_submit_button("💾 Save Relationship Changes", use_container_width=True)

                        if btn_submit_edit:
                            final_src = orig_v if flip_direction else orig_u
                            final_tgt = orig_u if flip_direction else orig_v

                            corr_val = curr_edge_data.get("correlation", None)
                            G_updated, warn = update_edge_in_graph(
                                G_active,
                                old_source=orig_u,
                                old_target=orig_v,
                                new_source=final_src,
                                new_target=final_tgt,
                                relationship_type=mod_rel_type,
                                description=mod_desc,
                                correlation=corr_val,
                            )
                            st.session_state[session_graph_key] = graph_to_dict(G_updated)
                            if warn:
                                st.warning(warn)
                            else:
                                st.success(f"✅ Relationship '{final_src} ➔ {final_tgt}' updated successfully!")
                            st.rerun()
                else:
                    st.caption("No edges available to edit.")

            # Tab 4: Save & Export Graph
            with ed_tab_save:
                st.markdown("##### 💾 Session State Persistence & Export")
                st.info("💡 **Active Session Persistence:** Your custom graph structure (nodes, directed edges, and descriptions) is automatically preserved in your Streamlit session state as you interact with the app.")

                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    graph_json_bytes = json.dumps(st.session_state[session_graph_key], indent=2)
                    st.download_button(
                        label="📥 Export Causal DAG Schema (JSON)",
                        data=graph_json_bytes,
                        file_name="causal_dag.json",
                        mime="application/json",
                        use_container_width=True,
                    )
                with col_exp2:
                    if st.button("🔄 Reset Graph to Initial State", use_container_width=True):
                        if is_upload_causal:
                            st.session_state.uploaded_causal_graph = generate_initial_causal_graph(
                                feature_cols=feature_cols,
                                target_col=target_col,
                                df=active_df,
                            )
                        else:
                            st.session_state.demo_causal_graph = graph_to_dict(build_causal_graph())
                        st.success("✅ Graph reset to initial state.")
                        st.rerun()

            st.divider()

            # ── 7. Tabular Relationship Overview ──────────────────────────
            st.subheader("📝 4. All Causal Relationship Hypotheses")
            st.caption("Complete tabular summary of all assumed directional dependencies in this graph.")

            edge_rows = get_graph_edge_descriptions(G_active, active_df)
            if edge_rows:
                edge_df = pd.DataFrame(edge_rows)
                st.dataframe(edge_df, use_container_width=True, hide_index=True)
            else:
                st.caption("No edges to display in table.")

            st.divider()

            # ── 8. Scientific Principles & Educational Content ────────────
            st.subheader("📐 " + CORRELATION_VS_CAUSATION["title"])
            st.markdown(CORRELATION_VS_CAUSATION["content"])

            st.divider()

            st.subheader("🛡️ " + TRUSTWORTHY_AI_CONTENT["title"])
            st.markdown(TRUSTWORTHY_AI_CONTENT["content"])

            st.divider()

            st.subheader("📖 " + SCIENTIFIC_DISCLAIMER_CONTENT["title"])
            st.markdown(SCIENTIFIC_DISCLAIMER_CONTENT["content"])

    # -----------------------------------------------------------------------
    # Tab 6: Counterfactual Simulator
    # -----------------------------------------------------------------------
    with tab_counterfactual:
        st.markdown(
            """
            <div style="text-align:center; padding:0.5rem 0;">
                <h2 style="background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                           font-size: 2rem; font-weight: 800;">🔄 Counterfactual Simulator</h2>
                <p style="color:#888; font-size:1rem; margin-top:-0.4rem;">
                    Evaluate <em>what-if</em> hypothetical scenarios to inspect model sensitivity and decision boundaries
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        # Scientific Disclaimer Callout
        st.warning(
            "⚠️ **Important Scientific & Methodological Disclaimer:** Counterfactual simulations in this environment "
            "evaluate the **mathematical sensitivity** of the trained machine learning pipeline. They determine what the "
            "model $\\hat{f}(x)$ would predict if specific input feature values were changed. "
            "Simulated changes **do not prove or guarantee real-world causal effects**, as statistical models "
            "may rely on correlational associations, confounders, or proxy signals. "
            "For causal reasoning, refer to the **Causal Analysis** structural graph."
        )

        if st.session_state.dataset_source == "upload":
            if st.session_state.uploaded_df is None:
                st.info("📂 No uploaded dataset found. Please upload a CSV file in the **Dataset Overview** tab first.")
            elif st.session_state.uploaded_model_results is None:
                st.warning("⚠️ No machine learning pipeline has been trained yet for this uploaded dataset. Please train a pipeline in the **Machine Learning Model** tab before running counterfactual simulations.")
            else:
                res = st.session_state.uploaded_model_results
                best_pipeline = res["best_pipeline"]
                best_name = res["best_name"]
                problem_type = res["problem_type"]
                target_col = res["target_col"]
                feature_cols = res["feature_cols"]
                numeric_features = res.get("numeric_features", [])
                categorical_features = res.get("categorical_features", [])
                target_encoder = res.get("target_encoder")
                X_train = res.get("X_train")
                X_test = res.get("X_test")
                y_test = res.get("y_test")
                up_df = st.session_state.uploaded_df
                up_filename = st.session_state.uploaded_filename or "Uploaded Dataset"

                st.markdown(
                    f"""
                    <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1.2rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                            <div>
                                <h3 style="margin: 0; color: #f8fafc; font-size: 1.2rem;">
                                    🎯 Dataset: <code>{up_filename}</code> &nbsp;|&nbsp; Model: <strong style="color: #818cf8;">{best_name}</strong>
                                </h3>
                                <p style="margin: 0.2rem 0 0; color: #94a3b8; font-size: 0.88rem;">
                                    Task: <strong style="color: #38bdf8;">{problem_type}</strong> &nbsp;|&nbsp; 
                                    Target: <code>{target_col}</code> &nbsp;|&nbsp; 
                                    Features: <strong>{len(feature_cols)}</strong> ({len(numeric_features)} Numerical, {len(categorical_features)} Categorical)
                                </p>
                            </div>
                            <div>
                                <span class="status-pill"><span class="status-dot"></span> Pipeline Ready</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Step 1: Set Baseline Observation
                st.markdown("### 1. Set Baseline Observation (Original)")
                st.caption("Choose whether to load an existing test record from your data or enter custom baseline feature values.")

                base_mode = st.radio(
                    "Baseline Input Source:",
                    ["📌 Select sample from Test Set", "✍️ Manual Custom Input"],
                    horizontal=True,
                    key="cf_base_mode_radio",
                )

                selected_row = None
                if base_mode == "📌 Select sample from Test Set" and X_test is not None and not X_test.empty:
                    sample_idx = st.slider(
                        "Test Sample Index:",
                        min_value=0,
                        max_value=len(X_test) - 1,
                        value=0,
                        key="cf_test_sample_idx_slider",
                    )
                    selected_row = X_test.iloc[sample_idx]
                    if y_test is not None:
                        actual_raw = y_test.iloc[sample_idx] if hasattr(y_test, "iloc") else y_test[sample_idx]
                        if problem_type == "Classification" and target_encoder is not None:
                            try:
                                actual_label_str = str(target_encoder.inverse_transform([actual_raw])[0])
                            except Exception:
                                actual_label_str = str(actual_raw)
                        else:
                            actual_label_str = f"{actual_raw:,.2f}" if isinstance(actual_raw, (int, float, np.number)) else str(actual_raw)
                        st.info(f"📋 Loaded **Test Sample #{sample_idx}** &mdash; Ground Truth in Dataset: `{target_col}` = **{actual_label_str}**")

                # Layout: Baseline Inputs (Left) and Counterfactual Inputs (Right)
                st.markdown("---")
                col_orig_box, col_mod_box = st.columns(2)

                orig_inputs = {}
                mod_inputs = {}

                with col_orig_box:
                    st.markdown("#### 📍 Baseline (Original Inputs)")
                    st.caption("Reference values used as the simulation benchmark.")

                    for f_col in feature_cols:
                        f_series = up_df[f_col]
                        clean_label = f_col.replace("_", " ").title()

                        if f_col in numeric_features:
                            f_num = pd.to_numeric(f_series, errors="coerce").dropna()
                            f_min = float(f_num.min()) if not f_num.empty else 0.0
                            f_max = float(f_num.max()) if not f_num.empty else 100.0
                            if f_min == f_max:
                                f_min -= 1.0
                                f_max += 1.0
                            f_step = round((f_max - f_min) / 100.0, 2)
                            f_step = max(0.01, f_step)

                            if selected_row is not None and pd.notnull(selected_row.get(f_col)):
                                val_orig = float(selected_row[f_col])
                            else:
                                val_orig = float(f_num.median()) if not f_num.empty else (f_min + f_max) / 2.0

                            orig_inputs[f_col] = st.number_input(
                                f"{clean_label}",
                                min_value=f_min,
                                max_value=f_max,
                                value=round(val_orig, 2),
                                step=f_step,
                                key=f"cf_up_orig_{f_col}",
                            )
                        else:
                            opts = [str(x) for x in f_series.dropna().unique()]
                            if not opts:
                                opts = ["Standard"]

                            if selected_row is not None and str(selected_row.get(f_col)) in opts:
                                default_idx = opts.index(str(selected_row[f_col]))
                            else:
                                default_idx = 0

                            orig_inputs[f_col] = st.selectbox(
                                f"{clean_label}",
                                options=opts,
                                index=default_idx,
                                key=f"cf_up_orig_{f_col}",
                            )

                with col_mod_box:
                    st.markdown("#### ✏️ Counterfactual (Modified Inputs)")
                    st.caption("Change any feature values to evaluate what-if sensitivity.")

                    for f_col in feature_cols:
                        f_series = up_df[f_col]
                        clean_label = f_col.replace("_", " ").title()

                        if f_col in numeric_features:
                            f_num = pd.to_numeric(f_series, errors="coerce").dropna()
                            f_min = float(f_num.min()) if not f_num.empty else 0.0
                            f_max = float(f_num.max()) if not f_num.empty else 100.0
                            if f_min == f_max:
                                f_min -= 1.0
                                f_max += 1.0
                            f_step = round((f_max - f_min) / 100.0, 2)
                            f_step = max(0.01, f_step)

                            # Default mod value to matching orig input value
                            default_mod = float(orig_inputs[f_col])

                            mod_inputs[f_col] = st.number_input(
                                f"{clean_label} (Modified)",
                                min_value=f_min,
                                max_value=f_max,
                                value=round(default_mod, 2),
                                step=f_step,
                                key=f"cf_up_mod_{f_col}",
                            )
                        else:
                            opts = [str(x) for x in f_series.dropna().unique()]
                            if not opts:
                                opts = ["Standard"]

                            cur_orig_val = str(orig_inputs[f_col])
                            default_mod_idx = opts.index(cur_orig_val) if cur_orig_val in opts else 0

                            mod_inputs[f_col] = st.selectbox(
                                f"{clean_label} (Modified)",
                                options=opts,
                                index=default_mod_idx,
                                key=f"cf_up_mod_{f_col}",
                            )

                st.markdown("")
                if st.button("🔄 Run Counterfactual Simulation", key="btn_run_uploaded_cf", type="primary", use_container_width=True):
                    with st.spinner("Evaluating counterfactual through trained ML pipeline..."):
                        try:
                            sim_result = simulate_pipeline_counterfactual(
                                pipeline=best_pipeline,
                                original_inputs=orig_inputs,
                                modified_inputs=mod_inputs,
                                feature_cols=feature_cols,
                                problem_type=problem_type,
                                target_encoder=target_encoder,
                                target_col=target_col,
                            )
                            st.session_state.uploaded_cf_result = sim_result
                        except Exception as e:
                            st.error(f"❌ Error during counterfactual simulation: {e}")

                if st.session_state.uploaded_cf_result is not None:
                    sim = st.session_state.uploaded_cf_result
                    st.divider()
                    st.markdown("### 🎯 Simulation Comparison & Prediction Shift")

                    # 3 Metric / Comparison Cards
                    rc1, rc2, rc3 = st.columns(3)

                    orig_badge_color = "#6366f1"

                    with rc1:
                        if problem_type == "Classification":
                            conf_text = f"Confidence: {sim['original_confidence']:.1%}" if sim["original_confidence"] is not None else ""
                            st.markdown(
                                f"""
                                <div style="text-align:center; padding:1.2rem; border-radius:12px; background:#1e293b; border:2px solid {orig_badge_color};">
                                    <h4 style="margin:0; color:#94a3b8; font-size:0.92rem; text-transform:uppercase; letter-spacing:0.05em;">Original Prediction</h4>
                                    <p style="font-size:2rem; font-weight:800; margin:0.3rem 0; color:#818cf8;">{sim['original_label']}</p>
                                    <p style="margin:0; color:#cbd5e1; font-size:0.88rem;">{conf_text}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            st.markdown(
                                f"""
                                <div style="text-align:center; padding:1.2rem; border-radius:12px; background:#1e293b; border:2px solid {orig_badge_color};">
                                    <h4 style="margin:0; color:#94a3b8; font-size:0.92rem; text-transform:uppercase; letter-spacing:0.05em;">Original Prediction</h4>
                                    <p style="font-size:2rem; font-weight:800; margin:0.3rem 0; color:#818cf8;">{sim['original_label']}</p>
                                    <p style="margin:0; color:#94a3b8; font-size:0.85rem;">Target: <code>{target_col}</code></p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    with rc2:
                        if problem_type == "Classification":
                            conf_text_mod = f"Confidence: {sim['modified_confidence']:.1%}" if sim["modified_confidence"] is not None else ""
                            border_style = "2px solid #10b981" if sim["prediction_changed"] else "2px dashed #94a3b8"
                            text_color = "#34d399" if sim["prediction_changed"] else "#e2e8f0"
                            st.markdown(
                                f"""
                                <div style="text-align:center; padding:1.2rem; border-radius:12px; background:#1e293b; border:{border_style};">
                                    <h4 style="margin:0; color:#94a3b8; font-size:0.92rem; text-transform:uppercase; letter-spacing:0.05em;">Counterfactual Prediction</h4>
                                    <p style="font-size:2rem; font-weight:800; margin:0.3rem 0; color:{text_color};">{sim['modified_label']}</p>
                                    <p style="margin:0; color:#cbd5e1; font-size:0.88rem;">{conf_text_mod}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            border_color = "#10b981" if sim.get("diff_value", 0) >= 0 else "#f43f5e"
                            val_color = "#34d399" if sim.get("diff_value", 0) >= 0 else "#fb7185"
                            st.markdown(
                                f"""
                                <div style="text-align:center; padding:1.2rem; border-radius:12px; background:#1e293b; border:2px solid {border_color};">
                                    <h4 style="margin:0; color:#94a3b8; font-size:0.92rem; text-transform:uppercase; letter-spacing:0.05em;">Counterfactual Prediction</h4>
                                    <p style="font-size:2rem; font-weight:800; margin:0.3rem 0; color:{val_color};">{sim['modified_label']}</p>
                                    <p style="margin:0; color:#94a3b8; font-size:0.85rem;">Target: <code>{target_col}</code></p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    with rc3:
                        if problem_type == "Classification":
                            if sim["prediction_changed"]:
                                shift_title = "⚡ Decision Flipped"
                                shift_summary = f"{sim['original_label']} ➔ {sim['modified_label']}"
                                shift_sub = f"Confidence Shift: {sim['diff_confidence']:+.1%}" if sim.get("diff_confidence") is not None else "Class flipped"
                                card_border = "2px solid #f59e0b"
                                card_color = "#fbbf24"
                            else:
                                shift_title = "⚖️ Prediction Preserved"
                                shift_summary = "Class Unchanged"
                                shift_sub = f"Confidence Shift: {sim['diff_confidence']:+.1%}" if sim.get("diff_confidence") is not None else "No class change"
                                card_border = "2px solid #334155"
                                card_color = "#94a3b8"
                            st.markdown(
                                f"""
                                <div style="text-align:center; padding:1.2rem; border-radius:12px; background:#1e293b; border:{card_border};">
                                    <h4 style="margin:0; color:#94a3b8; font-size:0.92rem; text-transform:uppercase; letter-spacing:0.05em;">{shift_title}</h4>
                                    <p style="font-size:1.6rem; font-weight:800; margin:0.4rem 0; color:{card_color};">{shift_summary}</p>
                                    <p style="margin:0; color:#cbd5e1; font-size:0.88rem;">{shift_sub}</p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                        else:
                            diff_v = sim.get("diff_value", 0.0)
                            pct_v = sim.get("pct_change", 0.0)
                            color_d = "#34d399" if diff_v >= 0 else "#fb7185"
                            st.markdown(
                                f"""
                                <div style="text-align:center; padding:1.2rem; border-radius:12px; background:#1e293b; border:2px solid #334155;">
                                    <h4 style="margin:0; color:#94a3b8; font-size:0.92rem; text-transform:uppercase; letter-spacing:0.05em;">Observed Model Shift</h4>
                                    <p style="font-size:1.8rem; font-weight:800; margin:0.4rem 0; color:{color_d};">{diff_v:+,.2f}</p>
                                    <p style="margin:0; color:#cbd5e1; font-size:0.88rem;">Relative Shift: <strong>{pct_v:+.1f}%</strong></p>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    # Human-Readable Explanation Banner
                    st.markdown("")
                    st.markdown(
                        f"""
                        <div style="background: rgba(99, 102, 241, 0.12); border-left: 4px solid #6366f1; border-radius: 8px; padding: 1.1rem 1.3rem; margin: 0.8rem 0 1.2rem 0;">
                            <h4 style="margin: 0 0 0.4rem 0; color: #818cf8; font-size: 1.05rem; display: flex; align-items: center; gap: 8px;">
                                📝 Human-Readable Counterfactual Narrative
                            </h4>
                            <p style="margin: 0; color: #e2e8f0; font-size: 0.96rem; line-height: 1.55;">
                                {sim['narrative']}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Changed Features Audit
                    st.markdown("#### 🔍 Feature Modifications Audit")
                    if len(sim["changed_details"]) == 0:
                        st.info("ℹ️ No input features were modified between the baseline and counterfactual scenario. Change one or more feature values above and re-run simulation.")
                    else:
                        st.markdown(
                            f"**{len(sim['changed_details'])}** input feature(s) modified:"
                        )
                        cols_pills = st.columns(min(3, len(sim["changed_details"])))
                        for idx_c, cd in enumerate(sim["changed_details"]):
                            with cols_pills[idx_c % len(cols_pills)]:
                                st.markdown(
                                    f"""
                                    <div style="background: #1e293b; border: 1px solid #3b82f6; border-radius: 8px; padding: 0.6rem 0.9rem; margin-bottom: 0.6rem;">
                                        <div style="color: #93c5fd; font-weight: 700; font-size: 0.9rem;">{cd['Feature']}</div>
                                        <div style="color: #f8fafc; font-size: 0.95rem; margin-top: 0.2rem;">
                                            <code>{cd['Original Value']}</code> ➔ <strong style="color: #34d399;"><code>{cd['Modified Value']}</code></strong>
                                        </div>
                                        <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.2rem;">Shift: {cd['Shift']}</div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True,
                                )

                        with st.expander("📋 View Complete Feature Audit Table"):
                            st.dataframe(sim["all_comparison_df"], use_container_width=True, hide_index=True)

                    # Comparative Plot
                    st.markdown("#### 📊 Prediction Shift Visualization")
                    fig_comp = plot_counterfactual_comparison(sim)
                    st.pyplot(fig_comp)
                    plt.close(fig_comp)

        else:
            # Demo Mode: Student Performance
            st.markdown(
                f"""
                <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid #334155; border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1.2rem;">
                    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                        <div>
                            <h3 style="margin: 0; color: #f8fafc; font-size: 1.2rem;">
                                🎓 Dataset: <code>student-performance (Demo)</code> &nbsp;|&nbsp; Model: <strong style="color: #818cf8;">{best_name}</strong>
                            </h3>
                            <p style="margin: 0.2rem 0 0; color: #94a3b8; font-size: 0.88rem;">
                                Task: <strong style="color: #38bdf8;">Classification</strong> &nbsp;|&nbsp; 
                                Target: <code>result</code> (PASS / FAIL) &nbsp;|&nbsp; 
                                Features: <strong>6</strong> Academic Factors
                            </p>
                        </div>
                        <div>
                            <span class="status-pill"><span class="status-dot"></span> Demo Pipeline Ready</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📍 1. Baseline Inputs (Original)")
                st.caption("Baseline academic profile")
                orig_study = st.number_input("Original Study Hours", min_value=0.0, max_value=12.0, value=2.0, step=0.5, key="demo_orig_study")
                orig_attend = st.number_input("Original Attendance (%)", min_value=0.0, max_value=100.0, value=50.0, step=5.0, key="demo_orig_attend")
                orig_prev = st.number_input("Original Previous Scores", min_value=0.0, max_value=100.0, value=45.0, step=5.0, key="demo_orig_prev")
                orig_sleep = st.number_input("Original Sleep Hours", min_value=3.0, max_value=12.0, value=6.0, step=0.5, key="demo_orig_sleep")
                orig_assign = st.number_input("Original Assignments Completed", min_value=0, max_value=10, value=3, step=1, key="demo_orig_assign")
                orig_partic = st.number_input("Original Class Participation", min_value=0, max_value=10, value=3, step=1, key="demo_orig_partic")

            with col2:
                st.markdown("#### ✏️ 2. Counterfactual Inputs (Modified)")
                st.caption("Modified academic profile")
                mod_study = st.number_input("Modified Study Hours", min_value=0.0, max_value=12.0, value=6.0, step=0.5, key="demo_mod_study")
                mod_attend = st.number_input("Modified Attendance (%)", min_value=0.0, max_value=100.0, value=85.0, step=5.0, key="demo_mod_attend")
                mod_prev = st.number_input("Modified Previous Scores", min_value=0.0, max_value=100.0, value=70.0, step=5.0, key="demo_mod_prev")
                mod_sleep = st.number_input("Modified Sleep Hours", min_value=3.0, max_value=12.0, value=8.0, step=0.5, key="demo_mod_sleep")
                mod_assign = st.number_input("Modified Assignments Completed", min_value=0, max_value=10, value=8, step=1, key="demo_mod_assign")
                mod_partic = st.number_input("Modified Class Participation", min_value=0, max_value=10, value=8, step=1, key="demo_mod_partic")

            if st.button("🔄 Run Counterfactual Simulation", key="btn_run_demo_cf", type="primary", use_container_width=True):
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

                with st.spinner("Evaluating demo model counterfactual..."):
                    sim_result = simulate_counterfactual(
                        best_model, scaler, le, feature_names, orig_inputs, mod_inputs
                    )
                    st.session_state.demo_cf_result = sim_result

            if st.session_state.demo_cf_result is not None:
                sim = st.session_state.demo_cf_result
                st.divider()
                st.markdown("### 🎯 Simulation Comparison & Prediction Shift")

                r1, r2, r3 = st.columns(3)
                orig_color = "#34d399" if sim["original_label"] == "PASS" else "#fb7185"
                mod_color = "#34d399" if sim["modified_label"] == "PASS" else "#fb7185"

                with r1:
                    st.markdown(
                        f"""
                        <div style="text-align:center; padding:1.2rem; border-radius:12px; background:#1e293b; border:2px solid #6366f1;">
                            <h4 style="margin:0; color:#94a3b8; font-size:0.92rem; text-transform:uppercase;">Original Prediction</h4>
                            <p style="font-size:2rem; font-weight:800; margin:0.3rem 0; color:{orig_color};">{sim['original_label']}</p>
                            <p style="margin:0; color:#cbd5e1; font-size:0.88rem;">Confidence: {sim['original_confidence']:.1%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with r2:
                    border_style = "2px solid #10b981" if sim["prediction_changed"] else "2px dashed #94a3b8"
                    st.markdown(
                        f"""
                        <div style="text-align:center; padding:1.2rem; border-radius:12px; background:#1e293b; border:{border_style};">
                            <h4 style="margin:0; color:#94a3b8; font-size:0.92rem; text-transform:uppercase;">Counterfactual Prediction</h4>
                            <p style="font-size:2rem; font-weight:800; margin:0.3rem 0; color:{mod_color};">{sim['modified_label']}</p>
                            <p style="margin:0; color:#cbd5e1; font-size:0.88rem;">Confidence: {sim['modified_confidence']:.1%}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with r3:
                    if sim["prediction_changed"]:
                        shift_title = "⚡ Decision Flipped"
                        shift_summary = f"{sim['original_label']} ➔ {sim['modified_label']}"
                        card_border = "2px solid #f59e0b"
                        card_color = "#fbbf24"
                    else:
                        shift_title = "⚖️ Prediction Preserved"
                        shift_summary = "Class Maintained"
                        card_border = "2px solid #334155"
                        card_color = "#94a3b8"
                    st.markdown(
                        f"""
                        <div style="text-align:center; padding:1.2rem; border-radius:12px; background:#1e293b; border:{card_border};">
                            <h4 style="margin:0; color:#94a3b8; font-size:0.92rem; text-transform:uppercase;">{shift_title}</h4>
                            <p style="font-size:1.6rem; font-weight:800; margin:0.4rem 0; color:{card_color};">{shift_summary}</p>
                            <p style="margin:0; color:#cbd5e1; font-size:0.88rem;">Outcome Transition</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                # Narrative Banner
                st.markdown("")
                st.markdown(
                    f"""
                    <div style="background: rgba(99, 102, 241, 0.12); border-left: 4px solid #6366f1; border-radius: 8px; padding: 1.1rem 1.3rem; margin: 0.8rem 0 1.2rem 0;">
                        <h4 style="margin: 0 0 0.4rem 0; color: #818cf8; font-size: 1.05rem;">
                            📝 Human-Readable Counterfactual Narrative
                        </h4>
                        <p style="margin: 0; color: #e2e8f0; font-size: 0.96rem; line-height: 1.55;">
                            {sim['narrative']}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # Changed Features Breakdown
                st.markdown("#### 🔍 Feature Modifications Breakdown")
                if len(sim["changed_details"]) == 0:
                    st.info("ℹ️ No input features were modified.")
                else:
                    cols_pills = st.columns(min(3, len(sim["changed_details"])))
                    for idx_c, cd in enumerate(sim["changed_details"]):
                        with cols_pills[idx_c % len(cols_pills)]:
                            st.markdown(
                                f"""
                                <div style="background: #1e293b; border: 1px solid #3b82f6; border-radius: 8px; padding: 0.6rem 0.9rem; margin-bottom: 0.6rem;">
                                    <div style="color: #93c5fd; font-weight: 700; font-size: 0.9rem;">{cd['Feature']}</div>
                                    <div style="color: #f8fafc; font-size: 0.95rem; margin-top: 0.2rem;">
                                        <code>{cd['Original Value']}</code> ➔ <strong style="color: #34d399;"><code>{cd['Modified Value']}</code></strong>
                                    </div>
                                    <div style="color: #94a3b8; font-size: 0.8rem; margin-top: 0.2rem;">{cd['Change Description']}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    with st.expander("📋 View Complete Feature Changes Table"):
                        st.dataframe(sim["changes_df"], use_container_width=True, hide_index=True)

                # Comparison Plot
                st.markdown("#### 📊 Prediction Probability Shift Visualization")
                fig_comp = plot_counterfactual_comparison(sim)
                st.pyplot(fig_comp)
                plt.close(fig_comp)


# ===========================================================================
# VIEW 3: EXPLAINABLE AI AGENT WORKSPACE
# ===========================================================================
elif st.session_state.current_workspace == WORKSPACE_AGENT:
    # Workspace Banner & Breadcrumb Navigation
    col_nav_left, col_nav_right = st.columns([1.8, 6.2])
    with col_nav_left:
        st.button(
            "🏠 ← Return to Platform Home",
            key="btn_back_home_agent",
            use_container_width=True,
            on_click=navigate_to,
            args=(WORKSPACE_HOME,),
        )
    with col_nav_right:
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: flex-end; gap: 8px;">
                <span style="font-size: 0.85rem; color: #94a3b8;">Current Location:</span>
                <span style="font-size: 0.85rem; color: #cbd5e1; font-weight: 600;">Home / Workspaces / Explainable AI Agent</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="workspace-banner" style="border-left: 4px solid #ec4899; margin-bottom: 1rem;">
            <div>
                <h1>🤖 Explainable AI Agent Workspace</h1>
                <p>Evidence-grounded conversational assistant powered by Google Gemini. Evaluates predictions using model metrics, SHAP attributions, structural causal DAGs, and counterfactual simulations.</p>
            </div>
            <div style="text-align: right;">
                <span class="status-pill" style="border-color: #ec4899; color: #f472b6;">
                    <span class="status-dot" style="background-color: #ec4899; box-shadow: 0 0 8px #ec4899;"></span>
                    Gemini Active
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Resolve Effective Gemini Key
    effective_key = get_gemini_api_key(st.session_state.get("agent_custom_api_key"))
    is_connected = (effective_key is not None)

    # API Key Configuration Notice / Expander
    if not is_connected:
        with st.expander("🔑 Configure Google Gemini API Key (Required for live chat)", expanded=True):
            st.info(
                "The Explainable AI Agent connects to Google Gemini (`google-genai`). "
                "Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey). "
                "You can enter it below for this session, or configure it via `export GEMINI_API_KEY='...'` "
                "or in `.streamlit/secrets.toml`."
            )
            col_k1, col_k2 = st.columns([3, 1])
            with col_k1:
                sess_key_input = st.text_input(
                    "Enter GEMINI_API_KEY:",
                    type="password",
                    placeholder="AIzaSy...",
                    key="input_session_gemini_key",
                    label_visibility="collapsed",
                )
            with col_k2:
                if st.button("💾 Save Key", key="btn_save_session_key", use_container_width=True, type="primary"):
                    if sess_key_input and sess_key_input.strip():
                        st.session_state.agent_custom_api_key = sess_key_input.strip()
                        st.rerun()
                    else:
                        st.warning("Please enter a valid key.")
    else:
        col_status_left, col_status_right = st.columns([5, 1])
        with col_status_left:
            st.markdown(
                f"""
                <div style="display: flex; align-items: center; gap: 8px; font-size: 0.84rem; color: #94a3b8; margin-bottom: 0.8rem;">
                    <span style="color: #34d399; font-weight: 700;">● Google Gemini Connected</span>
                    <span>| Model: <code style="color: #818cf8;">{DEFAULT_GEMINI_MODEL}</code></span>
                    <span>| Mode: <strong style="color: #38bdf8;">Data-Grounded Reasoning</strong></span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col_status_right:
            if st.session_state.agent_custom_api_key:
                if st.button("Reset Key", key="btn_reset_session_key", help="Clear session API key"):
                    st.session_state.agent_custom_api_key = None
                    st.rerun()

    # Extract Active Grounding Evidence
    evidence = extract_grounding_evidence(
        dataset_source=st.session_state.dataset_source,
        uploaded_df=st.session_state.get("uploaded_df"),
        uploaded_filename=st.session_state.get("uploaded_filename"),
        uploaded_model_results=st.session_state.get("uploaded_model_results"),
        uploaded_shap_cache=st.session_state.get("uploaded_shap_cache"),
        uploaded_causal_graph=st.session_state.get("uploaded_causal_graph"),
        uploaded_cf_result=st.session_state.get("uploaded_cf_result"),
        demo_cf_result=st.session_state.get("demo_cf_result"),
        latest_prediction=st.session_state.get("latest_model_prediction"),
        latest_shap=st.session_state.get("latest_shap_individual"),
        demo_model_name=best_name,
        demo_metrics=best_metrics,
    )

    # Main Agent Layout: Chat Workspace (Left) & Model Evidence Drawer (Right)
    col_chat_main, col_evidence_sidebar = st.columns([2.05, 0.95], gap="medium")

    with col_chat_main:
        # 1. Suggested Questions
        st.markdown("##### 💡 Suggested Questions")
        tab_q_model, tab_q_general = st.tabs(["🎯 Model & Prediction Explanations", "🔬 General XAI & Causal Science"])

        prompt_to_send = None

        with tab_q_model:
            q_col1, q_col2 = st.columns(2)
            with q_col1:
                if st.button("❓ Why did the model predict FAIL?", key="sq_why_fail", use_container_width=True):
                    prompt_to_send = "Why did the model predict FAIL?"
                if st.button("📊 Highest SHAP feature contributions?", key="sq_shap_top", use_container_width=True):
                    prompt_to_send = "Which features had the highest SHAP contribution for this prediction?"
            with q_col2:
                if st.button("🔬 What does the causal graph assume?", key="sq_causal_graph", use_container_width=True):
                    prompt_to_send = "What does the causal graph assume about these variables, and how does it relate to the prediction?"
                if st.button("🔄 What changes could alter the prediction?", key="sq_counterfactual", use_container_width=True):
                    prompt_to_send = "What changes could alter the model prediction based on counterfactual analysis?"

        with tab_q_general:
            g_col1, g_col2 = st.columns(2)
            with g_col1:
                if st.button("📖 SHAP values vs. causal effects?", key="gq_shap_vs_causal", use_container_width=True):
                    prompt_to_send = "What is the key scientific difference between SHAP feature attributions and causal effects?"
                if st.button("⚖️ Why doesn't correlation prove causation?", key="gq_corr_vs_caus", use_container_width=True):
                    prompt_to_send = "Why doesn't correlation prove causation in machine learning models, and how do confounders affect decisions?"
            with g_col2:
                if st.button("🛡️ Core pillars of Trustworthy AI?", key="gq_trustworthy_pillars", use_container_width=True):
                    prompt_to_send = "What are the core pillars of Trustworthy Intelligent Systems and Explainable AI?"
                if st.button("🔮 Counterfactuals in Explainable AI?", key="gq_cf_role", use_container_width=True):
                    prompt_to_send = "How do counterfactual simulations support human-interpretable Explainable AI?"

        st.markdown("---")

        # 2. Conversation History Rendering
        if not st.session_state.agent_chat_history:
            st.markdown(
                """
                <div style="background: rgba(30, 41, 59, 0.5); border: 1px dashed #475569; border-radius: 12px; padding: 2rem; text-align: center; margin: 1rem 0;">
                    <div style="font-size: 2.2rem; margin-bottom: 0.5rem;">🤖</div>
                    <h4 style="color: #f8fafc; margin: 0 0 0.5rem 0;">Explainable AI Agent Ready</h4>
                    <p style="color: #94a3b8; font-size: 0.92rem; max-width: 600px; margin: 0 auto;">
                        Ask any question about model predictions, SHAP attributions, structural causal DAGs, 
                        or general principles of Trustworthy AI. Choose a suggested inquiry above or enter a question below.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.agent_chat_history:
                avatar = "🧑‍💻" if msg["role"] == "user" else "🛡️"
                with st.chat_message(msg["role"], avatar=avatar):
                    st.markdown(msg["content"])

        # 3. Chat Input & Processing
        user_typed_prompt = st.chat_input("Ask a question about predictions, SHAP, causal graph, or general AI...")
        active_prompt = prompt_to_send or user_typed_prompt

        if active_prompt:
            st.session_state.agent_chat_history.append({"role": "user", "content": active_prompt})

            if not is_connected:
                err_text = (
                    "⚠️ **Gemini API Key Required:** Please configure your `GEMINI_API_KEY` in the top configuration "
                    "card or in `.streamlit/secrets.toml` to interact with the Explainable AI Agent.\n\n"
                    "You can get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey)."
                )
                st.session_state.agent_chat_history.append({"role": "assistant", "content": err_text})
                st.rerun()
            else:
                with st.spinner("🤖 Reasoning over model evidence and synthesizing explanation..."):
                    agent_resp = query_explainable_ai_agent(
                        user_question=active_prompt,
                        history=st.session_state.agent_chat_history[:-1],
                        evidence=evidence,
                        api_key=effective_key,
                    )

                    if agent_resp.get("success"):
                        ans = agent_resp["text"]
                    else:
                        e_msg = agent_resp.get("error", "Unknown error")
                        e_type = agent_resp.get("error_type", "ERROR")
                        guidance = agent_resp.get("guidance", "Please retry.")
                        ans = f"❌ **Agent Error ({e_type}):** {e_msg}\n\n💡 **Guidance:** {guidance}"

                    st.session_state.agent_chat_history.append({"role": "assistant", "content": ans})
                    st.rerun()

        # 4. Action Row (Clear Chat)
        if st.session_state.agent_chat_history:
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            col_c_left, _ = st.columns([1.5, 4.5])
            with col_c_left:
                if st.button("🗑️ Clear Conversation", key="btn_clear_agent_chat", use_container_width=True):
                    st.session_state.agent_chat_history = []
                    st.rerun()

    with col_evidence_sidebar:
        # Connected Model Evidence Drawer
        pred_item = evidence.get("prediction", {})
        shap_item = evidence.get("shap", {})
        causal_item = evidence.get("causal", {})
        cf_item = evidence.get("counterfactual", {})

        top_shap_feat = (shap_item.get("top_features") or [{}])[0].get("feature", "Attendance")
        top_shap_shift = (shap_item.get("top_features") or [{}])[0].get("shap_contribution", "-0.342")

        st.markdown(
            f"""
            <div style="background: #1e293b; border: 1px solid #334155; border-radius: 14px; padding: 1.3rem;">
                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.8rem;">
                    <h4 style="margin: 0; color: #f8fafc; font-size: 1rem; display: flex; align-items: center; gap: 8px;">
                        📡 Connected Evidence
                    </h4>
                    <span class="status-pill" style="font-size: 0.72rem; padding: 2px 8px; border-color: #10b981; color: #6ee7b7;">
                        Grounded
                    </span>
                </div>
                <div style="font-size: 0.82rem; color: #94a3b8; line-height: 1.5; margin-bottom: 1rem;">
                    The LLM references this structured evidence in real-time when explaining predictions.
                </div>

                <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px; margin-bottom: 0.8rem;">
                    <div style="color: #818cf8; font-weight: 700; font-size: 0.78rem; text-transform: uppercase;">Active Dataset & Model</div>
                    <div style="color: #f8fafc; font-weight: 600; font-size: 0.88rem; margin-top: 2px;">{evidence['dataset_name']}</div>
                    <div style="color: #cbd5e1; font-size: 0.8rem; margin-top: 2px;">Model: <strong>{evidence['active_model']}</strong></div>
                    <div style="color: #94a3b8; font-size: 0.78rem;">Target: <code>{evidence['target_col']}</code> | {evidence['metrics_summary']}</div>
                </div>

                <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px; margin-bottom: 0.8rem;">
                    <div style="color: #38bdf8; font-weight: 700; font-size: 0.78rem; text-transform: uppercase;">Active Prediction Telemetry</div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 3px;">
                        <span style="color: #cbd5e1; font-size: 0.84rem;">Prediction:</span>
                        <span style="color: #f8fafc; font-weight: 800; font-size: 1rem;">{pred_item.get('outcome', 'N/A')}</span>
                    </div>
                    <div style="display: flex; justify-content: space-between; align-items: baseline; margin-top: 2px;">
                        <span style="color: #cbd5e1; font-size: 0.84rem;">Confidence:</span>
                        <span style="color: #34d399; font-weight: 700; font-size: 0.85rem;">{pred_item.get('confidence') or 'N/A'}</span>
                    </div>
                </div>

                <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px; margin-bottom: 0.8rem;">
                    <div style="color: #f472b6; font-weight: 700; font-size: 0.78rem; text-transform: uppercase;">SHAP Attribution Feed</div>
                    <div style="color: #cbd5e1; font-size: 0.8rem; margin-top: 3px;">
                        Expected Base: <code>{shap_item.get('base_value', '0.50')}</code>
                    </div>
                    <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 2px;">
                        Top Influencer: <strong>{top_shap_feat}</strong> ({top_shap_shift})
                    </div>
                </div>

                <div style="background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 10px; margin-bottom: 0.8rem;">
                    <div style="color: #fbbf24; font-weight: 700; font-size: 0.78rem; text-transform: uppercase;">Causal & Counterfactual Feeds</div>
                    <div style="color: #cbd5e1; font-size: 0.8rem; margin-top: 3px;">
                        Causal DAG: <strong>{len(causal_item.get('nodes', []))}</strong> vertices
                    </div>
                    <div style="color: #94a3b8; font-size: 0.78rem; margin-top: 2px;">
                        Counterfactual: <strong>{cf_item.get('original_prediction', 'FAIL')} ➔ {cf_item.get('modified_prediction', 'PASS')}</strong>
                    </div>
                </div>

                <div style="margin-top: 1rem;">
                    <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 6px;">Want to train a new model or tweak features?</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.button(
            "🚀 Open Dataset Analysis Workspace →",
            key="btn_jump_to_ds_agent",
            use_container_width=True,
            type="secondary",
            on_click=navigate_to,
            args=(WORKSPACE_DATASET,),
        )

"""
llm_service.py — Google Gemini API Backend for Explainable AI Agent.

This module provides the backend service layer for interacting with Google's
Gemini API using the official Google Gen AI Python SDK (`google-genai`).

Documentation:
    https://ai.google.dev/gemini-api/docs

Configuration Instructions:
    To use this service, obtain a free API key from Google AI Studio:
    https://aistudio.google.com/app/apikey

    1. Local Environment Variables:
       - In bash/zsh terminal:
         export GEMINI_API_KEY="your-api-key-here"
       - Or add to your ~/.zshrc or ~/.bashrc file.

    2. Streamlit Secrets (for local Streamlit app & Streamlit Cloud):
       - Create a file at `.streamlit/secrets.toml` in your project root:
         GEMINI_API_KEY = "your-api-key-here"
       - On Streamlit Community Cloud: Go to App Settings -> Secrets and add:
         GEMINI_API_KEY = "your-api-key-here"

Key Features:
    • Secure API key discovery (Environment variables, Streamlit secrets fallback).
    • Keys are never logged, printed, or exposed in error messages.
    • Conversational context support (multi-turn chat history).
    • Robust error handling:
        - Missing or invalid API keys
        - Rate limits and quota exhaustion (HTTP 429)
        - Network connectivity and timeout failures
        - Empty or safety-blocked model responses
        - Generic API errors
    • Decoupled from Streamlit UI (can run in scripts, tests, CLI, or web apps).
    • Defaults to free-tier supported Gemini models (`gemini-2.5-flash`).
"""

import os
import sys
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

# Import official Google Gen AI SDK
try:
    from google import genai
    from google.genai import errors, types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    genai = None
    errors = None
    types = None

logger = logging.getLogger(__name__)

# Default free-tier capable models
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
SUPPORTED_FREE_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]

DEFAULT_SYSTEM_INSTRUCTION = (
    "You are an expert Explainable AI (XAI) assistant. Your mission is to provide clear, "
    "trustworthy, and scientifically rigorous explanations for machine learning models, "
    "SHAP feature attributions, causal graphs, and counterfactual simulations. "
    "Distinguish between statistical correlation and causal mechanisms. "
    "Communicate in a structured, accessible, and insightful manner."
)

__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "SUPPORTED_FREE_MODELS",
    "DEFAULT_SYSTEM_INSTRUCTION",
    "GeminiServiceError",
    "GeminiAPIKeyError",
    "GeminiRateLimitError",
    "GeminiNetworkError",
    "GeminiEmptyResponseError",
    "mask_api_key",
    "get_gemini_api_key",
    "is_api_key_configured",
    "get_gemini_client",
    "normalize_chat_history",
    "format_history_for_ui",
    "send_gemini_message",
    "ExplainableAIChatSession",
    "extract_grounding_evidence",
    "format_evidence_prompt",
    "query_explainable_ai_agent",
    "test_gemini_connection",
]


# ===========================================================================
# Custom Exception Hierarchy
# ===========================================================================

class GeminiServiceError(Exception):
    """Base exception for all Gemini LLM service errors."""
    def __init__(self, message: str, error_code: Optional[str] = None, guidance: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "GEMINI_ERROR"
        self.guidance = guidance or "Check your Gemini API configuration and network connectivity."


class GeminiAPIKeyError(GeminiServiceError):
    """Raised when the Gemini API key is missing, unauthorized, or invalid."""
    def __init__(self, message: str, guidance: Optional[str] = None):
        default_guidance = (
            "Please configure your Gemini API key by setting the GEMINI_API_KEY environment variable "
            "(`export GEMINI_API_KEY='your-key'`) or adding it to `.streamlit/secrets.toml` "
            "as `GEMINI_API_KEY = 'your-key'`. Get a free key at https://aistudio.google.com/app/apikey."
        )
        super().__init__(message, error_code="API_KEY_ERROR", guidance=guidance or default_guidance)


class GeminiRateLimitError(GeminiServiceError):
    """Raised when Gemini API rate limits or quota thresholds are exceeded (HTTP 429)."""
    def __init__(self, message: str, guidance: Optional[str] = None):
        default_guidance = (
            "The Gemini API rate limit or free tier quota has been reached. Please wait a moment "
            "before retrying (e.g., 30–60 seconds). For higher quotas, consider linking a billing account."
        )
        super().__init__(message, error_code="RATE_LIMIT_ERROR", guidance=guidance or default_guidance)


class GeminiNetworkError(GeminiServiceError):
    """Raised when a network failure or socket connection error prevents communication."""
    def __init__(self, message: str, guidance: Optional[str] = None):
        default_guidance = (
            "Unable to reach Google Gemini API servers. Please check your internet connection, "
            "DNS settings, proxy configuration, or firewall rules."
        )
        super().__init__(message, error_code="NETWORK_ERROR", guidance=guidance or default_guidance)


class GeminiEmptyResponseError(GeminiServiceError):
    """Raised when the model returns an empty text or the candidate was blocked by safety filters."""
    def __init__(self, message: str, guidance: Optional[str] = None):
        default_guidance = (
            "The model returned an empty response or content was filtered by safety settings. "
            "Try rephrasing your prompt or reducing constraints."
        )
        super().__init__(message, error_code="EMPTY_RESPONSE_ERROR", guidance=guidance or default_guidance)


# ===========================================================================
# API Key Management & Sanitization
# ===========================================================================

def mask_api_key(key: Optional[str]) -> str:
    """Mask an API key for safe debugging output (never exposes the raw secret)."""
    if not key or not isinstance(key, str):
        return "<Not Configured>"
    key = key.strip()
    if len(key) <= 8:
        return "********"
    return f"{key[:4]}...{key[-4:]}"


def get_gemini_api_key(explicit_key: Optional[str] = None) -> Optional[str]:
    """Retrieve the Gemini API key securely from supported configuration sources.

    Lookup hierarchy:
    1. Explicit `explicit_key` passed by the caller.
    2. Environment variable: `GEMINI_API_KEY`.
    3. Secondary environment variable: `GOOGLE_API_KEY`.
    4. Streamlit secrets (`st.secrets["GEMINI_API_KEY"]`), if running in a Streamlit environment.

    Returns:
        The valid API key string, or None if no key is configured.
    """
    # 1. Explicit parameter
    if explicit_key and isinstance(explicit_key, str) and explicit_key.strip():
        return explicit_key.strip()

    # 2. Environment variable GEMINI_API_KEY
    env_key = os.environ.get("GEMINI_API_KEY")
    if env_key and isinstance(env_key, str) and env_key.strip():
        return env_key.strip()

    # 3. Environment variable GOOGLE_API_KEY (Google standard)
    env_google_key = os.environ.get("GOOGLE_API_KEY")
    if env_google_key and isinstance(env_google_key, str) and env_google_key.strip():
        return env_google_key.strip()

    # 4. Streamlit secrets (safely handled without hard dependency)
    try:
        import streamlit as st  # type: ignore
        if hasattr(st, "secrets"):
            if "GEMINI_API_KEY" in st.secrets:
                secret_key = st.secrets["GEMINI_API_KEY"]
                if secret_key and isinstance(secret_key, str) and secret_key.strip():
                    return secret_key.strip()
            if "GOOGLE_API_KEY" in st.secrets:
                secret_key = st.secrets["GOOGLE_API_KEY"]
                if secret_key and isinstance(secret_key, str) and secret_key.strip():
                    return secret_key.strip()
    except Exception:
        # Streamlit not installed or secrets file not present
        pass

    return None


def is_api_key_configured(explicit_key: Optional[str] = None) -> bool:
    """Check whether a Gemini API key is configured without exposing it."""
    return get_gemini_api_key(explicit_key) is not None


# ===========================================================================
# Client Initialization & Verification
# ===========================================================================

def get_gemini_client(api_key: Optional[str] = None) -> "genai.Client":
    """Initialize and return an official Google Gen AI Client instance.

    Raises:
        GeminiServiceError: If the `google-genai` SDK is not installed.
        GeminiAPIKeyError: If no API key is found in environment or secrets.
    """
    if not GENAI_AVAILABLE or genai is None:
        raise GeminiServiceError(
            "The official `google-genai` Python SDK is not installed in the current environment.",
            error_code="SDK_NOT_INSTALLED",
            guidance="Install it using: pip install google-genai",
        )

    resolved_key = get_gemini_api_key(api_key)
    if not resolved_key:
        raise GeminiAPIKeyError(
            "Gemini API key is not configured.",
            guidance=(
                "Set the GEMINI_API_KEY environment variable (`export GEMINI_API_KEY='your-key'`) "
                "or create `.streamlit/secrets.toml` with `GEMINI_API_KEY = 'your-key'`. "
                "Get a free API key at https://aistudio.google.com/app/apikey"
            ),
        )

    return genai.Client(api_key=resolved_key)


# ===========================================================================
# Conversational Context Normalization
# ===========================================================================

def normalize_chat_history(history: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Normalize conversational history from various formats into valid Gemini ContentDicts.

    Supports:
        - Streamlit format: [{"role": "user"|"assistant", "content": "..."}]
        - GenAI format: [{"role": "user"|"model", "parts": [{"text": "..."}]}]
        - Tuple format: [("user", "..."), ("model", "...")]

    Returns:
        A list of dicts suitable for `client.chats.create(history=...)`.
    """
    if not history:
        return []

    normalized = []
    for item in history:
        if isinstance(item, dict):
            raw_role = item.get("role", "user")
            # Map standard assistant role to Gemini's 'model' role
            role = "model" if raw_role in ("assistant", "model", "bot", "system") else "user"

            # Check if parts already structured
            if "parts" in item and isinstance(item["parts"], list):
                parts = []
                for p in item["parts"]:
                    if isinstance(p, dict) and "text" in p:
                        parts.append({"text": str(p["text"])})
                    elif isinstance(p, str):
                        parts.append({"text": p})
                if parts:
                    normalized.append({"role": role, "parts": parts})
                    continue

            # Check for standard content string
            content = item.get("content", "")
            if content:
                normalized.append({"role": role, "parts": [{"text": str(content)}]})

        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            raw_role, raw_text = item[0], item[1]
            role = "model" if str(raw_role).lower() in ("assistant", "model", "bot") else "user"
            normalized.append({"role": role, "parts": [{"text": str(raw_text)}]})

    return normalized


def format_history_for_ui(genai_history: List[Any]) -> List[Dict[str, str]]:
    """Convert Gemini chat history into standard UI-friendly dictionaries.

    Returns:
        List of dicts: `[{"role": "user"|"assistant", "content": "..."}]`
    """
    ui_history = []
    for msg in genai_history:
        role = "assistant" if getattr(msg, "role", "") == "model" or (isinstance(msg, dict) and msg.get("role") == "model") else "user"

        text_content = ""
        # Handle GenAI Content object
        if hasattr(msg, "parts") and msg.parts:
            parts_texts = []
            for p in msg.parts:
                t = getattr(p, "text", None) or (p.get("text") if isinstance(p, dict) else str(p))
                if t:
                    parts_texts.append(t)
            text_content = "\n".join(parts_texts)
        elif isinstance(msg, dict):
            if "parts" in msg:
                text_content = "\n".join(
                    p.get("text", "") if isinstance(p, dict) else str(p) for p in msg["parts"]
                )
            elif "content" in msg:
                text_content = str(msg["content"])

        if text_content:
            ui_history.append({"role": role, "content": text_content})

    return ui_history


# ===========================================================================
# Core API Interaction Functions
# ===========================================================================

def send_gemini_message(
    prompt: str,
    history: Optional[List[Any]] = None,
    system_instruction: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
    raise_on_error: bool = False,
) -> Dict[str, Any]:
    """Send a user question to Google Gemini with conversational context support.

    Args:
        prompt: The user question or instruction to send.
        history: Optional conversation history to maintain multi-turn context.
        system_instruction: Optional system persona instruction.
        model: Model identifier (defaults to `gemini-2.5-flash`).
        api_key: Optional explicit API key (defaults to env or secrets).
        temperature: Sampling temperature (0.0 to 2.0, default 0.7).
        raise_on_error: If True, raises custom exceptions; if False, returns error dict.

    Returns:
        Dictionary containing:
            - "success" (bool): True if generation succeeded.
            - "text" (str): The response text generated by Gemini.
            - "model" (str): The model used for generation.
            - "history" (list): Updated conversational history formatted for UI.
            - "error" (str|None): Error message if failed.
            - "error_type" (str|None): Machine-readable error code.
            - "guidance" (str|None): User-friendly remediation instructions.
    """
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        err = GeminiEmptyResponseError("Empty prompt provided. Please enter a valid question.")
        if raise_on_error:
            raise err
        return {
            "success": False,
            "text": "",
            "model": model or DEFAULT_GEMINI_MODEL,
            "history": history or [],
            "error": err.message,
            "error_type": err.error_code,
            "guidance": err.guidance,
        }

    target_model = model or os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL

    try:
        # 1. Initialize client
        client = get_gemini_client(api_key=api_key)

        # 2. Configure generation parameters
        sys_instruction = system_instruction or DEFAULT_SYSTEM_INSTRUCTION
        config = types.GenerateContentConfig(
            system_instruction=sys_instruction,
            temperature=float(temperature),
        )

        # 3. Handle conversational context
        norm_history = normalize_chat_history(history)
        chat = client.chats.create(
            model=target_model,
            config=config,
            history=norm_history if norm_history else None,
        )

        # 4. Dispatch user message
        response = chat.send_message(clean_prompt)

        # 5. Validate response content
        response_text = response.text if response and hasattr(response, "text") and response.text else None
        if not response_text:
            # Check candidate details if available
            finish_reason = "UNKNOWN"
            if hasattr(response, "candidates") and response.candidates:
                c0 = response.candidates[0]
                finish_reason = str(getattr(c0, "finish_reason", "FILTERED"))

            raise GeminiEmptyResponseError(
                f"Gemini returned an empty response (finish reason: {finish_reason}).",
                guidance=(
                    "The model generated no content, potentially due to safety filters or token limits. "
                    "Try rephrasing your prompt with more context."
                ),
            )

        # 6. Extract updated conversational history
        raw_history = chat.get_history()
        updated_ui_history = format_history_for_ui(raw_history)

        return {
            "success": True,
            "text": response_text.strip(),
            "model": target_model,
            "history": updated_ui_history,
            "error": None,
            "error_type": None,
            "guidance": None,
        }

    except GeminiAPIKeyError as e:
        if raise_on_error:
            raise e
        return {
            "success": False,
            "text": "",
            "model": target_model,
            "history": history or [],
            "error": e.message,
            "error_type": e.error_code,
            "guidance": e.guidance,
        }

    except GeminiEmptyResponseError as e:
        if raise_on_error:
            raise e
        return {
            "success": False,
            "text": "",
            "model": target_model,
            "history": history or [],
            "error": e.message,
            "error_type": e.error_code,
            "guidance": e.guidance,
        }

    except Exception as e:
        # Translate and categorize SDK / network exceptions
        classified_err = _classify_gemini_exception(e, target_model)
        if raise_on_error:
            raise classified_err
        return {
            "success": False,
            "text": "",
            "model": target_model,
            "history": history or [],
            "error": classified_err.message,
            "error_type": classified_err.error_code,
            "guidance": classified_err.guidance,
        }


def _classify_gemini_exception(exc: Exception, model_name: str) -> GeminiServiceError:
    """Classify low-level exceptions into structured domain exceptions."""
    err_str = str(exc)
    err_lower = err_str.lower()

    # Check for google.genai.errors.APIError attributes
    status_code = getattr(exc, "code", None)
    if status_code is None and hasattr(exc, "status_code"):
        status_code = getattr(exc, "status_code")

    # 1. Rate Limit (HTTP 429) / Quota Exhaustion
    if status_code == 429 or "resource_exhausted" in err_lower or "rate limit" in err_lower or "quota" in err_lower:
        return GeminiRateLimitError(
            f"Gemini API rate limit reached (HTTP 429) for model '{model_name}'.",
            guidance=(
                "You have temporarily reached the free-tier request rate limit (RPM/RPD). "
                "Please wait 30–60 seconds before submitting another question."
            ),
        )

    # 2. Invalid or Unauthorized API Key (HTTP 400, 401, 403)
    if (
        status_code in (400, 401, 403)
        and any(k in err_lower for k in ("api key", "unauthorized", "invalid", "permission_denied", "api_key_invalid"))
    ) or "api key not valid" in err_lower:
        return GeminiAPIKeyError(
            "The configured Gemini API key is invalid or unauthorized.",
            guidance=(
                "Verify your API key at https://aistudio.google.com/app/apikey. "
                "Ensure GEMINI_API_KEY is set correctly without extra quotes or spaces."
            ),
        )

    # 3. Network or Connection Failures
    if (
        isinstance(exc, (ConnectionError, TimeoutError, OSError))
        or "connect" in err_lower
        or "timeout" in err_lower
        or "nodename nor servname" in err_lower
        or "network" in err_lower
        or "dns" in err_lower
    ):
        return GeminiNetworkError(
            f"Network connectivity error while connecting to Gemini API: {exc}",
            guidance="Check your internet connection, DNS resolution, or proxy configurations.",
        )

    # 4. Model not found or unavailable
    if status_code == 404 or "not found" in err_lower:
        return GeminiServiceError(
            f"The requested Gemini model '{model_name}' was not found or is unavailable.",
            error_code="MODEL_NOT_FOUND",
            guidance=f"Try switching to a supported free-tier model such as '{DEFAULT_GEMINI_MODEL}' or 'gemini-2.0-flash'.",
        )

    # 5. Generic API or Server Error
    return GeminiServiceError(
        f"Gemini API error occurred: {exc}",
        error_code="API_ERROR",
        guidance="An unexpected error occurred while communicating with the Gemini service.",
    )


# ===========================================================================
# Stateful Chat Session Class
# ===========================================================================

class GeminiChatSession:
    """A stateful conversational session manager for the Explainable AI Agent.

    Maintains multi-turn context and allows sequential queries.
    Completely decoupled from Streamlit, but easily stored in Streamlit session state.
    """

    def __init__(
        self,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
    ):
        self.system_instruction = system_instruction or DEFAULT_SYSTEM_INSTRUCTION
        self.model = model or DEFAULT_GEMINI_MODEL
        self.api_key = api_key
        self.temperature = temperature
        self._history: List[Dict[str, str]] = []

    @property
    def history(self) -> List[Dict[str, str]]:
        """Return the conversation history in UI-friendly format."""
        return list(self._history)

    def send(self, prompt: str, raise_on_error: bool = False) -> Dict[str, Any]:
        """Send a message within this continuous conversational session."""
        result = send_gemini_message(
            prompt=prompt,
            history=self._history,
            system_instruction=self.system_instruction,
            model=self.model,
            api_key=self.api_key,
            temperature=self.temperature,
            raise_on_error=raise_on_error,
        )

        if result.get("success"):
            self._history = result.get("history", [])

        return result

    def clear(self):
        """Reset conversation history for a fresh session."""
        self._history = []


# ===========================================================================
# Data-Aware Grounding & Evidence Extraction
# ===========================================================================

def extract_grounding_evidence(
    dataset_source: str = "demo",
    uploaded_df: Optional[Any] = None,
    uploaded_filename: Optional[str] = None,
    uploaded_model_results: Optional[Dict[str, Any]] = None,
    uploaded_shap_cache: Optional[Dict[str, Any]] = None,
    uploaded_causal_graph: Optional[Dict[str, Any]] = None,
    uploaded_cf_result: Optional[Dict[str, Any]] = None,
    demo_cf_result: Optional[Dict[str, Any]] = None,
    latest_prediction: Optional[Dict[str, Any]] = None,
    latest_shap: Optional[Dict[str, Any]] = None,
    demo_model_name: str = "Random Forest Classifier",
    demo_metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract and minimize structured evidence from active ML, SHAP, and Causal modules.

    Collects:
        1. Model type, target, and performance metrics.
        2. Latest observation input values and model prediction/probability.
        3. SHAP local and global feature attribution rankings.
        4. Qualitative structural causal DAG hypotheses.
        5. Counterfactual what-if simulation results.

    Returns:
        A structured dictionary ready for formatting into the agent's context.
    """
    is_upload = (dataset_source == "upload" and uploaded_df is not None)

    # 1. Dataset & Model Metadata
    if is_upload and uploaded_model_results:
        res = uploaded_model_results
        dataset_name = uploaded_filename or "Uploaded Custom Dataset"
        problem_type = res.get("problem_type", "Classification")
        target_col = res.get("target_col", "Target")
        active_model = res.get("best_name", "Best Pipeline")
        bm = res.get("best_metrics", {})
        if problem_type == "Classification":
            metrics_str = f"Accuracy: {bm.get('accuracy', 0):.1%}, F1: {bm.get('f1', 0):.1%}"
        else:
            metrics_str = f"R²: {bm.get('r2', 0):.4f}, RMSE: {bm.get('rmse', 0):,.2f}"
        feature_cols = res.get("feature_cols", [])
    else:
        dataset_name = "Student Performance (Demo Dataset)"
        problem_type = "Classification"
        target_col = "result"
        active_model = demo_model_name
        bm = demo_metrics or {"f1": 0.88, "accuracy": 0.89}
        metrics_str = f"F1: {bm.get('f1', 0):.1%}, Accuracy: {bm.get('accuracy', 0):.1%}"
        feature_cols = ["study_hours", "attendance", "previous_scores", "sleep_hours", "assignments_completed", "class_participation"]

    # 2. Prediction Evidence
    pred_data = None
    cf_res = uploaded_cf_result if is_upload else demo_cf_result

    if latest_prediction:
        pred_data = {
            "outcome": str(latest_prediction.get("predicted_label", "Unknown")),
            "confidence": f"{latest_prediction['confidence']:.1%}" if latest_prediction.get("confidence") is not None else None,
            "probabilities": latest_prediction.get("probabilities"),
            "inputs": {str(k): v for k, v in latest_prediction.get("inputs", {}).items()},
        }
    elif latest_shap and latest_shap.get("prediction_label"):
        pred_data = {
            "outcome": str(latest_shap.get("prediction_label")),
            "confidence": f"{latest_shap['prediction_probability']:.1%}" if latest_shap.get("prediction_probability") is not None else None,
            "probabilities": latest_shap.get("class_probabilities"),
            "inputs": {str(k): v for k, v in latest_shap.get("input_values", {}).items()} if latest_shap.get("input_values") else None,
        }
    elif cf_res:
        pred_data = {
            "outcome": str(cf_res.get("original_label", "Unknown")),
            "confidence": f"{cf_res['original_confidence']:.1%}" if cf_res.get("original_confidence") is not None else None,
            "probabilities": cf_res.get("original_class_probs"),
            "inputs": cf_res.get("original_inputs"),
        }
    else:
        # Default baseline context for demo
        pred_data = {
            "outcome": "FAIL",
            "confidence": "82.4%",
            "probabilities": {"FAIL": 0.824, "PASS": 0.176},
            "inputs": {
                "Study Hours": 2.0,
                "Attendance": "50%",
                "Previous Scores": 45.0,
                "Sleep Hours": 6.0,
                "Assignments Completed": 3,
                "Class Participation": 3,
            },
        }

    # 3. SHAP Evidence
    shap_data = None
    if latest_shap and latest_shap.get("feature_effects"):
        top_effects = sorted(
            latest_shap["feature_effects"],
            key=lambda x: abs(x.get("shap_value", 0.0)),
            reverse=True,
        )[:5]
        shap_data = {
            "base_value": round(float(latest_shap.get("base_value", 0.0)), 3),
            "top_features": [
                {
                    "feature": str(eff.get("feature", "")),
                    "value": str(eff.get("feature_value", "")),
                    "shap_contribution": f"{eff.get('shap_value', 0.0):+.3f}",
                    "direction": "Pushed prediction towards outcome" if eff.get("shap_value", 0) > 0 else "Pushed prediction away from outcome",
                }
                for eff in top_effects
            ],
            "narrative": latest_shap.get("narrative"),
        }
    elif is_upload and uploaded_shap_cache and uploaded_shap_cache.get("explanation"):
        exp = uploaded_shap_cache["explanation"]
        try:
            mean_abs = exp.abs.mean(0).values if hasattr(exp, "abs") else []
            feat_names = list(getattr(exp, "feature_names", []))
            top_pairs = sorted(zip(feat_names, mean_abs), key=lambda x: x[1], reverse=True)[:5]
            shap_data = {
                "base_value": "Model training expectation",
                "top_features": [
                    {"feature": fn, "mean_abs_shap": f"{float(val):.3f}"}
                    for fn, val in top_pairs
                ],
                "narrative": "Global feature importance derived from fitted explainer.",
            }
        except Exception:
            pass

    if not shap_data:
        # Demo SHAP attributions
        shap_data = {
            "base_value": 0.50,
            "top_features": [
                {"feature": "Attendance", "value": "50%", "shap_contribution": "-0.342", "direction": "Strongly drove FAIL prediction"},
                {"feature": "Study Hours", "value": "2.0 hrs", "shap_contribution": "-0.285", "direction": "Drove FAIL prediction"},
                {"feature": "Previous Scores", "value": "45.0", "shap_contribution": "-0.190", "direction": "Drove FAIL prediction"},
                {"feature": "Assignments Completed", "value": "3/10", "shap_contribution": "-0.082", "direction": "Subtle negative impact"},
                {"feature": "Sleep Hours", "value": "6.0 hrs", "shap_contribution": "+0.015", "direction": "Minor positive baseline"},
            ],
            "narrative": "Low attendance (50%) and low study hours (2 hrs) produced the strongest negative SHAP shifts pulling the outcome below the passing boundary.",
        }

    # 4. Causal Graph Hypotheses
    causal_data = None
    if is_upload and uploaded_causal_graph and "edges" in uploaded_causal_graph:
        edges = uploaded_causal_graph.get("edges", [])
        edge_strs = [f"{e.get('source')} ➔ {e.get('target')}" for e in edges[:8]]
        causal_data = {
            "nodes": uploaded_causal_graph.get("nodes", []),
            "assumed_edges": edge_strs or ["No custom causal edges defined yet."],
            "disclaimer": "All relationships are user-configured hypotheses; not empirically verified causal proofs.",
        }
    else:
        causal_data = {
            "nodes": ["Study Hours", "Attendance", "Previous Scores", "Sleep Hours", "Assignments Completed", "Class Participation", "Exam Score / Result"],
            "assumed_edges": [
                "Study Hours ➔ Assignments Completed (Direct)",
                "Study Hours ➔ Exam Score (Direct)",
                "Attendance ➔ Class Participation (Direct)",
                "Attendance ➔ Exam Score (Direct)",
                "Previous Scores ➔ Exam Score (Confounder / Prior)",
                "Sleep Hours ➔ Study Hours (Upstream factor)",
            ],
            "disclaimer": "Causal DAG reflects domain theoretical hypotheses. Observational associations do not prove unconfounded causality.",
        }

    # 5. Counterfactual Simulation Evidence
    cf_data = None
    if cf_res:
        cf_data = {
            "original_prediction": str(cf_res.get("original_label", "Unknown")),
            "original_confidence": f"{cf_res['original_confidence']:.1%}" if cf_res.get("original_confidence") is not None else None,
            "modified_prediction": str(cf_res.get("modified_label", "Unknown")),
            "modified_confidence": f"{cf_res['modified_confidence']:.1%}" if cf_res.get("modified_confidence") is not None else None,
            "prediction_flipped": cf_res.get("prediction_changed", False),
            "changed_features": cf_res.get("changed_features", []),
            "changes_summary": [
                f"{cd.get('Feature')}: {cd.get('Original Value')} ➔ {cd.get('Modified Value')} ({cd.get('Shift')})"
                for cd in cf_res.get("changed_details", [])
            ],
            "narrative": cf_res.get("narrative", ""),
        }
    else:
        # Demo counterfactual baseline
        cf_data = {
            "original_prediction": "FAIL (82.4% confidence)",
            "modified_prediction": "PASS (88.5% confidence)",
            "prediction_flipped": True,
            "changed_features": ["Attendance", "Study Hours"],
            "changes_summary": [
                "Attendance: 50% ➔ 85% (+35)",
                "Study Hours: 2.0 ➔ 6.0 (+4.0)",
            ],
            "narrative": "Simulating an increase in Attendance from 50% to 85% and Study Hours from 2 to 6 flipped the model prediction from FAIL to PASS.",
        }

    return {
        "dataset_name": dataset_name,
        "problem_type": problem_type,
        "target_col": target_col,
        "active_model": active_model,
        "metrics_summary": metrics_str,
        "feature_cols": feature_cols,
        "prediction": pred_data,
        "shap": shap_data,
        "causal": causal_data,
        "counterfactual": cf_data,
    }


def format_evidence_prompt(evidence: Dict[str, Any]) -> str:
    """Format structured evidence into a minimized, readable context block."""
    lines = [
        "### STRUCTURED MODEL EVIDENCE (Grounding Context):",
        f"- **Dataset Environment**: `{evidence.get('dataset_name')}`",
        f"- **Task & Target**: {evidence.get('problem_type')} on target `{evidence.get('target_col')}`",
        f"- **Active Model**: {evidence.get('active_model')} ({evidence.get('metrics_summary')})",
    ]

    # Prediction
    pred = evidence.get("prediction")
    if pred:
        conf_str = f" (Confidence: {pred['confidence']})" if pred.get("confidence") else ""
        lines.append(f"- **Current Model Prediction**: **{pred.get('outcome')}**{conf_str}")
        if pred.get("probabilities"):
            probs_str = ", ".join(f"{k}: {v:.1%}" if isinstance(v, (int, float)) else f"{k}: {v}" for k, v in pred["probabilities"].items())
            lines.append(f"  - Class Probabilities: {probs_str}")
        if pred.get("inputs"):
            inputs_str = ", ".join(f"{k}: {v}" for k, v in list(pred["inputs"].items())[:8])
            lines.append(f"  - Input Attributes: {inputs_str}")

    # SHAP
    shap = evidence.get("shap")
    if shap:
        lines.append("- **SHAP Game-Theoretic Feature Attributions**:")
        if shap.get("base_value") is not None:
            lines.append(f"  - Base Expected Value: {shap['base_value']}")
        for f in shap.get("top_features", [])[:5]:
            feat_name = f.get("feature")
            val = f.get("value")
            contrib = f.get("shap_contribution") or f.get("mean_abs_shap")
            dir_str = f" [{f.get('direction')}]" if f.get("direction") else ""
            lines.append(f"  - `{feat_name}` ({val}): SHAP shift {contrib}{dir_str}")
        if shap.get("narrative"):
            lines.append(f"  - Attribution Summary: {shap['narrative']}")

    # Causal
    causal = evidence.get("causal")
    if causal:
        lines.append("- **Structural Causal Graph Hypotheses**:")
        lines.append(f"  - Direct Assumed Paths: {'; '.join(causal.get('assumed_edges', [])[:6])}")
        lines.append(f"  - Methodological Note: {causal.get('disclaimer')}")

    # Counterfactual
    cf = evidence.get("counterfactual")
    if cf:
        lines.append("- **Counterfactual Sensitivity Simulation**:")
        lines.append(f"  - Baseline ➔ Counterfactual: {cf.get('original_prediction')} ➔ **{cf.get('modified_prediction')}**")
        if cf.get("changes_summary"):
            lines.append(f"  - Tested Input Variations: {', '.join(cf['changes_summary'])}")
        if cf.get("narrative"):
            lines.append(f"  - Simulation Narrative: {cf['narrative']}")

    return "\n".join(lines)


def query_explainable_ai_agent(
    user_question: str,
    history: Optional[List[Any]] = None,
    evidence: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a data-aware grounded query to the Explainable AI Agent.

    When evidence is provided, grounds Gemini's reasoning in model metrics, SHAP,
    causal graph assumptions, and counterfactuals. When answering general questions,
    responds using core domain knowledge without fabricating numbers.
    """
    clean_q = (user_question or "").strip()
    if not clean_q:
        return {
            "success": False,
            "text": "",
            "error": "Empty question provided.",
            "error_type": "EMPTY_PROMPT",
            "guidance": "Please enter a valid question in the chat input.",
        }

    # Build grounded system instruction
    evidence_block = format_evidence_prompt(evidence) if evidence else "No active dataset evidence loaded. Operating in General Knowledge Mode."

    system_instruction = (
        "You are an expert Explainable AI (XAI) and Trustworthy AI Agent. "
        "Your mission is to provide rigorous, scientifically transparent explanations of machine learning "
        "models, SHAP attributions, causal graphs, and counterfactual 'what-if' simulations.\n\n"
        "RESPONSE GUIDELINES:\n"
        "1. When the user asks about a specific model prediction or outcome (e.g. 'Why did the model predict FAIL?'), "
        "structure your explanation into these distinct sections:\n"
        "   - **🎯 Prediction Summary**: What the model predicted, including probability/confidence if available.\n"
        "   - **⚡ Key Influencing Features**: The main input attributes that led to this decision.\n"
        "   - **🔍 SHAP Evidence**: Explicit game-theoretic contributions pushing the outcome away from expected baseline.\n"
        "   - **🔬 Causal Graph Assumptions**: What the structural causal DAG assumes about domain relationships, explicitly noting that these are qualitative hypotheses rather than proven unconfounded causality.\n"
        "   - **🔄 What-If Alternatives**: What minimal or counterfactual changes could shift or invert the model's decision.\n\n"
        "2. SCIENTIFIC INTEGRITY RULES:\n"
        "   - NEVER invent or hallucinate prediction numbers, SHAP values, or features not present in the evidence.\n"
        "   - ALWAYS distinguish statistical correlation and model sensitivity from real-world causal effects.\n"
        "   - When answering general questions about AI, ML, XAI, Causal AI, or Trustworthy AI, provide structured, clear educational answers.\n"
        "   - Do NOT expose internal chain-of-thought or raw JSON payloads. Provide polished Markdown.\n\n"
        f"{evidence_block}"
    )

    return send_gemini_message(
        prompt=clean_q,
        history=history,
        system_instruction=system_instruction,
        model=model,
        api_key=api_key,
        temperature=0.6,
        raise_on_error=False,
    )


# ===========================================================================
# Self-Test & Diagnostic Utility
# ===========================================================================

def test_gemini_connection(
    api_key: Optional[str] = None,
    test_question: str = "Explain what Explainable AI (XAI) is in one short sentence.",
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Test the Gemini API connection using a simple question.

    Returns diagnostic information, connection status, latency, and sample response.
    """
    import time

    resolved_key = get_gemini_api_key(api_key)
    masked_key = mask_api_key(resolved_key)
    target_model = model or DEFAULT_GEMINI_MODEL

    print(f"\n{'=' * 65}")
    print("🔍 GOOGLE GEMINI API CONNECTION TEST")
    print(f"{'=' * 65}")
    print(f"• SDK Available:    {'✅ google-genai installed' if GENAI_AVAILABLE else '❌ google-genai NOT installed'}")
    print(f"• Target Model:     {target_model}")
    print(f"• API Key Status:   {'✅ Configured (' + masked_key + ')' if resolved_key else '❌ Not Found'}")
    print(f"• Test Question:    \"{test_question}\"")
    print(f"{'-' * 65}")

    if not resolved_key:
        print("❌ Cannot test live API: GEMINI_API_KEY is not set.")
        print("\n📖 Configuration Instructions:")
        print("  1. Get a free API key at: https://aistudio.google.com/app/apikey")
        print("  2. Set environment variable: export GEMINI_API_KEY='your-key'")
        print("  3. Or create .streamlit/secrets.toml with: GEMINI_API_KEY = 'your-key'")
        print(f"{'=' * 65}\n")
        return {
            "success": False,
            "status": "MISSING_API_KEY",
            "message": "GEMINI_API_KEY is not configured.",
            "masked_key": masked_key,
            "model": target_model,
        }

    start_time = time.time()
    result = send_gemini_message(
        prompt=test_question,
        model=target_model,
        api_key=resolved_key,
        raise_on_error=False,
    )
    elapsed = time.time() - start_time

    if result["success"]:
        print(f"✅ Connection Succeeded in {elapsed:.2f}s!")
        print(f"\n🤖 Gemini Response:\n{result['text']}")
        print(f"\n{'=' * 65}\n")
        return {
            "success": True,
            "status": "CONNECTED",
            "latency_seconds": round(elapsed, 2),
            "response": result["text"],
            "model": target_model,
            "masked_key": masked_key,
        }
    else:
        print(f"❌ Connection Failed ({result['error_type']}) after {elapsed:.2f}s:")
        print(f"  Error:    {result['error']}")
        print(f"  Guidance: {result['guidance']}")
        print(f"{'=' * 65}\n")
        return {
            "success": False,
            "status": result["error_type"],
            "error": result["error"],
            "guidance": result["guidance"],
            "latency_seconds": round(elapsed, 2),
            "model": target_model,
            "masked_key": masked_key,
        }


if __name__ == "__main__":
    # When run directly, perform connection test
    test_gemini_connection()


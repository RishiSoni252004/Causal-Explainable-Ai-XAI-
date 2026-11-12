"""
test_llm_service.py — Unit tests for Gemini API backend service.
Tests key resolution, conversational history, exception handling, and error categorization.
"""

import os
import unittest
from unittest.mock import patch, MagicMock

from src.llm_service import (
    get_gemini_api_key,
    mask_api_key,
    is_api_key_configured,
    get_gemini_client,
    normalize_chat_history,
    format_history_for_ui,
    send_gemini_message,
    GeminiChatSession,
    GeminiServiceError,
    GeminiAPIKeyError,
    GeminiRateLimitError,
    GeminiNetworkError,
    GeminiEmptyResponseError,
    _classify_gemini_exception,
    test_gemini_connection,
    DEFAULT_GEMINI_MODEL,
)


class TestGeminiKeyManagement(unittest.TestCase):
    def test_explicit_key_priority(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env_key_123"}):
            key = get_gemini_api_key(explicit_key="explicit_key_456")
            self.assertEqual(key, "explicit_key_456")

    def test_environment_variable_gemini_key(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env_gemini_test_key"}, clear=True):
            key = get_gemini_api_key()
            self.assertEqual(key, "env_gemini_test_key")

    def test_environment_variable_google_key_fallback(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "google_test_key"}, clear=True):
            key = get_gemini_api_key()
            self.assertEqual(key, "google_test_key")

    def test_no_key_configured(self):
        with patch.dict(os.environ, {}, clear=True), patch("streamlit.secrets", {}, create=True):
            key = get_gemini_api_key()
            self.assertIsNone(key)
            self.assertFalse(is_api_key_configured())

    def test_mask_api_key(self):
        self.assertEqual(mask_api_key(""), "<Not Configured>")
        self.assertEqual(mask_api_key(None), "<Not Configured>")
        self.assertEqual(mask_api_key("12345"), "********")
        masked = mask_api_key("AIzaSyD-LongSecretKey123456789")
        self.assertTrue(masked.startswith("AIza"))
        self.assertTrue(masked.endswith("6789"))
        self.assertNotIn("LongSecretKey", masked)


class TestChatHistoryNormalization(unittest.TestCase):
    def test_normalize_empty_history(self):
        self.assertEqual(normalize_chat_history([]), [])
        self.assertEqual(normalize_chat_history(None), [])

    def test_normalize_streamlit_format(self):
        ui_history = [
            {"role": "user", "content": "What is SHAP?"},
            {"role": "assistant", "content": "SHAP explains model outputs based on Shapley values."},
        ]
        norm = normalize_chat_history(ui_history)
        self.assertEqual(len(norm), 2)
        self.assertEqual(norm[0]["role"], "user")
        self.assertEqual(norm[0]["parts"][0]["text"], "What is SHAP?")
        self.assertEqual(norm[1]["role"], "model")
        self.assertEqual(norm[1]["parts"][0]["text"], "SHAP explains model outputs based on Shapley values.")

    def test_normalize_tuple_format(self):
        tuples = [
            ("user", "Hello"),
            ("assistant", "Hi there"),
        ]
        norm = normalize_chat_history(tuples)
        self.assertEqual(len(norm), 2)
        self.assertEqual(norm[0]["role"], "user")
        self.assertEqual(norm[1]["role"], "model")

    def test_format_history_for_ui(self):
        genai_hist = [
            {"role": "user", "parts": [{"text": "Question 1"}]},
            {"role": "model", "parts": [{"text": "Answer 1"}]},
        ]
        ui_formatted = format_history_for_ui(genai_hist)
        self.assertEqual(len(ui_formatted), 2)
        self.assertEqual(ui_formatted[0], {"role": "user", "content": "Question 1"})
        self.assertEqual(ui_formatted[1], {"role": "assistant", "content": "Answer 1"})


class TestGeminiErrorHandling(unittest.TestCase):
    def test_empty_prompt_handling(self):
        result = send_gemini_message("", api_key="any_key", raise_on_error=False)
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "EMPTY_RESPONSE_ERROR")

    def test_missing_api_key_handling(self):
        with patch.dict(os.environ, {}, clear=True), patch("streamlit.secrets", {}, create=True):
            result = send_gemini_message("Hello", api_key=None, raise_on_error=False)
            self.assertFalse(result["success"])
            self.assertEqual(result["error_type"], "API_KEY_ERROR")
            self.assertIn("not configured", result["error"])

            with self.assertRaises(GeminiAPIKeyError):
                send_gemini_message("Hello", api_key=None, raise_on_error=True)

    def test_invalid_api_key_rejection(self):
        """Test with an explicitly invalid API key to verify SDK error classification."""
        result = send_gemini_message(
            prompt="Hello",
            api_key="AIzaSyDummyInvalidKeyForTest999",
            raise_on_error=False,
        )
        self.assertFalse(result["success"])
        self.assertEqual(result["error_type"], "API_KEY_ERROR")
        self.assertIn("invalid", result["error"].lower())

    def test_rate_limit_classification(self):
        class Mock429(Exception):
            code = 429
            def __str__(self):
                return "Resource has been exhausted (e.g. check quota)."

        err = _classify_gemini_exception(Mock429(), "gemini-2.5-flash")
        self.assertIsInstance(err, GeminiRateLimitError)
        self.assertEqual(err.error_code, "RATE_LIMIT_ERROR")

    def test_network_error_classification(self):
        exc = ConnectionError("Failed to establish a new connection: [Errno 8] nodename nor servname provided")
        err = _classify_gemini_exception(exc, "gemini-2.5-flash")
        self.assertIsInstance(err, GeminiNetworkError)
        self.assertEqual(err.error_code, "NETWORK_ERROR")


class TestGeminiChatSession(unittest.TestCase):
    def test_session_lifecycle(self):
        session = GeminiChatSession(
            system_instruction="You are a helpful assistant.",
            model="gemini-2.5-flash",
            api_key="dummy_key",
        )
        self.assertEqual(session.history, [])
        session.clear()
        self.assertEqual(session.history, [])


class TestConnectionDiagnostic(unittest.TestCase):
    def test_connection_diagnostic_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch("src.llm_service.get_gemini_api_key", return_value=None):
                diag = test_gemini_connection(api_key=None)
                self.assertFalse(diag["success"])
                self.assertEqual(diag["status"], "MISSING_API_KEY")


if __name__ == "__main__":
    unittest.main()


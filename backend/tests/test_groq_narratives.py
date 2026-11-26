"""Tests for Groq narrative calls."""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import app.services.groq_narratives as gn

PRED = {"probability": 0.8, "top_features": [{"feature": "bmi", "value": 31.0, "shap_value": 0.12}]}


def _client_returning(status, body):
    resp = MagicMock(status_code=status)
    resp.json.return_value = body
    client = MagicMock()
    client.post = AsyncMock(return_value=resp)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, client


def test_payload_uses_configured_model_and_low_reasoning():
    with patch.object(gn, "GROQ_MODEL", "openai/gpt-oss-120b"):
        payload = gn._groq_payload("hi", 500)
    assert payload["model"] == "openai/gpt-oss-120b"
    assert payload["reasoning_effort"] == "low"


def test_payload_skips_reasoning_for_other_models():
    with patch.object(gn, "GROQ_MODEL", "llama-3.1-8b-instant"):
        assert "reasoning_effort" not in gn._groq_payload("hi", 500)


def test_returns_model_text_on_success():
    ctx, _ = _client_returning(200, {"choices": [{"message": {"content": " Risk is high. "}}]})
    with patch.object(gn, "GROQ_API_KEY", "k"), patch("httpx.AsyncClient", return_value=ctx):
        out = asyncio.run(gn.generate_shap_narrative({}, PRED, "er_visit_30d"))
    assert out == "Risk is high."


def test_falls_back_and_logs_on_error_status(caplog):
    ctx, _ = _client_returning(404, {"error": {"message": "model decommissioned"}})
    with patch.object(gn, "GROQ_API_KEY", "k"), patch("httpx.AsyncClient", return_value=ctx), \
            caplog.at_level("WARNING"):
        out = asyncio.run(gn.generate_shap_narrative({}, PRED, "er_visit_30d"))
    assert out.startswith("This patient has a high er visit 30d risk")
    assert "Groq returned 404" in caplog.text


def test_falls_back_on_empty_content():
    ctx, _ = _client_returning(200, {"choices": [{"message": {"content": ""}}]})
    with patch.object(gn, "GROQ_API_KEY", "k"), patch("httpx.AsyncClient", return_value=ctx):
        out = asyncio.run(gn.generate_shap_narrative({}, PRED, "er_visit_30d"))
    assert out.startswith("This patient has a high")

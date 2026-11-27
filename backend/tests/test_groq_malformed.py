"""A 200 reply with a broken body must become ProviderError, so the fallback chain runs instead of a 500."""
import httpx
import pytest

from app.services import llm_providers as lp


@pytest.mark.parametrize("body", [b"not json", b"{}", b'{"choices": []}', b'{"choices": [{"message": null}]}'])
def test_malformed_groq_body_is_provider_error(monkeypatch, body):
    def fake_post(self, url, **kw):
        return httpx.Response(200, content=body, request=httpx.Request("POST", url))
    monkeypatch.setattr(httpx.Client, "post", fake_post)
    p = lp.GroqProvider(api_key="test-key")
    with pytest.raises(lp.ProviderError):
        p.complete("hi")

"""LLM providers behind one small interface (mirrors llm-eval-kit, no dependency on it).

    provider.complete(prompt, system=None, max_tokens=512, temperature=0.0) -> Completion

Groq is OpenAI-compatible; Bedrock uses the Converse API through a lazily imported
boto3 so the API image does not need AWS unless NARRATIVE_PROVIDER=bedrock.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class ProviderError(RuntimeError):
    """Any failure to get text back: HTTP error, timeout, empty answer, missing SDK."""


@dataclass
class Completion:
    text: str
    model: str
    provider: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cached: bool = False


class GroqProvider:
    name = "groq"

    def __init__(self, api_key: str | None = None, model: str | None = None, timeout: float = 15.0):
        self.api_key = api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def payload(self, prompt: str, system: str | None, max_tokens: int, temperature: float) -> dict:
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        body = {"model": self.model, "messages": messages,
                "max_completion_tokens": max_tokens, "temperature": temperature}
        # gpt-oss reasons before answering and the reasoning counts toward the token budget
        if "gpt-oss" in self.model:
            body["reasoning_effort"] = "low"
        return body

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 512,
                 temperature: float = 0.0) -> Completion:
        if not self.available:
            raise ProviderError("GROQ_API_KEY not set")
        start = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(GROQ_URL, headers={"Authorization": f"Bearer {self.api_key}"},
                                   json=self.payload(prompt, system, max_tokens, temperature))
        except httpx.HTTPError as e:
            raise ProviderError(f"groq request failed ({type(e).__name__})") from e
        if resp.status_code != 200:
            raise ProviderError(f"groq returned {resp.status_code}")
        try:
            data = resp.json()
            text = (data["choices"][0]["message"].get("content") or "").strip()
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            raise ProviderError(f"groq returned a malformed body ({type(e).__name__})") from e
        if not text:
            raise ProviderError("groq returned empty content")
        usage = data.get("usage") or {}
        return Completion(text=text, model=self.model, provider=self.name,
                          input_tokens=int(usage.get("prompt_tokens", 0)),
                          output_tokens=int(usage.get("completion_tokens", 0)),
                          latency_ms=round((time.perf_counter() - start) * 1000, 1))


def bedrock_client(service: str = "bedrock-runtime"):
    try:
        import boto3  # lazy: only needed for Bedrock and Guardrails
    except ImportError as e:
        raise ProviderError("boto3 not installed") from e
    return boto3.client(service, region_name=os.getenv("AWS_REGION", os.getenv("BEDROCK_REGION", "us-east-1")))


class BedrockProvider:
    name = "bedrock"

    def __init__(self, model: str | None = None, client=None):
        self.model = model or os.getenv("BEDROCK_MODEL_ID", "")
        self._client = client

    @property
    def available(self) -> bool:
        return bool(self.model)

    def client(self):
        if self._client is None:
            self._client = bedrock_client()
        return self._client

    def complete(self, prompt: str, *, system: str | None = None, max_tokens: int = 512,
                 temperature: float = 0.0) -> Completion:
        if not self.available:
            raise ProviderError("BEDROCK_MODEL_ID not set")
        kwargs = {"modelId": self.model,
                  "messages": [{"role": "user", "content": [{"text": prompt}]}],
                  "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature}}
        if system:
            kwargs["system"] = [{"text": system}]
        start = time.perf_counter()
        try:
            resp = self.client().converse(**kwargs)
        except ProviderError:
            raise
        except Exception as e:  # botocore ClientError, credentials, throttling, network
            raise ProviderError(f"bedrock converse failed ({type(e).__name__})") from e
        blocks = resp.get("output", {}).get("message", {}).get("content", [])
        # gpt-oss on Bedrock also returns reasoningContent blocks; keep only the answer text
        text = "".join(b.get("text", "") for b in blocks if "text" in b).strip()
        if not text:
            raise ProviderError("bedrock returned empty content")
        usage = resp.get("usage") or {}
        return Completion(text=text, model=self.model, provider=self.name,
                          input_tokens=int(usage.get("inputTokens", 0)),
                          output_tokens=int(usage.get("outputTokens", 0)),
                          latency_ms=round((time.perf_counter() - start) * 1000, 1))


def build_provider(name: str):
    if name == "groq":
        return GroqProvider()
    if name == "bedrock":
        return BedrockProvider()
    raise ValueError(f"unknown narrative provider {name!r}")

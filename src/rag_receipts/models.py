"""Model clients: local Ollama by default, or Anthropic / OpenAI-compatible with your own key."""

from __future__ import annotations

import os
import time
from typing import Any, Protocol

import httpx
import numpy as np
from numpy.typing import NDArray


class ChatModel(Protocol):
    name: str

    def complete(self, system: str, user: str, json_output: bool = False) -> str: ...


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str], is_query: bool = False) -> NDArray[np.float32]: ...


class OllamaChat:
    def __init__(self, model: str = "qwen2.5:3b", host: str | None = None) -> None:
        self.name = f"ollama/{model}"
        self._model = model
        self._host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def complete(self, system: str, user: str, json_output: bool = False) -> str:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            # Deterministic output keeps evaluation runs comparable; the cap stops a small
            # model that starts repeating itself inside a JSON string.
            "options": {"temperature": 0, "seed": 7, "num_ctx": 8192, "num_predict": 1024},
        }
        if json_output:
            body["format"] = "json"
        response = httpx.post(f"{self._host}/api/chat", json=body, timeout=300)
        if response.status_code == 500 and "repeat limit" in response.text:
            # A small model can loop on repetitive sources (e.g. `# ///` blocks). One retry
            # with a repeat penalty; if it still loops, the model gave no usable answer.
            body["options"]["repeat_penalty"] = 1.2
            response = httpx.post(f"{self._host}/api/chat", json=body, timeout=300)
            if response.status_code == 500 and "repeat limit" in response.text:
                return ""
        if response.status_code >= 500:
            response = _post_with_retry(f"{self._host}/api/chat", body)
        response.raise_for_status()
        return str(response.json()["message"]["content"])


class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", host: str | None = None) -> None:
        self.name = f"ollama/{model}"
        self._model = model
        self._host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    def embed(self, texts: list[str], is_query: bool = False) -> NDArray[np.float32]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 32):
            response = _post_with_retry(
                f"{self._host}/api/embed",
                {"model": self._model, "input": texts[start : start + 32]},
            )
            vectors.extend(response.json()["embeddings"])
        return _normalise(np.asarray(vectors, dtype=np.float32))


# Qwen3-Embedding expects an instruction on queries and nothing on documents.
QWEN3_QUERY_TEMPLATE = (
    "Instruct: Given a question, retrieve passages from technical documents that answer it\n"
    "Query: {text}"
)


class OpenAICompatibleEmbedder:
    """Any `/v1/embeddings` server: OpenAI, vLLM, LM Studio, or llama.cpp's `llama-server`.

    `query_template` wraps questions (not documents) for models trained with an instruction,
    such as Qwen3-Embedding; use `{text}` where the question goes.
    """

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        query_template: str | None = None,
    ) -> None:
        self.name = f"openai/{model}"
        self._model = model
        self._base = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._query_template = query_template

    def embed(self, texts: list[str], is_query: bool = False) -> NDArray[np.float32]:
        if is_query and self._query_template:
            texts = [self._query_template.format(text=t) for t in texts]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 32):
            response = _post_with_retry(
                f"{self._base}/embeddings",
                {"model": self._model, "input": texts[start : start + 32]},
                headers={"Authorization": f"Bearer {self._key}"} if self._key else None,
            )
            rows = sorted(response.json()["data"], key=lambda row: row["index"])
            vectors.extend(row["embedding"] for row in rows)
        return _normalise(np.asarray(vectors, dtype=np.float32))


def embedder(spec: str, base_url: str | None = None, query_template: str | None = None) -> Embedder:
    """Build an embedder from `provider/model`; a bare model name means Ollama."""
    provider, _, model = spec.partition("/")
    if not model:
        return OllamaEmbedder(provider)
    if provider == "ollama":
        return OllamaEmbedder(model)
    if provider == "openai":
        if query_template is None and "qwen3" in model.lower():
            query_template = QWEN3_QUERY_TEMPLATE
        return OpenAICompatibleEmbedder(model, base_url=base_url, query_template=query_template)
    raise ValueError(f"unknown embedding provider {provider!r}; use ollama/ or openai/")


class AnthropicChat:
    def __init__(self, model: str = "claude-sonnet-5", api_key: str | None = None) -> None:
        self.name = f"anthropic/{model}"
        self._model = model
        self._key = api_key or os.environ["ANTHROPIC_API_KEY"]

    def complete(self, system: str, user: str, json_output: bool = False) -> str:
        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": self._key, "anthropic-version": "2023-06-01"},
            json={
                "model": self._model,
                "max_tokens": 2048,
                "temperature": 0,
                "system": system,
                "messages": [{"role": "user", "content": user}],
            },
            timeout=300,
        )
        response.raise_for_status()
        return "".join(b["text"] for b in response.json()["content"] if b["type"] == "text")


class OpenAICompatibleChat:
    """Any server that speaks the OpenAI chat completions API (OpenAI, vLLM, LM Studio, ...)."""

    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None) -> None:
        self.name = f"openai/{model}"
        self._model = model
        self._base = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self._key = api_key or os.environ.get("OPENAI_API_KEY", "")

    def complete(self, system: str, user: str, json_output: bool = False) -> str:
        body: dict[str, Any] = {
            "model": self._model,
            "temperature": 0,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        if json_output:
            body["response_format"] = {"type": "json_object"}
        response = httpx.post(
            f"{self._base}/chat/completions",
            headers={"Authorization": f"Bearer {self._key}"},
            json=body,
            timeout=300,
        )
        response.raise_for_status()
        return str(response.json()["choices"][0]["message"]["content"])


def chat_model(spec: str) -> ChatModel:
    """Build a chat model from `provider/model`, e.g. `ollama/qwen2.5:3b`."""
    provider, _, model = spec.partition("/")
    if provider == "ollama":
        return OllamaChat(model or "qwen2.5:3b")
    if provider == "anthropic":
        return AnthropicChat(model or "claude-sonnet-5")
    if provider == "openai":
        return OpenAICompatibleChat(model or "gpt-4o-mini")
    raise ValueError(f"unknown provider {provider!r}; use ollama/, anthropic/ or openai/")


def _post_with_retry(
    url: str, body: dict[str, Any], attempts: int = 3, headers: dict[str, str] | None = None
) -> httpx.Response:
    """POST to a model server, retrying 5xx and connection errors with back-off."""
    for attempt in range(attempts):
        try:
            response = httpx.post(url, json=body, headers=headers, timeout=300)
            if response.status_code < 500:
                response.raise_for_status()
                return response
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
        time.sleep(2**attempt)
    response.raise_for_status()
    return response


def _normalise(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.where(norms == 0, 1, norms)).astype(np.float32)

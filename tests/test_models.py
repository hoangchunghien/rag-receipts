from typing import Any

import httpx
import pytest

from rag_receipts import models
from rag_receipts.models import QWEN3_QUERY_TEMPLATE, embedder


def test_qwen3_wraps_queries_but_not_documents(monkeypatch: pytest.MonkeyPatch) -> None:
    sent: list[list[str]] = []

    def fake_post(url: str, body: dict[str, Any], **_: Any) -> httpx.Response:
        sent.append(body["input"])
        data = [{"index": i, "embedding": [1.0, 0.0]} for i in range(len(body["input"]))]
        return httpx.Response(200, json={"data": data}, request=httpx.Request("POST", url))

    monkeypatch.setattr(models, "_post_with_retry", fake_post)
    qwen = embedder("openai/qwen3-embedding-8b", base_url="http://localhost:8081/v1")
    qwen.embed(["a passage"])
    qwen.embed(["a question?"], is_query=True)
    assert sent == [["a passage"], [QWEN3_QUERY_TEMPLATE.format(text="a question?")]]


def test_a_bare_model_name_means_ollama() -> None:
    assert embedder("nomic-embed-text").name == "ollama/nomic-embed-text"

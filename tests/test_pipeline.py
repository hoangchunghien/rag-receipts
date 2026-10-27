"""End to end with a fake model and a bag-of-words embedder, so the tests need no GPU."""

import json
import re
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
from numpy.typing import NDArray

from rag_receipts.answer import NOT_FOUND, ask_checked, ask_plain
from rag_receipts.documents import Document, chunk_document, load_documents
from rag_receipts.index import Index
from rag_receipts.render import to_html, to_terminal

DOC = """Title: Strict zip

Abstract
========

This PEP proposes adding an optional ``strict`` boolean keyword
parameter to the built-in ``zip``.

When enabled, a ``ValueError`` is raised if one of the arguments
is exhausted before the others.
"""


class WordEmbedder:
    name = "test/words"
    vocabulary: ClassVar[list[str]] = ["zip", "strict", "valueerror", "exhausted", "django"]

    def embed(self, texts: list[str]) -> NDArray[np.float32]:
        rows = [[t.lower().count(w) + 0.01 for w in self.vocabulary] for t in texts]
        vectors = np.asarray(rows, dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


class ScriptedModel:
    """Returns queued replies in order and records what it was asked."""

    name = "test/scripted"

    def __init__(self, *replies: str) -> None:
        self.replies = list(replies)
        self.prompts: list[str] = []

    def complete(self, system: str, user: str, json_output: bool = False) -> str:
        self.prompts.append(user)
        return self.replies.pop(0)


@pytest.fixture
def index(tmp_path: Path) -> Index:
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "strict-zip.md").write_text(DOC)
    built = Index(tmp_path / "index.db", WordEmbedder())
    built.add(load_documents([tmp_path / "docs"]))
    return built


def claims(*items: tuple[str, str, str]) -> str:
    rows = [{"sentence": s, "source": src, "quote": q} for s, src, q in items]
    return json.dumps({"claims": rows})


def test_chunks_keep_offsets_into_the_original_text() -> None:
    doc = Document("d", "t", "p", DOC)
    for chunk in chunk_document(doc, max_chars=120):
        assert DOC[chunk.start : chunk.end] == chunk.text


def test_a_verified_claim_is_answered_with_its_receipt(index: Index) -> None:
    model = ScriptedModel(
        claims(("zip() gets a strict flag.", "S1", "adding an optional strict boolean keyword"))
    )
    answer = ask_checked("What does strict do in zip?", index, model)
    assert answer.text == "zip() gets a strict flag."
    assert "[1] Strict zip" in to_terminal(answer)
    assert "<mark>" in to_html(answer)


def test_an_invented_quote_is_cut_and_the_repair_round_can_fix_it(index: Index) -> None:
    model = ScriptedModel(
        claims(("zip() gets a strict flag.", "S1", "zip now has a strict mode for safety")),
        claims(("zip() gets a strict flag.", "S1", "adding an optional strict boolean keyword")),
    )
    answer = ask_checked("What does strict do in zip?", index, model)
    assert answer.text == "zip() gets a strict flag."
    assert answer.repaired == 1
    assert "quote does not occur" in model.prompts[1]


def test_when_nothing_survives_the_answer_says_so(index: Index) -> None:
    model = ScriptedModel(
        claims(("Django uses middleware.", "S1", "Django middleware is configured in settings")),
        claims(),
    )
    answer = ask_checked("How do I configure Django middleware?", index, model)
    assert answer.refused
    assert answer.text == NOT_FOUND
    assert len(answer.cut) == 1


def test_unparseable_model_output_is_a_refusal_not_a_crash(index: Index) -> None:
    answer = ask_checked("strict zip?", index, ScriptedModel("not json at all", "{}"), repair=False)
    assert answer.refused


def test_plain_mode_is_the_baseline_without_checks(index: Index) -> None:
    answer = ask_plain("strict zip?", index, ScriptedModel("zip gets strict in 3.10 [S1]"))
    assert not answer.refused
    assert re.search(r"\[S1\]", answer.text)
    assert ask_plain("django?", index, ScriptedModel("NOT_FOUND")).refused

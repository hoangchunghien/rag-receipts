"""Ask a question against the index, in checked mode or as a plain-RAG baseline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .index import Hit, Index
from .models import ChatModel
from .verify import Claim, check_claim, fold, missing_names, unsupported_specifics

NOT_FOUND = "The documents I was given do not answer this."

CHECKED_SYSTEM = """You answer questions using ONLY the numbered sources you are given.

Reply with JSON of this shape:
{"claims": [{"sentence": "...", "source": "S1", "quote": "..."}]}

Rules:
- Each claim is one sentence of your answer.
- "source" is the id of the single source that supports the sentence.
- "quote" is copied word for word from that source: at least 6 consecutive words that
  support the sentence. Do not paraphrase the quote, do not join pieces of text.
- Do not use anything you know that is not in the sources.
- If the sources do not answer the question, reply {"claims": []}."""

PLAIN_SYSTEM = """You answer questions using ONLY the numbered sources you are given.
Write a short answer and cite sources inline like [S1].
If the sources do not answer the question, reply exactly: NOT_FOUND"""


@dataclass
class Answer:
    question: str
    mode: str
    model: str
    hits: list[Hit]
    text: str
    claims: list[Claim] = field(default_factory=list)
    refused: bool = False
    raw: str = ""
    first_pass: int = 0
    repaired: int = 0
    # Question names that no cited passage mentions (check 4); set means the answer was refused.
    off_topic: list[str] = field(default_factory=list)

    @property
    def kept(self) -> list[Claim]:
        return [] if self.off_topic else [c for c in self.claims if c.verified]

    @property
    def set_aside(self) -> list[Claim]:
        """Claims that passed checks 1 to 3 but answer a different question (check 4)."""
        return [c for c in self.claims if c.verified] if self.off_topic else []

    @property
    def cut(self) -> list[Claim]:
        return [c for c in self.claims if not c.verified]


def _sources_block(hits: list[Hit]) -> tuple[str, dict[str, str]]:
    sources = {f"S{i + 1}": hit.chunk.text for i, hit in enumerate(hits)}
    block = "\n\n".join(
        f"[{sid}] ({hits[i].title})\n{text}" for i, (sid, text) in enumerate(sources.items())
    )
    return block, sources


REPAIR_SYSTEM = """Some sentences of your answer were rejected because their quotes are not
copied word for word from the source they cite. For each rejected sentence, either give a
quote copied exactly from one of the sources (at least 6 consecutive words), or leave the
sentence out. Do not add new sentences.

Reply with JSON of this shape:
{"claims": [{"sentence": "...", "source": "S1", "quote": "..."}]}"""


def ask_checked(
    question: str, index: Index, model: ChatModel, k: int = 6, repair: bool = True
) -> Answer:
    """Answer with claims, check each one, and give rejected claims one chance to be fixed.

    The repair round tells the model which quotes were not found. It cannot loosen the
    checks: a repaired claim passes the same string match as any other.
    """
    hits = index.search(question, k=k)
    block, sources = _sources_block(hits)
    prompt = f"Sources:\n\n{block}\n\nQuestion: {question}"
    raw = model.complete(CHECKED_SYSTEM, prompt, True)
    claims = [check_claim(c, sources, question) for c in _parse_claims(raw)]
    first_pass = len(claims)
    rejected = [c for c in claims if not c.verified]
    fixed: list[Claim] = []
    if repair and rejected:
        listing = "\n".join(
            f"- {c.sentence} (cited {c.source}; {'; '.join(c.problems)})" for c in rejected
        )
        raw_repair = model.complete(
            REPAIR_SYSTEM, f"{prompt}\n\nRejected sentences:\n{listing}", True
        )
        repaired = [check_claim(c, sources, question) for c in _parse_claims(raw_repair)]
        for claim in repaired:
            claim.repaired = True
        fixed = [c for c in repaired if c.verified]
        fixed_sentences = {fold(c.sentence) for c in fixed}
        still_cut = [c for c in rejected if fold(c.sentence) not in fixed_sentences]
        claims = [c for c in claims if c.verified] + fixed + still_cut
        raw = f"{raw}\n--- repair ---\n{raw_repair}"
    kept = [c for c in claims if c.verified]
    off_topic = missing_names(question, [sources[c.source] for c in kept]) if kept else []
    if off_topic:
        kept = []
    text = " ".join(c.sentence.strip() for c in kept) if kept else NOT_FOUND
    return Answer(
        question,
        "checked",
        model.name,
        hits,
        text,
        claims,
        refused=not kept,
        raw=raw,
        first_pass=first_pass,
        repaired=len(fixed),
        off_topic=off_topic,
    )


def ask_plain(question: str, index: Index, model: ChatModel, k: int = 6) -> Answer:
    hits = index.search(question, k=k)
    block, _ = _sources_block(hits)
    raw = model.complete(PLAIN_SYSTEM, f"Sources:\n\n{block}\n\nQuestion: {question}")
    refused = "NOT_FOUND" in raw or not raw.strip()
    return Answer(
        question,
        "plain",
        model.name,
        hits,
        NOT_FOUND if refused else raw.strip(),
        refused=refused,
        raw=raw,
    )


def plain_unsupported(answer: Answer) -> list[str]:
    """Per sentence of a plain answer: numbers or code names its cited sources do not contain."""
    problems: list[str] = []
    sources = {f"S{i + 1}": hit.chunk.text for i, hit in enumerate(answer.hits)}
    text = re.sub(r"```.*?```", " ", answer.text, flags=re.DOTALL)
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        cited = [sources[s] for s in re.findall(r"\[(S\d+)\]", sentence) if s in sources]
        problems.extend(
            unsupported_specifics(sentence, cited or list(sources.values()), answer.question)
        )
    return problems


def _parse_claims(raw: str) -> list[Claim]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
    items = data.get("claims", []) if isinstance(data, dict) else []
    claims = []
    for item in items:
        if isinstance(item, dict) and isinstance(item.get("sentence"), str):
            claims.append(
                Claim(
                    sentence=item["sentence"],
                    source=str(item.get("source", "")).strip("[] "),
                    quote=str(item.get("quote", "")),
                )
            )
    return claims

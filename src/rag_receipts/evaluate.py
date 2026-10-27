"""Score checked answers against the plain-RAG baseline on one question set.

Each question is either answerable from the corpus (with the documents that answer it) or a
trap: plausible, on topic, and absent from the corpus, so a model that answers it is answering
from memory, not from the documents.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .answer import Answer, ask_checked, ask_plain, plain_unsupported
from .index import Index
from .models import ChatModel


def run_eval(questions_path: Path, index: Index, model: ChatModel, out: Path, k: int = 6) -> str:
    questions = [json.loads(line) for line in questions_path.read_text().splitlines() if line]
    rows: list[dict[str, Any]] = []
    with out.open("w") as sink:
        for q in questions:
            for _mode, ask in (("plain", ask_plain), ("checked", ask_checked)):
                answer = ask(q["question"], index, model, k=k)
                row = _score(q, answer)
                rows.append(row)
                sink.write(json.dumps(row) + "\n")
    return summarise(rows, model.name)


class ReplayModel:
    """Plays back a previous run's model outputs, so changed checks can be re-scored for free.

    Outputs are replayed in the order the pipeline asks for them. If a changed check now
    rejects a claim that passed before, the pipeline asks for a repair the old run never
    made; the replay answers with no claims, which is the conservative reading.
    """

    def __init__(self, name: str, outputs: list[str]) -> None:
        self.name = name
        self._outputs = outputs

    def complete(self, system: str, user: str, json_output: bool = False) -> str:
        return self._outputs.pop(0) if self._outputs else '{"claims": []}'


def rescore(
    questions_path: Path, index: Index, previous: Path, out: Path, k: int = 6, model: str = ""
) -> str:
    """Re-run the checks on the model outputs saved in `previous`, without calling a model."""
    questions = {
        q["id"]: q
        for q in (json.loads(line) for line in questions_path.read_text().splitlines() if line)
    }
    old_rows = [json.loads(line) for line in previous.read_text().splitlines() if line]
    model_name = (old_rows[0].get("model") if old_rows else None) or model or "replay"
    rows: list[dict[str, Any]] = []
    with out.open("w") as sink:
        for old in old_rows:
            q = questions[old["id"]]
            replay = ReplayModel(model_name, old["raw"].split("\n--- repair ---\n"))
            ask = ask_checked if old["mode"] == "checked" else ask_plain
            row = _score(q, ask(q["question"], index, replay, k=k))
            rows.append(row)
            sink.write(json.dumps(row) + "\n")
    return summarise(rows, model_name)


def _score(q: dict[str, Any], answer: Answer) -> dict[str, Any]:
    doc_of = {f"S{i + 1}": hit.chunk.doc_id for i, hit in enumerate(answer.hits)}
    if answer.mode == "checked":
        cited_docs = sorted({doc_of[c.source] for c in answer.kept})
        problems = [p for c in answer.cut for p in c.problems]
        unsupported: list[str] = []
    else:
        cited = re.findall(r"\[(S\d+)\]", answer.text)
        cited_docs = sorted({doc_of[s] for s in cited if s in doc_of})
        problems = []
        unsupported = [] if answer.refused else plain_unsupported(answer)
    return {
        "id": q["id"],
        "answerable": q["answerable"],
        "mode": answer.mode,
        "model": answer.model,
        "answered": not answer.refused,
        "cited_expected": bool(set(cited_docs) & set(q.get("expect", []))),
        "cited_docs": cited_docs,
        "first_pass": answer.first_pass,
        "repaired": answer.repaired,
        "reattributed": sum(c.reattributed_from is not None for c in answer.kept),
        "kept": len(answer.kept),
        "off_topic": answer.off_topic,
        "problems": problems,
        "unsupported_specifics": unsupported,
        "text": answer.text,
        "raw": answer.raw,
    }


def summarise(rows: list[dict[str, Any]], model_name: str) -> str:
    def pct(n: int, d: int) -> str:
        return f"{100 * n / d:.0f}% ({n}/{d})" if d else "n/a"

    lines = [f"Model: {model_name}", "", "| | plain RAG | checked |", "|---|---|---|"]
    by_mode = {m: [r for r in rows if r["mode"] == m] for m in ("plain", "checked")}
    answerable = {m: [r for r in rs if r["answerable"]] for m, rs in by_mode.items()}
    traps = {m: [r for r in rs if not r["answerable"]] for m, rs in by_mode.items()}

    def row(label: str, plain: str, checked: str) -> None:
        lines.append(f"| {label} | {plain} | {checked} |")

    def rate(m: str, rows_: list[dict[str, Any]], test: Any) -> str:
        return pct(sum(bool(test(r)) for r in rows_), len(rows_))

    row("Answerable: answered", *(rate(m, answerable[m], lambda r: r["answered"]) for m in by_mode))
    row(
        "Answerable: answered with a citation you can follow",
        *(rate(m, answerable[m], lambda r: r["answered"] and r["cited_docs"]) for m in by_mode),
    )
    row(
        "Answerable: ... and it points at the right PEP",
        *(rate(m, answerable[m], lambda r: r["answered"] and r["cited_expected"]) for m in by_mode),
    )
    row(
        "Traps: answered anyway (lower is better)",
        *(rate(m, traps[m], lambda r: r["answered"]) for m in by_mode),
    )
    plain_answers = [r for r in by_mode["plain"] if r["answered"]]
    row(
        "Answers with a number or code name not in their sources",
        rate("plain", plain_answers, lambda r: r["unsupported_specifics"]),
        "0 (check 3 cuts them)",
    )

    checked = by_mode["checked"]
    first = sum(r["first_pass"] for r in checked)
    first_kept = sum(r["kept"] - r["repaired"] for r in checked)
    repaired = sum(r["repaired"] for r in checked)
    moved = sum(r["reattributed"] for r in checked)
    reasons = Counter(
        re.sub(r"number \S+ is not|`[^`]*` is not", "a specific is not", p)
        for r in checked
        for p in r["problems"]
    )
    lines += [
        "",
        f"Checked mode: the model proposed {first} claims; {first_kept} passed first time, "
        f"{repaired} more passed after one repair round, {moved} kept claims had their "
        "citation corrected to the source that really holds the quote.",
        f"Check 4 set aside {sum(bool(r.get('off_topic')) for r in checked if r['answerable'])} "
        f"answerable and {sum(bool(r.get('off_topic')) for r in checked if not r['answerable'])} "
        "trap answers whose cited passages never mention the names the question asks about.",
        "Still cut after repair:",
    ]
    for reason, n in reasons.most_common():
        lines.append(f"- {reason}: {n}")
    return "\n".join(lines)

"""receipts: index documents, ask questions with checked citations, and measure it."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .answer import ask_checked, ask_plain
from .documents import load_documents
from .index import Index
from .models import OllamaEmbedder, chat_model
from .render import to_html, to_terminal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="receipts", description=__doc__)
    parser.add_argument("--index", default="receipts.db", help="index file (default receipts.db)")
    parser.add_argument("--embedder", default="nomic-embed-text", help="Ollama embedding model")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="add files or folders to the index")
    ingest.add_argument("paths", nargs="+", type=Path)

    ask = sub.add_parser("ask", help="answer a question with receipts")
    ask.add_argument("question")
    ask.add_argument("--model", default="ollama/qwen2.5:3b", help="provider/model")
    ask.add_argument("--plain", action="store_true", help="plain RAG, no checks (the baseline)")
    ask.add_argument("--html", type=Path, help="also write an HTML receipt to this file")
    ask.add_argument("-k", type=int, default=6, help="sources to retrieve")

    evaluate = sub.add_parser("eval", help="score checked vs plain answers on a question set")
    evaluate.add_argument("questions", type=Path)
    evaluate.add_argument("--model", default="ollama/qwen2.5:3b")
    evaluate.add_argument("--out", type=Path, default=Path("eval-results.jsonl"))
    evaluate.add_argument("-k", type=int, default=6)

    args = parser.parse_args(argv)
    index = Index(args.index, OllamaEmbedder(args.embedder))

    if args.command == "ingest":
        added = index.add(load_documents(args.paths))
        docs, chunks = index.stats()
        print(f"added {added} chunks; index now has {docs} documents, {chunks} chunks")
        return 0

    if args.command == "ask":
        model = chat_model(args.model)
        answer = (ask_plain if args.plain else ask_checked)(args.question, index, model, k=args.k)
        print(to_terminal(answer))
        if args.html:
            args.html.write_text(to_html(answer), encoding="utf-8")
            print(f"\nreceipt written to {args.html}", file=sys.stderr)
        return 0

    from .evaluate import run_eval

    report = run_eval(args.questions, index, chat_model(args.model), args.out, k=args.k)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

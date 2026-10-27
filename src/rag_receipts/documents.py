"""Load source documents and split them into chunks that remember where they came from."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = {".md", ".markdown", ".rst", ".txt"}


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    path: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """A slice of one document. `start` and `end` are character offsets into `Document.text`."""

    chunk_id: str
    doc_id: str
    start: int
    end: int
    text: str


def load_documents(paths: Iterable[Path]) -> Iterator[Document]:
    """Read every supported file under the given files or directories, in a stable order."""
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file() and _supported(p))
        elif _supported(path):
            files.append(path)
    for file in sorted(files):
        text = _read(file)
        if text.strip():
            yield Document(doc_id=file.stem, title=_title(file, text), path=str(file), text=text)


def _supported(path: Path) -> bool:
    return path.suffix.lower() in TEXT_SUFFIXES or path.suffix.lower() == ".pdf"


def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as error:  # pragma: no cover - depends on the optional extra
            message = "PDF support needs the extra: pip install 'rag-receipts[pdf]'"
            raise SystemExit(message) from error
        return "\n\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    return path.read_text(encoding="utf-8", errors="replace")


def _title(path: Path, text: str) -> str:
    # PEPs and many docs carry a "Title:" header; Markdown uses "# ".
    for line in text.splitlines()[:40]:
        if line.startswith("Title:"):
            return line.removeprefix("Title:").strip()
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    return path.stem


_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def chunk_document(
    doc: Document, max_chars: int = 1200, overlap_paragraphs: int = 1
) -> list[Chunk]:
    """Pack whole paragraphs into chunks of at most `max_chars`.

    Paragraph boundaries keep quotes intact: a sentence the model quotes is far more likely
    to sit inside one chunk than across two. The last paragraph of a chunk is repeated at
    the start of the next one, so an answer that needs both sides of a boundary can still
    quote from a single chunk.
    """
    spans = _paragraph_spans(doc.text)
    chunks: list[Chunk] = []
    i = 0
    while i < len(spans):
        start, end = spans[i]
        j = i + 1
        while j < len(spans) and spans[j][1] - start <= max_chars:
            end = spans[j][1]
            j += 1
        text = doc.text[start:end]
        chunk_id = f"{doc.doc_id}#{len(chunks)}"
        chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc.doc_id, start=start, end=end, text=text))
        i = max(j - overlap_paragraphs, i + 1) if j < len(spans) else j
    return chunks


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = 0
    for match in _PARAGRAPH_BREAK.finditer(text):
        if text[position : match.start()].strip():
            spans.append((position, match.start()))
        position = match.end()
    if text[position:].strip():
        spans.append((position, len(text)))
    return spans

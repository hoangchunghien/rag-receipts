"""A single-file SQLite index: full-text search for exact terms, embeddings for meaning."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .documents import Chunk, Document, chunk_document
from .models import Embedder

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (doc_id TEXT PRIMARY KEY, title TEXT, path TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY, doc_id TEXT, start INTEGER, end INTEGER, text TEXT, embedding BLOB
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, text);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    title: str
    score: float


class Index:
    def __init__(self, path: Path | str, embedder: Embedder) -> None:
        self._db = sqlite3.connect(str(path))
        self._db.executescript(SCHEMA)
        self._embedder = embedder
        self._matrix: np.ndarray | None = None
        self._ids: list[str] = []

    def add(self, documents: Iterable[Document], max_chars: int = 1200) -> int:
        """Index documents, replacing any earlier version with the same id. Returns chunks added."""
        stored = self._db.execute("SELECT value FROM meta WHERE key = 'embedder'").fetchone()
        if stored and stored[0] != self._embedder.name:
            raise ValueError(
                f"index was built with {stored[0]}, not {self._embedder.name}; use a new index file"
            )
        count = 0
        for doc in documents:
            chunks = chunk_document(doc, max_chars=max_chars)
            vectors = self._embedder.embed([f"{doc.title}\n\n{c.text}" for c in chunks])
            with self._db:
                self._db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc.doc_id,))
                self._db.execute(
                    "DELETE FROM chunks_fts WHERE chunk_id LIKE ?", (f"{doc.doc_id}#%",)
                )
                self._db.execute(
                    "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?)",
                    (doc.doc_id, doc.title, doc.path, doc.text),
                )
                for chunk, vector in zip(chunks, vectors, strict=True):
                    self._db.execute(
                        "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            chunk.chunk_id,
                            chunk.doc_id,
                            chunk.start,
                            chunk.end,
                            chunk.text,
                            vector.astype(np.float32).tobytes(),
                        ),
                    )
                    self._db.execute(
                        "INSERT INTO chunks_fts VALUES (?, ?)", (chunk.chunk_id, chunk.text)
                    )
                self._db.execute(
                    "INSERT OR REPLACE INTO meta VALUES ('embedder', ?)", (self._embedder.name,)
                )
            count += len(chunks)
        self._matrix = None
        return count

    def search(self, query: str, k: int = 6, candidates: int = 30) -> list[Hit]:
        """Hybrid search: keyword and vector rankings fused with reciprocal rank fusion.

        Keywords find exact names (`TypeVarTuple`, `PEP 604`) that embeddings blur; embeddings
        find paraphrases that keywords miss. RRF needs no score calibration between the two.
        """
        keyword = self._keyword_ranking(query, candidates)
        vector = self._vector_ranking(query, candidates)
        fused: dict[str, float] = {}
        for ranking in (keyword, vector):
            for rank, chunk_id in enumerate(ranking):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (60 + rank)
        best = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:k]
        return [self._hit(cid, fused[cid]) for cid in best]

    def document(self, doc_id: str) -> Document | None:
        row = self._db.execute(
            "SELECT doc_id, title, path, text FROM documents WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        return Document(*row) if row else None

    def chunk(self, chunk_id: str) -> Chunk | None:
        row = self._db.execute(
            "SELECT chunk_id, doc_id, start, end, text FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return Chunk(*row) if row else None

    def stats(self) -> tuple[int, int]:
        docs = self._db.execute("SELECT count(*) FROM documents").fetchone()[0]
        chunks = self._db.execute("SELECT count(*) FROM chunks").fetchone()[0]
        return int(docs), int(chunks)

    def _keyword_ranking(self, query: str, limit: int) -> list[str]:
        terms = re.findall(r"\w+", query)
        if not terms:
            return []
        # Quote every term so FTS5 syntax in the question (AND, NEAR, -) is read as text.
        match = " OR ".join(f'"{t}"' for t in terms)
        rows = self._db.execute(
            "SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH ? "
            "ORDER BY bm25(chunks_fts) LIMIT ?",
            (match, limit),
        ).fetchall()
        return [r[0] for r in rows]

    def _vector_ranking(self, query: str, limit: int) -> list[str]:
        if self._matrix is None:
            rows = self._db.execute("SELECT chunk_id, embedding FROM chunks").fetchall()
            if not rows:
                return []
            self._ids = [r[0] for r in rows]
            self._matrix = np.vstack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        query_vector = self._embedder.embed([query])[0]
        scores = self._matrix @ query_vector
        order = np.argsort(-scores)[:limit]
        return [self._ids[i] for i in order]

    def _hit(self, chunk_id: str, score: float) -> Hit:
        row = self._db.execute(
            "SELECT c.chunk_id, c.doc_id, c.start, c.end, c.text, d.title "
            "FROM chunks c JOIN documents d USING (doc_id) WHERE c.chunk_id = ?",
            (chunk_id,),
        ).fetchone()
        return Hit(chunk=Chunk(*row[:5]), title=row[5], score=score)

"""Optional disk-backed lexical index with the same tokenizer and scoped API.

Each process owns its database; durable wiki pages remain the source of truth.
FTS5 uses its native multi-column BM25, so scores need not equal the Python engine.
"""
from __future__ import annotations

import sqlite3
import tempfile
import threading
from collections import Counter
from pathlib import Path

from app.core.exceptions import RetrievalError, ConfigurationError
from app.retrieval.bm25 import BM25Config
from app.retrieval.tokenizer import tokenize


class SQLiteBM25Index:
    def __init__(self, config: BM25Config):
        if config.variant != "bm25":
            raise ConfigurationError("SQLite lexical backend supports native BM25, not bm25x")
        if config.k1 != BM25Config().k1 or config.b != BM25Config().b:
            raise ConfigurationError("SQLite uses FTS5 native BM25; custom k1/b require the memory backend")
        self.config = config
        self._directory = tempfile.TemporaryDirectory(prefix="nano-rag-lexical-")
        self._db = sqlite3.connect(Path(self._directory.name) / "index.sqlite3", check_same_thread=False)
        self._lock = threading.RLock()
        self._db.execute("CREATE VIRTUAL TABLE pages USING fts5(id UNINDEXED, body, title, context, aliases, questions)")
        self._db.execute("CREATE VIRTUAL TABLE vocab USING fts5vocab(pages, 'row')")

    def _tokenize(self, text):
        return tokenize(text, self.config.tokenizer)

    @property
    def document_count(self):
        with self._lock:
            return self._db.execute("SELECT count(*) FROM pages").fetchone()[0]

    def add_document(self, doc_id, text, title="", fields=None):
        fields = fields or {}
        with self._lock, self._db:
            exists = self._db.execute("SELECT rowid FROM pages WHERE id = ?", (doc_id,)).fetchone()
            if not exists and self.document_count >= self.config.max_documents:
                raise RetrievalError("SQLite lexical index capacity reached")
            self._db.execute("DELETE FROM pages WHERE id = ?", (doc_id,))
            values = [" ".join(self._tokenize(v)) for v in (
                text, title, fields.get("context", ""), fields.get("aliases", ""), fields.get("questions", ""))]
            self._db.execute("INSERT INTO pages(id,body,title,context,aliases,questions) VALUES (?,?,?,?,?,?)",
                             [doc_id, *values])

    def remove_document(self, doc_id):
        with self._lock, self._db:
            self._db.execute("DELETE FROM pages WHERE id = ?", (doc_id,))

    def search(self, query, top_k, allowed_doc_ids=None):
        return self.search_weighted(query, top_k, allowed_doc_ids=allowed_doc_ids)

    def search_weighted(self, query, top_k, allowed_doc_ids=None, *, term_weights=None):
        weights = dict(Counter(self._tokenize(query))) if term_weights else dict.fromkeys(self._tokenize(query), 1.0)
        for term, weight in (term_weights or {}).items():
            if weight > 0:
                weights[term] = weights.get(term, 0) + weight
        scores = {}
        with self._lock:
            field_weights = self.config.field_weights
            for term, weight in weights.items():
                literal = '"' + term.replace('"', '""') + '"'
                rows = self._db.execute("SELECT id,bm25(pages,0,1,?,?,?,?) FROM pages WHERE pages MATCH ?",
                    (self.config.title_weight, field_weights.get("context", 0),
                     field_weights.get("aliases", 0), field_weights.get("questions", 0), literal))
                for key, score in rows:
                    if score < 0 and (allowed_doc_ids is None or key in allowed_doc_ids):
                        scores[key] = scores.get(key, 0) - score * weight
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]

    def document_frequencies(self):
        with self._lock:
            return dict(self._db.execute("SELECT term,doc FROM vocab")), self.document_count

    def clear(self):
        with self._lock, self._db:
            self._db.execute("DELETE FROM pages")

    def close(self):
        self._db.close()
        self._directory.cleanup()

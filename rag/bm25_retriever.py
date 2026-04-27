"""
LexShield AI — BM25 Keyword Retriever
======================================
Keyword-based retrieval using BM25Okapi from rank_bm25.
Complements vector search in the hybrid pipeline.

Key design choices:
  • Indexes context_text (header-prefixed) for richer keyword matching
  • Legal-aware tokenizer — preserves section numbers, strips legal boilerplate stopwords
  • Score normalised to [0, 1] for fusion with vector scores
  • Index built from chunks.json at startup; call rebuild() after re-ingestion
  • Memory safe on 8 GB RAM for up to ~20,000 chunks
"""

import os
import re
import json
import gc
from pathlib import Path
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("Run: pip install rank-bm25")

import numpy as np

# ── Legal stopwords ───────────────────────────────────────────────────────────
# Extends English defaults with Indian-legal boilerplate that adds noise to BM25.
LEGAL_STOPWORDS: frozenset[str] = frozenset({
    # English function words
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "this", "that", "these", "those", "it", "its", "he", "she", "they",
    "we", "said", "such", "any", "all", "which", "who", "whom",
    "under", "into", "upon", "after", "before", "during", "within",
    # Indian-legal boilerplate
    "thereof", "therein", "thereto", "thereby", "herein", "hereof",
    "hereby", "whereas", "notwithstanding", "pursuant", "aforesaid",
    "abovementioned", "hereunder", "thereunder", "howsoever",
    # Context prefix tokens (from our contextual chunker)
    "chapter", "general", "provisions", "supreme", "court", "judgment","ministry",
})


def tokenize(text: str) -> list[str]:
    """
    Legal-aware BM25 tokenizer.

    Preserves:
      - Section numbers ("420", "21A")
      - Important abbreviations ("IPC", "BNS", "CrPC", "PIL")
      - Hyphenated legal terms ("non-bailable")

    Removes: stopwords, tokens < 2 chars (unless digit), punctuation.
    """
    text = text.lower()
    # Normalise punctuation — keep hyphens inside words, kill the rest
    text = re.sub(r'(?<!\w)-(?!\w)', ' ', text)   # standalone hyphens → space
    text = re.sub(r'[^\w\s\-]', ' ', text)
    tokens = text.split()
    return [
        t for t in tokens
        if t not in LEGAL_STOPWORDS
        and (len(t) > 1 or t.isdigit())
        and not t.startswith('_')
    ]


# ── BM25 retriever ────────────────────────────────────────────────────────────

class BM25Retriever:
    """
    BM25Okapi index over the LexShield legal corpus.

    Usage:
        from rag.bm25_retriever import bm25_retriever
        results = bm25_retriever.search("Section 420 cheating", n_results=8)
    """

    def __init__(self, chunks_path: str = "data/processed/chunks.json"):
        self.chunks_path = chunks_path
        self.chunks:   list[dict]         = []
        self.bm25:     Optional[BM25Okapi] = None
        self._ready    = False
        self._build_index()

    # ── Index construction ────────────────────────────────────────────────────

    def _build_index(self) -> None:
        path = Path(self.chunks_path)
        if not path.exists():
            raise FileNotFoundError(
                f"[BM25] chunks.json not found at {self.chunks_path}\n"
                "Run: python -m data.preprocessor   first."
            )

        print(f"[BM25] Loading {self.chunks_path} ...")
        with open(path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)
        print(f"[BM25] {len(self.chunks)} chunks loaded.")

        # Index context_text if available (richer), else plain text
        corpus_texts = [
            c.get("context_text") or c.get("text", "")
            for c in self.chunks
        ]

        print("[BM25] Tokenising corpus ...")
        tokenized = [tokenize(t) for t in corpus_texts]

        print("[BM25] Fitting BM25Okapi ...")
        self.bm25   = BM25Okapi(tokenized)
        self._ready = True
        gc.collect()
        print(f"[BM25] Index ready ({len(self.chunks)} docs).")

    def rebuild(self) -> None:
        """Reload chunks.json and rebuild index. Call after re-ingestion."""
        self.chunks  = []
        self.bm25    = None
        self._ready  = False
        self._build_index()

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query:     str,
        n_results: int   = 8,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        BM25 keyword search over the corpus.

        Returns list of result dicts (sorted descending by bm25_score_norm):
          chunk_id, text, context_text, source, doc_type,
          section, section_title, chapter, chunk_type,
          bm25_score        — raw BM25 score
          bm25_score_norm   — normalised to [0, 1] (max score = 1.0)
        """
        if not self._ready:
            raise RuntimeError("[BM25] Index not ready. Call _build_index() first.")

        tokens = tokenize(query)
        if not tokens:
            return []   # nothing to search

        raw_scores: np.ndarray = self.bm25.get_scores(tokens)  # shape: (n_chunks,)

        # Normalise to [0, 1] against this query's max score
        max_score = float(raw_scores.max()) if raw_scores.max() > 0 else 1.0
        norm_scores = raw_scores / max_score

        top_n      = min(n_results, len(self.chunks))
        top_idx    = np.argsort(raw_scores)[::-1][:top_n]

        results: list[dict] = []
        for idx in top_idx:
            raw  = float(raw_scores[idx])
            norm = float(norm_scores[idx])

            if raw <= 0.0 or norm < min_score:
                continue

            c = self.chunks[idx]
            results.append({
                "chunk_id":        c.get("chunk_id",      f"bm25_{idx}"),
                "text":            c.get("text",          ""),
                "context_text":    c.get("context_text",  c.get("text", "")),
                "source":          c.get("source",        ""),
                "doc_type":        c.get("doc_type",      ""),
                "section":         c.get("section",       ""),
                "section_title":   c.get("section_title", ""),
                "chapter":         c.get("chapter",       ""),
                "chunk_type":      c.get("chunk_type",    ""),
                "bm25_score":      round(raw,  4),
                "bm25_score_norm": round(norm, 4),
            })

        return results

    # ── Utilities ─────────────────────────────────────────────────────────────

    def count(self) -> int:
        return len(self.chunks)

    def tokenize_query(self, query: str) -> list[str]:
        """Expose tokenizer for testing / debugging."""
        return tokenize(query)


# ── Singleton ─────────────────────────────────────────────────────────────────
bm25_retriever = BM25Retriever()
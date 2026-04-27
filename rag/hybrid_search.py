"""
LexShield AI — Hybrid Search (Vector + BM25 Fusion)
=====================================================
Combines ChromaDB cosine-similarity search with BM25 keyword search.

Fusion strategy: Reciprocal Rank Fusion (RRF)  [default]
  • RRF is robust to score-scale differences between vector and BM25
  • score(chunk) = Σ  1 / (k + rank_in_list)  across both lists
  • k=60 is the standard constant (smoother rank differences)

Also supports: weighted linear combination
  • score = α × vector_score_norm + (1-α) × bm25_score_norm

Why RRF over weighted?
  • No need to tune α per query type
  • Chunks found by both retrieval methods get a strong natural boost
  • Empirically better on mixed and semantic queries

Result dict keys (superset of Week 1 vector search results):
  chunk_id, text, source, doc_type, section, section_title,
  chapter, chunk_type, vector_score, bm25_score, bm25_score_norm,
  hybrid_score, retrieval_source
"""

import os
from typing import Literal

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from rag.vectorstore    import vectorstore
from rag.bm25_retriever import bm25_retriever

# ── RRF constant ──────────────────────────────────────────────────────────────
RRF_K = 60   # standard value from the original RRF paper

# ── ToC filter ────────────────────────────────────────────────────────────────

def _is_toc_chunk(text: str) -> bool:
    import re
    lines = text.strip().splitlines()
    if not lines:
        return False
    toc = sum(
        1 for l in lines
        if len(l.strip()) < 80 and (
            l.count('.') / max(len(l.strip()), 1) > 0.3
            or re.match(r'^\d[\d\s\.]+$', l.strip())
        )
    )
    return toc / max(len(lines), 1) > 0.65


# ── Fusion functions ──────────────────────────────────────────────────────────

def rrf_scores(
    vector_results: list[dict],
    bm25_results:   list[dict],
    k:              int = RRF_K,
) -> dict[str, float]:
    """
    Reciprocal Rank Fusion.
    Returns {chunk_id: rrf_score} for all chunks from both lists.
    """
    scores: dict[str, float] = {}
    for rank, r in enumerate(vector_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, r in enumerate(bm25_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def weighted_scores(
    vector_results: list[dict],
    bm25_results:   list[dict],
    alpha:          float = 0.6,
) -> dict[str, float]:
    """
    Weighted linear combination of normalised scores.
    α=0.6 gives a slight semantic-search advantage.
    """
    scores: dict[str, float] = {}
    for r in vector_results:
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + alpha * r.get("score", 0.0)
    for r in bm25_results:
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + (1 - alpha) * r.get("bm25_score_norm", 0.0)
    return scores


# ── Main hybrid searcher ──────────────────────────────────────────────────────

class HybridSearcher:
    """
    Two-stage retrieval:
      1. Retrieve fetch_k candidates from vector store + BM25 each
      2. Fuse ranks / scores → deduplicate → re-rank → return top n_results

    Parameters
    ----------
    fusion : "rrf" | "weighted"
        Fusion strategy (default: "rrf").
    alpha : float
        Weight for vector scores when fusion="weighted" (ignored for "rrf").
    fetch_multiplier : int
        How many extras to fetch before fusion. fetch_k = n_results * fetch_multiplier.
    """

    def __init__(
        self,
        fusion:           Literal["rrf", "weighted"] = "rrf",
        alpha:            float = 0.6,
        fetch_multiplier: int   = 3,
    ):
        self.fusion           = fusion
        self.alpha            = alpha
        self.fetch_multiplier = fetch_multiplier

    # ── Core search ───────────────────────────────────────────────────────────

    def search(
        self,
        query:            str,
        n_results:        int   = 8,
        min_vector_score: float = 0.20,
        filter_toc:       bool  = True,
    ) -> list[dict]:
        """
        Hybrid retrieval.

        Returns up to n_results dicts with hybrid_score field.
        Chunks found by both methods appear once with combined score.
        """
        fetch_k = n_results * self.fetch_multiplier

        # ── Step 1: Independent retrievals ───────────────────────────────────
        vector_raw  = vectorstore.search(query, n_results=fetch_k)
        vector_hits = [r for r in vector_raw if r.get("score", 0) >= min_vector_score]

        bm25_hits   = bm25_retriever.search(query, n_results=fetch_k)

        # ── Step 2: Build unified chunk lookup ────────────────────────────────
        lookup: dict[str, dict] = {}

        for r in vector_hits:
            cid = r["chunk_id"]
            lookup[cid] = {
                "chunk_id":        cid,
                "text":            r.get("text", ""),
                "source":          r.get("source", ""),
                "doc_type":        r.get("doc_type", ""),
                "section":         r.get("section", ""),
                "section_title":   r.get("section_title", ""),
                "chapter":         r.get("chapter", ""),
                "chunk_type":      r.get("chunk_type", ""),
                "vector_score":    r.get("score", 0.0),
                "bm25_score":      0.0,
                "bm25_score_norm": 0.0,
                "retrieval_source": "vector",
            }

        for r in bm25_hits:
            cid = r["chunk_id"]
            if cid in lookup:
                lookup[cid]["bm25_score"]      = r.get("bm25_score",      0.0)
                lookup[cid]["bm25_score_norm"]  = r.get("bm25_score_norm", 0.0)
                lookup[cid]["retrieval_source"] = "both"
                # Use text from BM25 (raw text, not context_text prefix) when available
                if r.get("text") and not lookup[cid].get("text"):
                    lookup[cid]["text"] = r["text"]
            else:
                lookup[cid] = {
                    "chunk_id":        cid,
                    "text":            r.get("text", ""),
                    "source":          r.get("source", ""),
                    "doc_type":        r.get("doc_type", ""),
                    "section":         r.get("section", ""),
                    "section_title":   r.get("section_title", ""),
                    "chapter":         r.get("chapter", ""),
                    "chunk_type":      r.get("chunk_type", ""),
                    "vector_score":    0.0,
                    "bm25_score":      r.get("bm25_score",      0.0),
                    "bm25_score_norm": r.get("bm25_score_norm", 0.0),
                    "retrieval_source": "bm25",
                }

        # ── Step 3: Compute fusion scores ─────────────────────────────────────
        if self.fusion == "rrf":
            fused = rrf_scores(vector_hits, bm25_hits)
        else:
            fused = weighted_scores(vector_hits, bm25_hits, self.alpha)

        # ── Step 4: Attach hybrid_score + sort ────────────────────────────────
        results: list[dict] = []
        for cid, chunk in lookup.items():
            chunk["hybrid_score"] = round(fused.get(cid, 0.0), 6)
            results.append(chunk)

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # ── Step 5: Optional ToC filtering ───────────────────────────────────
        if filter_toc:
            results = [
                r for r in results
                if not _is_toc_chunk(r.get("text", ""))
                and len(r.get("text", "").split()) >= 15
            ]

        return results[:n_results]

    # ── Debug / test helper ───────────────────────────────────────────────────

    def search_explain(
        self,
        query:     str,
        n_results: int = 5,
    ) -> list[dict]:
        """
        Like search() but attaches a human-readable score_breakdown to each result.
        Useful for comparing hybrid vs vector-only in tests.
        """
        results = self.search(query, n_results=n_results)
        for r in results:
            src = r.get("retrieval_source", "?")
            tag = {
                "vector": "V   ",
                "bm25":   " B  ",
                "both":   "V+B ",
            }.get(src, "?   ")
            r["score_breakdown"] = (
                f"{tag}| vector={r.get('vector_score',0):.3f} "
                f"bm25={r.get('bm25_score_norm',0):.3f} "
                f"hybrid={r.get('hybrid_score',0):.6f}"
            )
        return results

    # ── Pure vector search (for comparison baseline) ──────────────────────────

    def search_vector_only(
        self,
        query:     str,
        n_results: int   = 8,
        min_score: float = 0.25,
    ) -> list[dict]:
        """Week 1-style pure vector search. Used in test comparisons."""
        raw = vectorstore.search(query, n_results=n_results)
        return [r for r in raw if r.get("score", 0) >= min_score]


# ── Singleton ─────────────────────────────────────────────────────────────────
hybrid_searcher = HybridSearcher(fusion="rrf", alpha=0.6, fetch_multiplier=3)
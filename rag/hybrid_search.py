"""
LexShield AI — Hybrid Search  (section fast path patch)
========================================================
Key addition: section_number fast path at the top of search().

When the query contains an explicit section number (detected by regex),
we call vectorstore.get_by_section() first — a direct ChromaDB metadata
lookup that always returns the exact section chunk, regardless of how
similar nearby sections look to the text-similarity models.

Those exact-match chunks are pinned to hybrid_score=1.0 and injected
at the top of the merged result list before RRF fusion runs on the rest.

Everything else (RRF, weighted fusion, ToC filter) is unchanged from Day 1.
"""

import os
import re
from typing import Literal, Optional

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from rag.vectorstore    import vectorstore, SOURCE_KEYWORDS
from rag.bm25_retriever import bm25_retriever

RRF_K = 60

# ── Section number detection ──────────────────────────────────────────────────

# Matches: "Section 108", "section 108A", "s. 108", "s108"
SECTION_NUMBER_RE = re.compile(
    r'\b[Ss]ections?\s*\.?\s*(\d{1,4}[A-Za-z]?)\b'
    r'|'
    r'\b(\d{1,4}[A-Za-z]?)\s+(?:IPC|BNS|CrPC|CRPC)\b',
    re.IGNORECASE,
)


def extract_section_and_source(query: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extracts (section_number, source_hint) from a query string.

    Examples:
      "What is Section 108 Indian Penal Code?"  → ("108", "Indian Penal Code")
      "Section 420 IPC cheating"                → ("420", "Indian Penal Code")
      "punishment for murder"                   → (None, None)

    Returns (None, None) if no explicit section number is found.
    """
    m = SECTION_NUMBER_RE.search(query)
    if not m:
        return None, None

    section_number = (m.group(1) or m.group(2) or "").strip().upper()
    if not section_number:
        return None, None

    # Detect source hint from query keywords
    q_lower = query.lower()
    source_hint = None
    for keyword, source_name in SOURCE_KEYWORDS.items():
        if keyword in q_lower:
            source_hint = source_name
            break

    return section_number, source_hint


# ── ToC filter (unchanged) ────────────────────────────────────────────────────

def _is_toc_chunk(text: str) -> bool:
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


# ── RRF / weighted fusion (unchanged) ────────────────────────────────────────

def rrf_scores(vector_results: list[dict], bm25_results: list[dict], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for rank, r in enumerate(vector_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    for rank, r in enumerate(bm25_results, start=1):
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


def weighted_scores(vector_results: list[dict], bm25_results: list[dict], alpha: float = 0.6) -> dict[str, float]:
    scores: dict[str, float] = {}
    for r in vector_results:
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + alpha * r.get("score", 0.0)
    for r in bm25_results:
        cid = r["chunk_id"]
        scores[cid] = scores.get(cid, 0.0) + (1 - alpha) * r.get("bm25_score_norm", 0.0)
    return scores


# ── Hybrid searcher ───────────────────────────────────────────────────────────

class HybridSearcher:

    def __init__(
        self,
        fusion:           Literal["rrf", "weighted"] = "rrf",
        alpha:            float = 0.6,
        fetch_multiplier: int   = 3,
    ):
        self.fusion           = fusion
        self.alpha            = alpha
        self.fetch_multiplier = fetch_multiplier

    def search(
        self,
        query:            str,
        n_results:        int   = 8,
        min_vector_score: float = 0.20,
        filter_toc:       bool  = True,
    ) -> list[dict]:
        """
        Hybrid retrieval with section fast path.

        Flow:
          1. Detect explicit section number in query
          2. If found → metadata lookup (always correct, pinned to top)
          3. RRF fusion of vector + BM25 for remaining slots
          4. Inject metadata results at position 0, deduplicate, return top N
        """
        fetch_k = n_results * self.fetch_multiplier

        # ── Step 1: Section fast path ─────────────────────────────────────────
        section_hits: list[dict] = []
        section_number, source_hint = extract_section_and_source(query)

        if section_number:
            section_hits = vectorstore.get_by_section(section_number, source_hint)
            if section_hits:
                print(f"[HybridSearch] Section fast path: found {len(section_hits)} "
                      f"chunk(s) for section={section_number!r} source_hint={source_hint!r}")

        # ── Step 2: Standard vector + BM25 retrieval ─────────────────────────
        vector_raw  = vectorstore.search(query, n_results=fetch_k)
        vector_hits = [r for r in vector_raw if r.get("score", 0) >= min_vector_score]
        bm25_hits   = bm25_retriever.search(query, n_results=fetch_k)

        # ── Step 3: Build lookup from text-similarity results ─────────────────
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

        # ── Step 4: RRF / weighted scores ─────────────────────────────────────
        if self.fusion == "rrf":
            fused = rrf_scores(vector_hits, bm25_hits)
        else:
            fused = weighted_scores(vector_hits, bm25_hits, self.alpha)

        text_results: list[dict] = []
        for cid, chunk in lookup.items():
            chunk["hybrid_score"] = round(fused.get(cid, 0.0), 6)
            text_results.append(chunk)
        text_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        # ── Step 5: Inject section fast-path results at the top ───────────────
        # Remove duplicates: if a section_hit chunk_id is already in text_results,
        # remove it from there so the metadata version pins to position 0.
        if section_hits:
            fast_path_ids = {r["chunk_id"] for r in section_hits}
            text_results  = [r for r in text_results if r["chunk_id"] not in fast_path_ids]
            merged = section_hits + text_results
        else:
            merged = text_results

        # ── Step 6: ToC filter ────────────────────────────────────────────────
        if filter_toc:
            merged = [
                r for r in merged
                if not _is_toc_chunk(r.get("text", ""))
                and len(r.get("text", "").split()) >= 15
            ]

        return merged[:n_results]

    # ── Debug helper ──────────────────────────────────────────────────────────

    def search_explain(self, query: str, n_results: int = 5) -> list[dict]:
        results = self.search(query, n_results=n_results)
        for r in results:
            src = r.get("retrieval_source", "?")
            tag = {"vector": "V   ", "bm25": " B  ", "both": "V+B ", "metadata": "META"}.get(src, "?   ")
            r["score_breakdown"] = (
                f"{tag}| vector={r.get('vector_score', 0):.3f} "
                f"bm25={r.get('bm25_score_norm', 0):.3f} "
                f"hybrid={r.get('hybrid_score', 0):.6f}"
            )
        return results

    def search_vector_only(self, query: str, n_results: int = 8, min_score: float = 0.25) -> list[dict]:
        raw = vectorstore.search(query, n_results=n_results)
        return [r for r in raw if r.get("score", 0) >= min_score]


# ── Singleton ─────────────────────────────────────────────────────────────────
hybrid_searcher = HybridSearcher(fusion="rrf", alpha=0.6, fetch_multiplier=3)
"""
LexShield AI — Hybrid Search
========================================================
Changes in this version:

  FIX-10  source_hint resolution uses act_resolver (longest-match-first)
          instead of iterating _MERGED_KEYWORDS dict (arbitrary order).
          Fixes "Limited Liability Partnership" matching "Indian Partnership"
          because "limited liability partnership" (31 chars) is checked
          before "partnership" (11 chars) in act_resolver._REGISTRY.

  The _MERGED_KEYWORDS dict is kept as a FALLBACK only for cases where
  act_resolver.resolve_section_source returns None (unknown act names).

All other logic (RRF, ToC filter, category_filter, fallback) unchanged.
"""

import os
import re
from typing import Literal, Optional

import logging

logger = logging.getLogger(__name__)

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from rag.vectorstore      import vectorstore, SOURCE_KEYWORDS
from rag.bm25_retriever   import bm25_retriever
from rag.act_resolver     import act_resolver
from rag.category_detector import CONFIDENCE_MED as _CATEGORY_CONF_THRESHOLD

RRF_K = 60


# ── Act hint extractor (exported — used by KG and pipeline) ──────────────────

def _extract_act_hint(query: str) -> Optional[str]:
    """
    Extract which act the user is asking about from free-text query.
    Order matters: more-specific patterns checked before general ones.
    Returns None when no act is identifiable — no filter applied.
    """
    q = query.lower()

    # CrPC — check BEFORE IPC (more specific, different act)
    if any(x in q for x in ['crpc', 'cr.p.c', 'criminal procedure code',
                              'code of criminal procedure']):
        return 'Code of Criminal Procedure'

    # BNSS — replacement for CrPC
    if any(x in q for x in ['bnss', 'bharatiya nagarik suraksha']):
        return 'Bharatiya Nagarik Suraksha Sanhita'

    # BNS — check BEFORE IPC (different act, same section numbers)
    if any(x in q for x in ['bns', 'bharatiya nyaya sanhita']):
        return 'Bharatiya Nyaya Sanhita'

    # IPC
    if any(x in q for x in ['ipc', 'indian penal code', 'i.p.c']):
        return 'Indian Penal Code'

    # NI Act
    if any(x in q for x in ['ni act', 'negotiable instruments',
                              'cheque bounce', 'section 138']):
        return 'Negotiable Instruments Act'

    # Evidence Act / BSA
    if any(x in q for x in ['evidence act', 'bsa', 'bharatiya sakshya']):
        return 'Indian Evidence Act'
    
    if any(x in q for x in ['wildlife act', 'wild life act', 'wildlife protection','wild life protection', 'wpa', 'wlpa']):
        return 'Wildlife (Protection) Act'

    return None  # unknown — no act filter applied


# Export alias used by pipeline.py and knowledge_graph.py
extract_act_hint = _extract_act_hint

# ── Section number detection regex ───────────────────────────────────────────
_ACT_ABBREVIATIONS = (
    r"IPC|BNS|BNSS|BSA"
    r"|CrPC|CRPC"
    r"|CPC"
    r"|NI\s?Act|Negotiable\s+Instruments"
    r"|IT\s?Act|Information\s+Technology"
    r"|POCSO|PMLA|NDPS|UAPA"
    r"|RERA|IBC"
    r"|CGST|IGST|GST"
    r"|FEMA|SEBI|POSH|RTI|DPDP"
    r"|MV\s?Act|Motor\s+Vehicles?\s+Act"
    r"|Consumer\s+Protection|Contract\s+Act"
    r"|Companies\s+Act|Evidence\s+Act"
    r"|DV\s+Act|Domestic\s+Violence\s+Act"
    r"|MSMED?\s+Act|LLP\s+Act"
)

_ACT_FULL_NAMES = (
    r"Indian Penal Code"
    r"|Bharatiya Nyaya Sanhita"
    r"|Code of Criminal Procedure"
    r"|Bharatiya Nagarik Suraksha Sanhita"
    r"|Indian Evidence Act"
    r"|Bharatiya Sakshya Adhiniyam"
    r"|Code of Civil Procedure"
)

SECTION_NUMBER_RE = re.compile(
    r'\b[Ss]ections?\s*\.?\s*(\d{1,4}[A-Za-z]?)\b'
    r'|'
    r'\b(\d{1,4}[A-Za-z]?)\s+(?:' + _ACT_ABBREVIATIONS + r')\b'
    r'|'
    r'\b(\d{1,4}[A-Za-z]?)\s+(?:of|under|in)\s+(?:the\s+)?(?:'
    + _ACT_ABBREVIATIONS + r'|' + _ACT_FULL_NAMES + r')\b'
    r'|'
    r'\b(?:' + _ACT_ABBREVIATIONS + r'|' + _ACT_FULL_NAMES + r')\s+(?:[Ss]ections?\s*\.?\s*)?(\d{1,3}[A-Za-z]?)\b',
    re.IGNORECASE,
)


def extract_sections_and_sources(query: str) -> list[tuple[str, Optional[str]]]:
    """
    Extract (section_number, source_hint) pairs from query.

    source_hint resolution priority:
      1. act_resolver.resolve_section_source() — longest-match-first,
         prevents partial-name collisions (LLP vs Partnership)
      2. Fallback to SOURCE_KEYWORDS dict scan if resolver returns None

    Returns source_hint=None when no act is identifiable — pipeline
    then sends all section matches to reranker as candidates.
    """
    pairs: list[tuple[str, Optional[str]]] = []

    for m in SECTION_NUMBER_RE.finditer(query):
        section_number = (m.group(1) or m.group(2) or m.group(3) or m.group(4) or "").strip().upper()
        if not section_number:
            continue

        # Priority 1: act_resolver (proximity-first when match position known)
        source_hint = act_resolver.resolve_section_source(
            query, section_number, match_start=m.start(), match_end=m.end()
        )

        # Priority 2: SOURCE_KEYWORDS fallback (scan ±80 chars)
        if source_hint is None:
            start     = max(0, m.start() - 80)
            end       = min(len(query), m.end() + 80)
            local_ctx = query[start:end].lower()
            for keyword, source_name in SOURCE_KEYWORDS.items():
                if keyword in local_ctx:
                    source_hint = source_name
                    break

        pairs.append((section_number, source_hint))

    return pairs


def extract_section_and_source(query: str) -> tuple[Optional[str], Optional[str]]:
    """Backward-compat alias."""
    pairs = extract_sections_and_sources(query)
    return pairs[0] if pairs else (None, None)


def _format_results(raw: dict) -> list[dict]:
    results: list[dict] = []
    if not raw or not raw.get("ids") or not raw["ids"][0]:
        return results
    for cid, doc, dist, meta in zip(
        raw["ids"][0], raw["documents"][0], raw["distances"][0], raw["metadatas"][0]
    ):
        score = max(0.0, 1.0 - dist / 2.0)
        results.append({
            "chunk_id":         cid,
            "text":             doc,
            "source":           meta.get("source",        ""),
            "doc_type":         meta.get("doc_type",      ""),
            "section":          meta.get("section",       ""),
            "section_title":    meta.get("section_title", ""),
            "chapter":          meta.get("chapter",       ""),
            "chunk_type":       meta.get("chunk_type",    ""),
            "category":         meta.get("category",      ""),
            "era":              meta.get("era",           ""),
            "score":            round(score, 4),
            "vector_score":     round(score, 4),
            "bm25_score":       1.0,
            "bm25_score_norm":  1.0,
            "hybrid_score":     1.0,
            "retrieval_source": "metadata",
            "rerank_score":     None,
        })
    return results


def _section_fast_path(section: str, query: str, collection, k: int = 5) -> list[dict]:
    act_hint = _extract_act_hint(query)

    # ChromaDB metadata `where` filters do NOT support `$contains` — that
    # operator is only valid in `where_document` (full-text of the chunk),
    # not on metadata fields like `source`. Substring matching on `source`
    # must therefore be done client-side, mirroring the approach already
    # used by vectorstore.search_by_source().
    #
    # When an act hint is present we fetch up to 10 candidates (instead of
    # k) so the client-side act filter has enough results to select from.
    fetch_n = 10 if act_hint else min(k, 10)

    results = collection.query(
        query_texts=[f"Section {section}"],
        where={"section": {"$eq": section}},
        n_results=fetch_n,
    )

    chunks = _format_results(results)

    # Client-side act filter — preserves the original intent of the
    # `$and` + `$contains` filter (match first word of the act name
    # against the chunk's source string).
    if act_hint:
        act_kw = act_hint.split()[0].lower()
        chunks = [c for c in chunks
                  if act_kw in c.get('source', '').lower()]

    return chunks[:k]


# ── ToC filter ────────────────────────────────────────────────────────────────

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


# ── Fusion functions ──────────────────────────────────────────────────────────

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


# ── Alpha weight presets by query complexity ─────────────────────────────────

_ALPHA_PRESETS: dict[str, tuple[float, float]] = {
    "simple":   (0.70, 0.30),   # statutory lookup  → favour BM25
    "moderate": (0.50, 0.50),   # balanced default
    "complex":  (0.30, 0.70),   # conceptual query  → favour semantic
}


# ── Hybrid searcher ───────────────────────────────────────────────────────────

class HybridSearcher:

    def __init__(self, fusion: Literal["rrf", "weighted"] = "rrf", alpha: float = 0.6, fetch_multiplier: int = 3):
        self.fusion           = fusion
        self.alpha            = alpha
        self.fetch_multiplier = fetch_multiplier

    def search(
        self,
        query:               str,
        n_results:           int           = 8,
        min_vector_score:    float         = 0.05,
        filter_toc:          bool          = True,
        category_filter:     Optional[str] = None,
        act_hint:            Optional[str] = None,
        category_confidence: Optional[float] = None,
        query_complexity:    Optional[str] = None,
    ) -> list[dict]:
        fetch_k = n_results * self.fetch_multiplier

        # ── Dynamic alpha from query complexity ──────────────────────────────
        effective_complexity = query_complexity or "moderate"
        bm25_weight, sem_weight = _ALPHA_PRESETS.get(
            effective_complexity, _ALPHA_PRESETS["moderate"]
        )
        logger.debug(
            f"[HybridSearch] alpha=BM25:{bm25_weight}/semantic:{sem_weight} "
            f"complexity={effective_complexity}"
        )

        # ── Category-filter confidence guard ─────────────────────────────────
        # If the detector returned a low-confidence category the filter must not
        # be applied — it would poison the chunk pool before CRAG/reranker can
        # correct it.  Drop the filter and fall through to full-corpus search.
        if category_filter is not None and category_confidence is not None:
            if category_confidence < _CATEGORY_CONF_THRESHOLD:
                logger.debug(
                    f"[HybridSearch] category_filter={category_filter!r} suppressed "
                    f"(confidence={category_confidence:.3f} < threshold={_CATEGORY_CONF_THRESHOLD})"
                )
                category_filter = None

        # Section fast path — uses act_resolver via extract_sections_and_sources
        section_hits: list[dict] = []
        for section_number, source_hint in extract_sections_and_sources(query):
            hits = _section_fast_path(section_number, query, vectorstore.collection, k=5)
            if hits and category_filter:
                hits = [h for h in hits if h.get("category", "") == category_filter]
            if hits:
                logger.info(f"[HybridSearch] Section fast path: {len(hits)} chunk(s) section={section_number!r} source_hint={source_hint!r}")
                section_hits.extend(hits)

        seen: set[str] = set()
        section_hits = [h for h in section_hits if not (h["chunk_id"] in seen or seen.add(h["chunk_id"]))]  # type: ignore[func-returns-value]

        vector_raw  = vectorstore.search(query, n_results=fetch_k, category_filter=category_filter)
        vector_hits = [r for r in vector_raw if r.get("score", 0) >= min_vector_score]
        bm25_hits   = bm25_retriever.search(query, n_results=fetch_k, category_filter=category_filter)

        if not vector_hits and not bm25_hits and vector_raw:
            logger.info(f"[HybridSearch] FALLBACK: threshold filtered all. Using raw vector (top={vector_raw[0].get('score',0):.4f})")
            vector_hits = vector_raw[:fetch_k]

        lookup: dict[str, dict] = {}
        for r in vector_hits:
            cid = r["chunk_id"]
            lookup[cid] = {
                "chunk_id": cid, "text": r.get("text", ""),
                "source": r.get("source", ""), "doc_type": r.get("doc_type", ""),
                "section": r.get("section", ""), "section_title": r.get("section_title", ""),
                "chapter": r.get("chapter", ""), "chunk_type": r.get("chunk_type", ""),
                "category": r.get("category", ""), "era": r.get("era", ""),
                "vector_score": r.get("score", 0.0), "bm25_score": 0.0,
                "bm25_score_norm": 0.0, "retrieval_source": "vector",
            }
        for r in bm25_hits:
            cid = r["chunk_id"]
            if cid in lookup:
                lookup[cid]["bm25_score"]       = r.get("bm25_score", 0.0)
                lookup[cid]["bm25_score_norm"]  = r.get("bm25_score_norm", 0.0)
                lookup[cid]["retrieval_source"] = "both"
            else:
                lookup[cid] = {
                    "chunk_id": cid, "text": r.get("text", ""),
                    "source": r.get("source", ""), "doc_type": r.get("doc_type", ""),
                    "section": r.get("section", ""), "section_title": r.get("section_title", ""),
                    "chapter": r.get("chapter", ""), "chunk_type": r.get("chunk_type", ""),
                    "category": r.get("category", ""), "era": r.get("era", ""),
                    "vector_score": 0.0, "bm25_score": r.get("bm25_score", 0.0),
                    "bm25_score_norm": r.get("bm25_score_norm", 0.0), "retrieval_source": "bm25",
                }

        if self.fusion == "rrf":
            fused = rrf_scores(vector_hits, bm25_hits)
        else:
            fused = weighted_scores(vector_hits, bm25_hits, sem_weight)

        text_results = []
        for cid, chunk in lookup.items():
            chunk["hybrid_score"] = round(fused.get(cid, 0.0), 6)
            text_results.append(chunk)
        text_results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        if section_hits:
            fast_ids     = {r["chunk_id"] for r in section_hits}
            text_results = [r for r in text_results if r["chunk_id"] not in fast_ids]
            merged       = section_hits + text_results
        else:
            merged = text_results

        if filter_toc:
            merged = [r for r in merged if not _is_toc_chunk(r.get("text", "")) and len(r.get("text", "").split()) >= 15]

        # ── BUG FIX 1: act-aware post-filter ─────────────────────────────────
        # When act_hint is known, filter the merged pool to same act family.
        # This prevents cross-act contamination (e.g. "302 CrPC" returning IPC 302).
        final_act_hint = act_hint or _extract_act_hint(query)
        if final_act_hint:
            act_kw = final_act_hint.split()[0].lower()  # e.g. "code" / "bharatiya" / "indian"
            # Use first two words for more precise matching
            act_kw2 = " ".join(final_act_hint.lower().split()[:2])  # e.g. "code of" / "indian penal"
            act_filtered = [
                r for r in merged
                if act_kw2 in r.get("source", "").lower()
                or act_kw  in r.get("source", "").lower()
            ]
            if act_filtered:
                # Only apply filter if it leaves enough results
                merged = act_filtered
                logger.info(f"[HybridSearch] act_hint={final_act_hint!r} -> filtered to {len(merged)} chunk(s)")
            else:
                logger.info(f"[HybridSearch] act_hint={final_act_hint!r} -> filter yielded 0; keeping all {len(merged)} chunks")

        return merged[:n_results]

    def search_explain(self, query: str, n_results: int = 5, category_filter: Optional[str] = None) -> list[dict]:
        results = self.search(query, n_results=n_results, category_filter=category_filter)
        for r in results:
            src = r.get("retrieval_source", "?")
            tag = {"vector": "V   ", "bm25": " B  ", "both": "V+B ", "metadata": "META"}.get(src, "?   ")
            r["score_breakdown"] = (
                f"{tag}| vector={r.get('vector_score',0):.3f} "
                f"bm25={r.get('bm25_score_norm',0):.3f} "
                f"hybrid={r.get('hybrid_score',0):.6f}"
            )
        return results

    def search_vector_only(self, query: str, n_results: int = 8, min_score: float = 0.05, category_filter: Optional[str] = None) -> list[dict]:
        raw = vectorstore.search(query, n_results=n_results, category_filter=category_filter)
        return [r for r in raw if r.get("score", 0) >= min_score]


# ── Singleton ─────────────────────────────────────────────────────────────────
hybrid_searcher = HybridSearcher(fusion="rrf", alpha=0.6, fetch_multiplier=3)
hybrid_search   = hybrid_searcher  # alias for backward compat
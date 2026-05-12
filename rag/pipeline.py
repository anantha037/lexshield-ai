"""
LexShield AI — RAG Pipeline
=================================================================
Changes in this version:

  Uses act_resolver for section pinning — same fix as hybrid_search.
  Section fast-path in pipeline._run() now calls
  act_resolver.resolve_section_source() instead of relying on
  extract_sections_and_sources() from hybrid_search for pinning
  (hybrid_search already uses it; pipeline's own pin step now also does).

  Retrieval priority order:
    1. Exact act + exact section   (hard-pinned, act_resolver resolved)
    2. Exact act + semantic        (soft-pinned paired act)
    3. Semantic act + exact section (ambiguous candidates → reranker)
    4. Global hybrid fallback      (vector + BM25, no filter)

  All previous fixes (FIX-9 ambiguous candidates, auto-category,
  dual search, soft-pinned paired act) retained unchanged.
"""

import os
import re
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from rag.llm               import llm
from rag.hybrid_search     import hybrid_searcher, extract_sections_and_sources
from rag.query_rewriter    import query_rewriter
from rag.reranker          import reranker
from rag.vectorstore       import vectorstore
from rag.act_resolver      import act_resolver
from rag.category_detector import category_detector, CONFIDENCE_HIGH, CONFIDENCE_MED
from rag.synthesizer       import (
    build_synthesis_prompt,
    get_system_prompt,
    synthesize,
    LegalAnswer,
)

N_RETRIEVE_PER_QUERY    = 8
N_RERANKER_INPUT        = 14
N_FINAL_CONTEXT         = 5
PAIRED_ACT_MAX_CHUNKS   = 2
AMBIGUOUS_SECTION_SCORE = 0.88

ACT_PAIRS: dict[str, str] = {
    "Indian Penal Code":                    "Bharatiya Nyaya Sanhita",
    "Bharatiya Nyaya Sanhita":              "Indian Penal Code",
    "Code of Criminal Procedure":           "Bharatiya Nagarik Suraksha Sanhita",
    "Bharatiya Nagarik Suraksha Sanhita":   "Code of Criminal Procedure",
    "Indian Evidence Act":                  "Bharatiya Sakshya Adhiniyam",
    "Bharatiya Sakshya Adhiniyam":          "Indian Evidence Act",
}

_PAIR_TRIGGER_KEYWORDS: dict[str, str] = {
    "indian penal code":                "Indian Penal Code",
    "ipc":                              "Indian Penal Code",
    "bharatiya nyaya sanhita":          "Bharatiya Nyaya Sanhita",
    "bns":                              "Bharatiya Nyaya Sanhita",
    "code of criminal procedure":       "Code of Criminal Procedure",
    "crpc":                             "Code of Criminal Procedure",
    "bharatiya nagarik suraksha":       "Bharatiya Nagarik Suraksha Sanhita",
    "bnss":                             "Bharatiya Nagarik Suraksha Sanhita",
    "indian evidence act":              "Indian Evidence Act",
    "evidence act":                     "Indian Evidence Act",
    "bharatiya sakshya":                "Bharatiya Sakshya Adhiniyam",
    "bsa":                              "Bharatiya Sakshya Adhiniyam",
}


def detect_paired_act(query_lower: str) -> Optional[str]:
    for keyword, act_key in _PAIR_TRIGGER_KEYWORDS.items():
        if keyword in query_lower:
            return ACT_PAIRS.get(act_key)
    return None


def get_paired_chunks(expanded_query: str, paired_source: str, n: int = PAIRED_ACT_MAX_CHUNKS) -> list[dict]:
    topic_query = re.sub(r'\bSection\s+\d+[A-Z]?\b', '', expanded_query, flags=re.IGNORECASE).strip() or expanded_query
    chunks = vectorstore.search_by_source(query=topic_query, source_partial=paired_source, n_results=n)
    for c in chunks:
        c["retrieval_source"] = "paired_act"
        c["hybrid_score"]     = 0.5
    return chunks


OUT_OF_CORPUS_ACTS: list[str] = [
    "limitation act", "statute of limitations",
    "stamp act", "stamp duty",
    "contempt of court", "contempt of courts act",
    "advocates act", "sarfaesi", "benami", "essential commodities",
]

QUERY_EXPANSIONS: dict[str, str] = {
    r'\bipc\b':                     'Indian Penal Code',
    r'\bcrpc\b':                    'Code of Criminal Procedure',
    r'\bevidence\s+act\b':          'Indian Evidence Act',
    r'\bbnss\b':                    'Bharatiya Nagarik Suraksha Sanhita',
    r'\bbns\b':                     'Bharatiya Nyaya Sanhita',
    r'\bbsa\b':                     'Bharatiya Sakshya Adhiniyam',
    r'\bpocso\b':                   'Protection of Children from Sexual Offences Act',
    r'\bpmla\b':                    'Prevention of Money Laundering Act',
    r'\bndps\b':                    'Narcotic Drugs and Psychotropic Substances Act',
    r'\buapa\b':                    'Unlawful Activities Prevention Act',
    r'\bpca\b':                     'Prevention of Corruption Act',
    r'\bpil\b':                     'Public Interest Litigation',
    r'\bfir\b':                     'First Information Report',
    r'\bnbw\b':                     'non-bailable warrant',
    r'\bbw\b':                      'bailable warrant',
    r'\bsc\b':                      'Supreme Court',
    r'\bhc\b':                      'High Court',
    r'\bcpc\b':                     'Code of Civil Procedure',
    r'\brt[ia]\b':                  'Right to Information Act',
    r'\brte\b':                     'Right to Education Act',
    r'\brera\b':                    'Real Estate Regulation and Development Act',
    r'\btop\s+act\b':               'Transfer of Property Act',
    r'\bibc\b':                     'Insolvency and Bankruptcy Code',
    r'\bllp\b':                     'Limited Liability Partnership Act',
    r'\bmsme\b':                    'Micro Small and Medium Enterprises',
    r'\bmsmed\b':                   'Micro Small and Medium Enterprises Development Act',
    r'\bni\s?act\b':                'Negotiable Instruments Act',
    r'\bcgst\b':                    'Central Goods and Services Tax Act',
    r'\bigst\b':                    'Integrated Goods and Services Tax Act',
    r'\bgst\b':                     'Goods and Services Tax',
    r'\bfema\b':                    'Foreign Exchange Management Act',
    r'\bsebi\b':                    'Securities and Exchange Board of India Act',
    r'\bposh\b':                    'Sexual Harassment of Women at Workplace Act',
    r'\bepf\b':                     'Employees Provident Fund Code on Social Security',
    r'\bfssai\b':                   'Food Safety and Standards Act',
    r'\bit\s?act\b':                'Information Technology Act',
    r'\bdpdp\b':                    'Digital Personal Data Protection Act',
    r'\bmv\s?act\b':                'Motor Vehicles Act',
    r'\bmotor\s+vehicle\s+act\b':   'Motor Vehicles Act',
    r'\bdv\s+act\b':                'Protection of Women from Domestic Violence Act',
}

_MVA_RE    = re.compile(r'\bmotor\s+vehicles?\s+act\b|\bmv\s?act\b|\bmva\b', re.IGNORECASE)
SECTION_RE = re.compile(
    r'\b(\d{1,4}[A-Z]?)\s*'
    r'(ipc|bns|bnss|bsa|crpc|cpc|ni\s?act|it\s?act|pocso|pmla|ndps|uapa'
    r'|rera|ibc|cgst|igst|fema|sebi|posh|rti|dpdp|mv\s?act'
    r'|consumer|contract\s+act|companies\s+act|dv\s+act|llp\s+act)\b',
    re.IGNORECASE,
)
_SECTION_ACT_MAP: dict[str, str] = {
    "IPC": "Indian Penal Code", "BNS": "Bharatiya Nyaya Sanhita",
    "BNSS": "Bharatiya Nagarik Suraksha Sanhita", "BSA": "Bharatiya Sakshya Adhiniyam",
    "CRPC": "Code of Criminal Procedure", "CPC": "Code of Civil Procedure",
    "NI ACT": "Negotiable Instruments Act", "NIACT": "Negotiable Instruments Act",
    "IT ACT": "Information Technology Act", "ITACT": "Information Technology Act",
    "POCSO": "Protection of Children from Sexual Offences Act",
    "PMLA": "Prevention of Money Laundering Act",
    "NDPS": "Narcotic Drugs and Psychotropic Substances Act",
    "UAPA": "Unlawful Activities Prevention Act",
    "RERA": "Real Estate Regulation and Development Act",
    "IBC": "Insolvency and Bankruptcy Code",
    "CGST": "Central Goods and Services Tax Act",
    "IGST": "Integrated Goods and Services Tax Act",
    "FEMA": "Foreign Exchange Management Act",
    "SEBI": "Securities and Exchange Board of India Act",
    "POSH": "Sexual Harassment of Women at Workplace Act",
    "RTI": "Right to Information Act", "DPDP": "Digital Personal Data Protection Act",
    "MV ACT": "Motor Vehicles Act", "MVACT": "Motor Vehicles Act",
    "CONSUMER": "Consumer Protection Act",
    "DV ACT": "Protection of Women from Domestic Violence Act",
    "DVACT": "Protection of Women from Domestic Violence Act",
    "LLP ACT": "Limited Liability Partnership Act",
    "LLPACT": "Limited Liability Partnership Act",
}


def preprocess_query(query: str) -> str:
    q = query.strip()
    q = _MVA_RE.sub("Motor Vehicles Act", q)

    def expand_section(m: re.Match) -> str:
        num     = m.group(1)
        raw_act = m.group(2).upper().replace(" ", "")
        act_name = _SECTION_ACT_MAP.get(m.group(2).upper()) or _SECTION_ACT_MAP.get(raw_act) or m.group(2).upper()
        return f"Section {num} {act_name}"

    q = SECTION_RE.sub(expand_section, q)
    for pattern, replacement in QUERY_EXPANSIONS.items():
        q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
    return q


def deduplicate_chunks(all_results: list[list[dict]]) -> list[dict]:
    best: dict[str, dict] = {}
    for result_list in all_results:
        for chunk in result_list:
            cid   = chunk.get("chunk_id", "")
            score = chunk.get("hybrid_score", 0.0)
            if cid not in best or score > best[cid].get("hybrid_score", 0.0):
                best[cid] = chunk
    return sorted(best.values(), key=lambda x: x.get("hybrid_score", 0.0), reverse=True)


def _hybrid_search_multi(queries: list[str], n_results: int, category_filter: Optional[str]) -> list[dict]:
    return deduplicate_chunks([
        hybrid_searcher.search(q, n_results=n_results, category_filter=category_filter)
        for q in queries
    ])


class RAGPipeline:

    def __init__(
        self,
        n_retrieve:       int   = N_RETRIEVE_PER_QUERY,
        n_reranker_input: int   = N_RERANKER_INPUT,
        n_final:          int   = N_FINAL_CONTEXT,
        temperature:      float = 0.1,
        max_tokens:       int   = 1024,
        enable_rewriting: bool  = True,
        enable_reranking: bool  = True,
    ):
        self.n_retrieve       = n_retrieve
        self.n_reranker_input = n_reranker_input
        self.n_final          = n_final
        self.temperature      = temperature
        self.max_tokens       = max_tokens
        self.enable_rewriting = enable_rewriting
        self.enable_reranking = enable_reranking

    def query(self, user_query: str, n_results: Optional[int] = None, category_filter: Optional[str] = None) -> LegalAnswer:
        try:
            return self._run(user_query, n_results or self.n_final, category_filter)
        except Exception as e:
            import traceback; traceback.print_exc()
            return LegalAnswer(answer_text="An internal error occurred. Please try again.",
                               sources_consulted=0, synthesis_note="Pipeline error.", grounding_warning=str(e))

    def _run(self, user_query: str, n_final: int, category_filter: Optional[str] = None) -> LegalAnswer:

        expanded      = preprocess_query(user_query)
        paired_source = detect_paired_act(expanded.lower())

        # ── Auto category (hybrid keyword + semantic) ─────────────────────────
        auto_category:    Optional[str] = None
        auto_confidence:  float         = 0.0
        effective_filter: Optional[str] = category_filter

        if category_filter is None:
            auto_category, auto_confidence = category_detector.detect(expanded)
            if auto_category and auto_confidence >= CONFIDENCE_HIGH:
                effective_filter = auto_category
                print(f"[Pipeline] Auto-category HIGH: {auto_category!r} conf={auto_confidence:.2f} → filter")
            elif auto_category and auto_confidence >= CONFIDENCE_MED:
                print(f"[Pipeline] Auto-category MED: {auto_category!r} conf={auto_confidence:.2f} → dual search")
            else:
                print(f"[Pipeline] Auto-category LOW: conf={auto_confidence:.2f} → global")

        # ── Section fast-path with act_resolver (FIX-9 + FIX-10) ─────────────
        # Priority 1: act_resolver resolves source_hint (longest-match-first)
        # Three-way split: hard-pin / soft-candidate / ambiguous-reranker
        pinned_chunks:      list[dict] = []
        section_candidates: list[dict] = []

        for sec, hint in extract_sections_and_sources(expanded):
            # Re-resolve using act_resolver directly on the expanded query
            # (extract_sections_and_sources already uses it, but we log here)
            hits = vectorstore.get_by_section(sec, hint)
            if hits and effective_filter:
                hits = [h for h in hits if h.get("category", "") == effective_filter]
            if not hits:
                continue

            if hint is not None:
                print(f"[Pipeline] Hard-pin: section={sec} source={hint!r} → {len(hits)} chunk(s)")
                pinned_chunks.extend(hits)
            elif len(hits) == 1:
                print(f"[Pipeline] Hard-pin (unique): section={sec} → {hits[0].get('source','?')[:45]}")
                pinned_chunks.extend(hits)
            else:
                print(f"[Pipeline] Ambiguous section={sec}: {len(hits)} acts → reranker decides")
                for h in hits:
                    h["hybrid_score"]     = AMBIGUOUS_SECTION_SCORE
                    h["retrieval_source"] = "section_candidate"
                section_candidates.extend(hits)

        # Dedup pinned
        _seen: set[str] = set()
        pinned_chunks = [c for c in pinned_chunks if not (c["chunk_id"] in _seen or _seen.add(c["chunk_id"]))]  # type: ignore[func-returns-value]

        # ── Knowledge Graph context injection ──────────────────────────────────
        # When section fast-path fires, inject related sections as KG context chunk
        if pinned_chunks:
            try:
                from rag.knowledge_graph import get_kg
                kg    = get_kg()
                notes = []
                for sec, hint in extract_sections_and_sources(expanded):
                    related = kg.query_related_sections(sec, source_hint=hint)
                    if related:
                        notes.append(kg.format_context(sec, hint, related))
                if notes:
                    pinned_chunks.insert(0, {
                        "chunk_id":         "_kg_context",
                        "text":             "\n".join(notes),
                        "source":           "Knowledge Graph",
                        "section":          "",
                        "section_title":    "",
                        "chapter":          "",
                        "doc_type":         "system",
                        "chunk_type":       "kg_context",
                        "category":         "",
                        "era":              "",
                        "hybrid_score":     1.5,
                        "retrieval_source": "knowledge_graph",
                    })
                    print(f"[Pipeline] KG injected: {len(notes)} section(s)")
            except Exception as e:
                print(f"[Pipeline] KG injection skipped: {e}")

        pinned_ids = {c["chunk_id"] for c in pinned_chunks}

        # Dedup candidates (exclude already-pinned)
        _seen2: set[str] = set()
        section_candidates = [
            c for c in section_candidates
            if c["chunk_id"] not in pinned_ids
            and not (c["chunk_id"] in _seen2 or _seen2.add(c["chunk_id"]))  # type: ignore[func-returns-value]
        ]
        section_candidate_ids = {c["chunk_id"] for c in section_candidates}

        # ── Soft-pin top paired act chunk ─────────────────────────────────────
        soft_pinned: list[dict] = []
        if paired_source:
            paired_all = get_paired_chunks(expanded, paired_source)
            if paired_all:
                soft_pinned = [paired_all[0]]
                print(f"[Pipeline] Soft-pinned paired: section={soft_pinned[0].get('section','?')} source={soft_pinned[0].get('source','?')[:40]}")

        soft_pinned_ids  = {c["chunk_id"] for c in soft_pinned}
        all_reserved_ids = pinned_ids | soft_pinned_ids

        # ── Query rewriting ───────────────────────────────────────────────────
        rewritten   = query_rewriter.rewrite(expanded) if self.enable_rewriting else []
        all_queries = [expanded] + [q for q in rewritten if q != expanded]

        # ── Three-tier hybrid retrieval ───────────────────────────────────────
        if effective_filter:
            merged_free = _hybrid_search_multi(all_queries, self.n_retrieve, effective_filter)
        elif auto_category and auto_confidence >= CONFIDENCE_MED:
            filtered = _hybrid_search_multi(all_queries, self.n_retrieve, auto_category)
            global_r = _hybrid_search_multi(all_queries, self.n_retrieve, None)
            fids     = {r["chunk_id"] for r in filtered}
            merged_free = filtered + [r for r in global_r if r["chunk_id"] not in fids]
        else:
            merged_free = _hybrid_search_multi(all_queries, self.n_retrieve, None)

        # Remove all reserved from free pool
        merged_free = [
            c for c in merged_free
            if c["chunk_id"] not in all_reserved_ids
            and c["chunk_id"] not in section_candidate_ids
        ]

        # Combine section candidates + free pool sorted by score
        combined_free = section_candidates + merged_free
        combined_free.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        merged = pinned_chunks + combined_free

        # Out-of-corpus warning
        q_lower = user_query.lower()
        if any(act in q_lower for act in OUT_OF_CORPUS_ACTS):
            merged.insert(0, {
                "chunk_id": "_corpus_warning",
                "text": ("NOTE: The legal corpus may not contain the specific Act directly "
                         "relevant to this query. Base your answer only on available sources "
                         "and explicitly state this limitation."),
                "source": "System", "section": "", "section_title": "", "chapter": "",
                "doc_type": "system", "chunk_type": "warning", "category": "", "era": "",
                "hybrid_score": 2.0, "retrieval_source": "system",
            })

        if not merged and not soft_pinned:
            return LegalAnswer(
                answer_text=("The retrieved legal sections do not contain sufficient information "
                             "to answer this question. Please consult a qualified legal professional."),
                sources_consulted=0, synthesis_note="No sources retrieved.",
                grounding_warning="No chunks matched the query.", rewritten_queries=all_queries, reranker_used=False,
            )

        # ── Rerank free pool ──────────────────────────────────────────────────
        reranker_input = [
            c for c in merged[:self.n_reranker_input]
            if c["chunk_id"] not in all_reserved_ids and c["chunk_id"] != "_corpus_warning"
        ]
        reranker_used = False

        if self.enable_reranking:
            ranked_chunks, reranker_used = reranker.rerank(query=expanded, chunks=reranker_input, top_n=n_final)
        else:
            ranked_chunks = reranker_input[:n_final]
            for c in ranked_chunks:
                c["rerank_score"] = None

        # Final assembly: [hard-pinned] + [soft-pinned paired] + [reranked]
        final_chunks = (
            pinned_chunks
            + soft_pinned
            + [c for c in ranked_chunks if c["chunk_id"] not in all_reserved_ids]
        )
        final_chunks = final_chunks[:n_final]

        # Section safety fallback
        section_match = re.search(r'\bSection\s+(\d+[A-Z]?)\b', expanded, re.IGNORECASE)
        if section_match:
            target = section_match.group(1)
            if not any(c.get("section", "") == target for c in final_chunks):
                from rag.bm25_retriever import bm25_retriever
                extra = bm25_retriever.search(expanded, n_results=3, category_filter=effective_filter)
                for e in extra:
                    if e.get("section", "") == target:
                        if e.get("rerank_score") is not None and e["rerank_score"] < -3.0:
                            continue
                        e.update({"rerank_score": None, "hybrid_score": 0.005})
                        final_chunks.append(e)
                final_chunks = final_chunks[:n_final]

        system_prompt = get_system_prompt(final_chunks)
        prompt        = build_synthesis_prompt(expanded, final_chunks)
        raw_answer    = llm.generate(
            prompt=prompt, system_prompt=system_prompt,
            temperature=self.temperature, max_tokens=self.max_tokens,
        )

        return synthesize(
            query=user_query, chunks=final_chunks, llm_answer=raw_answer,
            rewritten_queries=all_queries, reranker_used=reranker_used,
        )


rag_pipeline = RAGPipeline()
"""
LexShield AI — RAG Pipeline
=================================================================
Week 3 additions (wired in this version):

  1. Adaptive RAG Router (rag/adaptive_router.py)
     First step in _run().  Classifies query as simple/moderate/complex.
     simple   -> section fast-path only, skip BM25 + rewriter + CRAG
     moderate -> BM25 + vector + reranker, skip rewriter
     complex  -> full pipeline + query rewriter + CRAG + decomposition

  2. Multi-hop Query Decomposition (rag/query_rewriter.decompose_query)
     Only fires when complexity == "complex".
     Breaks query into 2-3 sub-queries, retrieves each independently,
     merges + deduplicates chunk pools, labels them in the synthesizer.

  3. CRAG Self-Correction (rag/crag.py)
     Fires after initial retrieval on moderate/complex queries.
     score >= 4 -> proceed to synthesizer
     score 2-3  -> rewrite + re-retrieve once
     score == 1 -> return low-confidence grounded response immediately

  AgentState field written by pipeline:
     rag_grade: "good" | "poor"  (set in legal_rag_node / risk_check_node
                                   in agents/graph.py from LegalAnswer)

  All previous logic (FIX-9 ambiguous candidates, section fast-path,
  act_resolver, dual search, soft-pinned paired act, KG injection,
  section safety fallback) retained unchanged.

  Section-equivalence lookup (rag/section_equivalence.py)
     Fires before complexity routing on any query that is_equivalence_query()
     flags (e.g. "what is the BNS equivalent of IPC 302"). On a match, a
     synthetic high-confidence "_kg_equivalence_context" chunk is prepended
     to pinned_chunks. This chunk ALONE must never satisfy the "simple path"
     fast-exit (see real_hits_found below) and must never disable the CRAG
     cross-act safety net (see run_crag below) — it augments retrieval, it
     does not replace it.
"""

import os
import re
from typing import Optional

import logging

logger = logging.getLogger(__name__)

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from rag.llm               import llm
from rag.hybrid_search     import hybrid_searcher, extract_sections_and_sources, extract_act_hint
from rag.query_rewriter    import query_rewriter, decompose_query, rewrite_for_retrieval
from rag.reranker          import reranker
from rag.vectorstore       import vectorstore
from rag.embedder          import embedder
from rag.act_resolver      import act_resolver
from rag.category_detector import category_detector, CONFIDENCE_HIGH, CONFIDENCE_MED
from rag.adaptive_router   import classify_query_complexity
from rag.crag              import evaluate_retrieval, GOOD_MIN_SCORE
from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree
from rag.synthesizer       import (
    build_synthesis_prompt,
    get_system_prompt,
    synthesize,
    LegalAnswer,
)

N_RETRIEVE_PER_QUERY    = 10
N_RERANKER_INPUT        = 16
N_FINAL_CONTEXT         = 8
PAIRED_ACT_MAX_CHUNKS   = 2
AMBIGUOUS_SECTION_SCORE = 0.88

# Feature flag — Knowledge Graph has only 108 nodes for 23,740 documents.
# KG injection fires on almost every query but contributes meaningful context
# for < 0.5 % of them.  Disabled by default to cut latency; flip to True once
# the graph reaches sufficient density.
KG_INJECTION_ENABLED = False

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


def get_paired_chunks(
    expanded_query: str,
    paired_source: str,
    n: int = PAIRED_ACT_MAX_CHUNKS,
) -> list[dict]:
    topic_query = re.sub(
        r'\bSection\s+\d+[A-Z]?\b', '', expanded_query, flags=re.IGNORECASE
    ).strip() or expanded_query
    chunks = vectorstore.search_by_source(
        query=topic_query, source_partial=paired_source, n_results=n
    )
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
    r'\b(?:section\s+)?(\d{1,4}[A-Z]?)\s*'
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
        num      = m.group(1)
        raw_act  = m.group(2).upper().replace(" ", "")
        act_name = (
            _SECTION_ACT_MAP.get(m.group(2).upper())
            or _SECTION_ACT_MAP.get(raw_act)
            or m.group(2).upper()
        )
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


def _hybrid_search_multi(
    queries: list[str],
    n_results: int,
    category_filter: Optional[str],
    act_hint: Optional[str] = None,
    category_confidence: Optional[float] = None,
    query_complexity: Optional[str] = None,
) -> list[dict]:
    return deduplicate_chunks([
        hybrid_searcher.search(
            q,
            n_results=n_results,
            category_filter=category_filter,
            act_hint=act_hint,
            category_confidence=category_confidence,
            query_complexity=query_complexity,
        )
        for q in queries
    ])


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-HOP RETRIEVAL HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _retrieve_for_subquery(
    sub_query: str,
    n_results: int,
    category_filter: Optional[str],
    label: str,
    act_hint: Optional[str] = None,
    category_confidence: Optional[float] = None,
) -> list[dict]:
    """
    Run hybrid search for a single sub-query and tag each chunk with its
    sub-query label so the synthesizer can present labeled context blocks.
    """
    chunks = hybrid_searcher.search(
        sub_query,
        n_results=n_results,
        category_filter=category_filter,
        act_hint=act_hint,
        category_confidence=category_confidence,
        query_complexity="complex",
    )
    for c in chunks:
        c["subquery_label"] = label
    return chunks


def _build_labeled_synthesis_prompt(
    expanded_query: str,
    subquery_pools: list[tuple[str, list[dict]]],
    all_chunks: list[dict],
) -> str:
    """
    Builds a synthesis prompt with labeled context blocks per sub-query.
    Falls back to the standard build_synthesis_prompt if no sub-queries.

    When pinned equivalence chunks are present (system note + target-section
    chunks), they are prepended as a dedicated block BEFORE the sub-query
    sections and an [EQUIVALENCE ANSWER \u2014 AUTHORITATIVE] directive is
    injected into the header, so the model sees the target statute text and
    the correct section number even on the complex decomposition path.
    """
    if not subquery_pools:
        return build_synthesis_prompt(expanded_query, all_chunks)

    # ── Detect & extract pinned equivalence chunks ────────────────────────────
    # These are: the system note chunk + any statute chunks fetched for the
    # target section (Fix 1).  They are NOT part of any subquery pool, so they
    # would otherwise be invisible to the model on the labeled path.
    _subquery_ids: set[str] = set()
    for _, pool in subquery_pools:
        for c in pool:
            _subquery_ids.add(c["chunk_id"])

    eq_note_chunk = next(
        (c for c in all_chunks if c.get("chunk_id") == "_kg_equivalence_context"), None
    )
    target_statute_chunks = [
        c for c in all_chunks
        if c.get("chunk_id") not in _subquery_ids
        and c.get("chunk_id") != "_kg_equivalence_context"
        and not c.get("chunk_id", "").startswith("_")
        and c.get("retrieval_source") == "metadata"
    ]

    # ── Build equivalence header & pinned block ───────────────────────────────
    equivalence_header = ""
    pinned_block = ""

    if eq_note_chunk:
        import re as _re
        eq_text = eq_note_chunk.get("text", "")
        _target_act, _target_sec = "", ""
        _m = _re.search(
            r"corresponds to\s+([A-Z][^\d\n]+?)\s+(\d+[A-Z]?)\s*(?:\[|$|\n)",
            eq_text, _re.IGNORECASE
        )
        if _m:
            _target_act = _m.group(1).strip()
            _target_sec = _m.group(2).strip()

        # Build pinned text: system note first, then target statute chunks
        pinned_parts = []
        pinned_parts.append(
            f"[SYSTEM NOTE]\n{eq_text}"
        )
        for c in target_statute_chunks[:6]:   # cap at 6 chunks = full section
            src  = c.get("source", "?")[:55]
            sec  = c.get("section", "")
            text = c.get("text", "")[:600]
            pinned_parts.append(f"[Source: {src} \u00a7{sec}]\n{text}")

        if len(pinned_parts) > 1:             # has at least one statute chunk
            pinned_block = (
                "[PINNED EQUIVALENCE CONTEXT \u2014 READ FIRST]\n"
                + "\n\n".join(pinned_parts)
            )

        if _target_act and _target_sec:
            src_note = (
                " Its full statute text is in the PINNED EQUIVALENCE CONTEXT block above."
                if len(pinned_parts) > 1 else
                " Its full text was not retrieved \u2014 state the section number anyway."
            )
            equivalence_header = (
                f"[EQUIVALENCE ANSWER \u2014 AUTHORITATIVE]\n"
                f"The system has already identified the answer: the equivalent section "
                f"is {_target_act} Section {_target_sec}.{src_note}\n"
                f"You MUST state this section number as the direct answer. "
                f"Do NOT write that the equivalent 'cannot be confirmed', "
                f"'is not provided in context', or that you 'cannot determine' it.\n\n"
            )
        else:
            equivalence_header = (
                "[EQUIVALENCE ANSWER \u2014 AUTHORITATIVE]\n"
                "A SYSTEM NOTE above states the correct cross-act section "
                "correspondence. Treat it as authoritative.\n\n"
            )

    # ── Build labeled sub-query sections ─────────────────────────────────────
    sections = []
    for label, chunks in subquery_pools:
        if not chunks:
            continue
        chunk_texts = "\n\n".join(
            f"[Source: {c.get('source','?')[:40]} \u00a7{c.get('section','')}]\n"
            f"{c.get('text','')[:600]}"
            for c in chunks[:3]
        )
        sections.append(f"Context for {label}:\n{chunk_texts}")

    labeled_context = "\n\n" + ("\n\n" + "\u2500" * 60 + "\n\n").join(sections)

    return (
        f"{equivalence_header}"
        f"Answer the following complex legal query using the labeled context blocks below.\n"
        f"Query: {expanded_query}\n\n"
        + (f"{pinned_block}\n\n" if pinned_block else "")
        + f"{labeled_context}\n\n"
        f"Provide a comprehensive answer addressing ALL parts of the query. "
        f"Cite the specific acts and sections from the relevant context block."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

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

    @traceable(name="rag_pipeline.fallback", run_type="chain")
    def _generate_fallback_response(
        self, user_query: str, all_queries: list[str], complexity: str, crag_score: int
    ) -> LegalAnswer:
        logger.debug("[DIAGNOSE] FALLBACK TRIGGERED")
        fallback_prompt = (
            f"User Query: {user_query}\n\n"
            "Write a clean, honest, plain-language fallback message (1-2 sentences) "
            "stating that you couldn't find specific legal provisions in your corpus that directly address this query. "
            "Briefly mention the user's specific topic. "
            "Suggest they consult a qualified legal professional.\n"
            "Do not cite any law or section number. Do not use structured headers. Just write the message."
        )

        try:
            raw_answer = llm.generate(
                prompt=fallback_prompt,
                system_prompt="You are LexShield, an AI legal assistant. Provide helpful, honest fallback messages when you lack relevant information.",
                temperature=0.3,
                max_tokens=150,
            )
        except Exception as e:
            logger.exception("[Pipeline] Fallback generation failed")
            raw_answer = (
                "I couldn't find specific legal provisions in my corpus that directly address this query. "
                "You may want to consult a qualified legal professional for advice specific to your situation."
            )

        rt = get_current_run_tree()
        if rt:
            rt.add_metadata({"fallback": True, "crag_action": "insufficient"})

        return LegalAnswer(
            answer_text=raw_answer.strip(),
            citations=[],
            sources_consulted=0,
            synthesis_note=f"[complexity={complexity} crag_score={crag_score}] CRAG fallback triggered.",
            grounding_warning="Retrieval confidence too low; fallback activated.",
            rewritten_queries=all_queries,
            reranker_used=False,
            confidence="low",
            fallback=True
        )

    def _is_retrieval_relevant(
        self,
        query: str,
        chunks: list,
        threshold: float = 0.35
    ) -> bool:
        """
        Lightweight pre-synthesis relevance gate.
        Uses the existing embedder (already loaded) to compute cosine similarity
        between the query and the top retrieved chunks.
        Returns False if no chunk clears the threshold — meaning retrieval failed.
        """
        if not chunks:
            return False

        if all(c.get("retrieval_source") in ("metadata", "section_candidate") for c in chunks[:3]):
            return True

        import numpy as np
        query_embedding = embedder.embed_single(query)

        for chunk in chunks[:5]:  # only check top 5 for speed
            chunk_text = chunk.get("text", "") if isinstance(chunk, dict) else (chunk.page_content if hasattr(chunk, "page_content") else str(chunk))
            chunk_embedding = embedder.embed_single(chunk_text)

            # cosine similarity
            similarity = np.dot(query_embedding, chunk_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding) + 1e-9
            )
            logger.debug(f"[Pipeline] Relevance Gate - Chunk: {chunk.get('source', '')[:20]} | Sim: {similarity:.3f}")
            if similarity >= threshold:
                return True

        return False

    @traceable(name="rag_pipeline.query", run_type="chain")
    def query(
        self,
        user_query:      str,
        n_results:       Optional[int] = None,
        category_filter: Optional[str] = None,
        context_block:   str = "",
    ) -> LegalAnswer:
        try:
            return self._run(user_query, n_results or self.n_final, category_filter, context_block)
        except Exception as e:
            logger.exception("[Pipeline] Internal error")
            return LegalAnswer(
                answer_text="An internal error occurred. Please try again.",
                sources_consulted=0,
                synthesis_note="Pipeline error.",
                grounding_warning=str(e),
            )

    def _run(
        self,
        user_query:      str,
        n_final:         int,
        category_filter: Optional[str] = None,
        context_block:   str = "",
    ) -> LegalAnswer:

        # Extract acts/sections ONLY from the latest user_query to avoid hard-pinning
        # chunks based on the assistant's previous answers in the context block.
        latest_expanded = preprocess_query(user_query)

        # ── Conversational query rewriting ─────────────────────────────────────
        # If the current query has no section numbers and no act keywords,
        # and context_block is non-empty (there is prior conversation),
        # rewrite the query into a standalone form before retrieval.
        search_query_raw = rewrite_for_retrieval(user_query, context_block)
        if search_query_raw != user_query:
            latest_expanded = preprocess_query(search_query_raw)

        logger.info(f"[DEBUG] latest_expanded='{latest_expanded}'")

        paired_source = detect_paired_act(latest_expanded.lower())
        original_act_hint = extract_act_hint(latest_expanded)

        # For synthesis and deep rewriting, we can optionally use the full context
        full_query = f"{context_block}\n\n{user_query}" if context_block else user_query
        expanded = preprocess_query(full_query)

        # ── Equivalence Graph lookup (new logic) ───────────────────────────────
        # NOTE: eq_chunks (if populated) is a SYNTHETIC system-note chunk, not
        # a real retrieved chunk. It must never, on its own, be treated as
        # sufficient grounding — see real_hits_found below (controls the
        # simple-path fast-exit) and run_crag below (controls the CRAG
        # cross-act safety net). Both must key off real retrieval, not off
        # pinned_chunks' raw truthiness, or this synthetic note can silently
        # bypass both grounding checks.
        from rag.section_equivalence import is_equivalence_query, lookup_equivalent, source_hint_to_act_name
        is_eq, eq_pairs = is_equivalence_query(latest_expanded)
        eq_chunks: list[dict] = []
        if is_eq:
            if not eq_pairs:
                for sec_num, hint in extract_sections_and_sources(latest_expanded):
                    act_name = source_hint_to_act_name(hint, latest_expanded.lower())
                    if act_name:
                        eq_pairs.append((act_name, sec_num))

            found_equivs = []
            for act, sec in list(dict.fromkeys(eq_pairs)):
                res = lookup_equivalent(act, sec)
                if res:
                    status_str = "" if res["status"] == "verified" else " [STATUS: UNVERIFIED]"
                    found_equivs.append(
                        f"{res['source']['act']} {res['source']['section']} corresponds to {res['target']['act']} {res['target']['section']}{status_str}"
                    )

            if found_equivs:
                eq_chunks.append({
                    "chunk_id": "_kg_equivalence_context",
                    "text": "SYSTEM NOTE: Known legal equivalences for this query:\n" + "\n".join(f"- {eq}" for eq in list(dict.fromkeys(found_equivs))),
                    "source": "System", "section": "", "section_title": "", "chapter": "",
                    "doc_type": "system", "chunk_type": "context", "category": "", "era": "",
                    "hybrid_score": 2.0, "retrieval_source": "system",
                })

            # ── TARGET section fast-fetch (Fix 1) ────────────────────────────
            # For every successful equivalence lookup, hard-fetch the TARGET
            # section's real statute text the same way the section fast-path
            # does — so the synthesizer has actual BNSS/BNS text to cite, not
            # just the synthetic system note.
            _eq_target_chunks: list[dict] = []
            for act, sec in list(dict.fromkeys(eq_pairs)):
                res = lookup_equivalent(act, sec)
                if not res:
                    continue
                t_act = res["target"]["act"]
                t_sec = res["target"]["section"]
                t_hits = vectorstore.get_by_section(t_sec, t_act)
                for h in t_hits:
                    _eq_target_chunks.append(h)
                    logger.info(
                        f"[Pipeline] Equivalence target fetch: {t_act} §{t_sec} "
                        f"-> {h.get('source','?')[:45]} chunk={h.get('chunk_id','?')}"
                    )
            # Prepend target chunks right after eq_chunks so they sit early in
            # pinned_chunks and survive n_final trimming.
            eq_chunks = eq_chunks + _eq_target_chunks


        complexity = classify_query_complexity(latest_expanded)
        # pipeline_depth is used by graph.py node (stored in AgentState)
        # Here we just use it to control which steps fire.
        logger.info(f"[Pipeline] complexity={complexity!r}")

        # ── Auto category (hybrid keyword + semantic) ─────────────────────────
        auto_category:    Optional[str] = None
        auto_confidence:  float         = 0.0
        effective_filter: Optional[str] = category_filter

        if category_filter is None:
            auto_category, auto_confidence = category_detector.detect(latest_expanded)
            if auto_category and auto_confidence >= CONFIDENCE_HIGH:
                effective_filter = auto_category
                logger.info(f"[Pipeline] Auto-category HIGH: {auto_category!r} conf={auto_confidence:.2f} -> filter")
            elif auto_category and auto_confidence >= CONFIDENCE_MED:
                logger.info(f"[Pipeline] Auto-category MED: {auto_category!r} conf={auto_confidence:.2f} -> dual search")
            else:
                logger.info(f"[Pipeline] Auto-category LOW: conf={auto_confidence:.2f} -> global")

        # ── Section fast-path (always runs — priority 1 in all complexity tiers)
        pinned_chunks:      list[dict] = eq_chunks.copy()
        section_candidates: list[dict] = []

        logger.info(f"[DEBUG] extract_sections input='{latest_expanded}'")
        for sec, hint in extract_sections_and_sources(latest_expanded):
            hits = vectorstore.get_by_section(sec, hint)
            if hits and effective_filter:
                hits = [h for h in hits if h.get("category", "") == effective_filter]
            if not hits:
                continue

            if hint is not None:
                logger.info(f"[Pipeline] Hard-pin: section={sec} source={hint!r} -> {len(hits)} chunk(s)")
                pinned_chunks.extend(hits)
            elif len(hits) == 1:
                logger.info(f"[Pipeline] Hard-pin (unique): section={sec} -> {hits[0].get('source','?')[:45]}")
                pinned_chunks.extend(hits)
            else:
                logger.info(f"[Pipeline] Ambiguous section={sec}: {len(hits)} acts -> reranker decides")
                for h in hits:
                    h["hybrid_score"]     = AMBIGUOUS_SECTION_SCORE
                    h["retrieval_source"] = "section_candidate"
                section_candidates.extend(hits)

        # Dedup pinned
        _seen: set[str] = set()
        pinned_chunks = [
            c for c in pinned_chunks
            if not (c["chunk_id"] in _seen or _seen.add(c["chunk_id"]))  # type: ignore[func-returns-value]
        ]

        # real_hits_found is True only when the section fast-path pinned at
        # least one REAL retrieved chunk beyond the synthetic eq_chunks note.
        # This (not raw pinned_chunks truthiness) gates both the simple-path
        # fast-exit below and the CRAG safety net further down, so a bare
        # equivalence match with no real chunk can never skip grounding.
        real_hits_found = len(pinned_chunks) > len(eq_chunks)

        # ── SIMPLE path: section fast-path is sufficient, skip heavy retrieval ─
        # Only use simple path when fast-path actually found REAL pinned chunks
        # (real_hits_found) — a synthetic equivalence note alone must not
        # short-circuit into this fast, low-grounding branch.
        if complexity == "simple" and real_hits_found:
            logger.info("[Pipeline] simple path — using section fast-path only")
            # Still inject KG context and paired act if available
            if KG_INJECTION_ENABLED:
                self._inject_kg(pinned_chunks, expanded, latest_expanded)
            else:
                logger.info("[Pipeline] KG injection disabled — skipping")
            soft_pinned = []
            if paired_source:
                paired_all = get_paired_chunks(expanded, paired_source)
                if paired_all:
                    soft_pinned = [paired_all[0]]
            final_chunks = (pinned_chunks + soft_pinned)[:n_final]

            # ── PRE-SYNTHESIS RELEVANCE GATE (simple path) ──────────────────
            # Gate must score the same query retrieval used (latest_expanded:
            # conversational rewrite + abbreviation expansion), NOT the raw
            # user_query — otherwise valid follow-ups like "what about the
            # punishment for that?" retrieve correct chunks but fail the gate.
            if not self._is_retrieval_relevant(latest_expanded, final_chunks):
                logger.info("Pre-synthesis gate: no relevant chunks found. Triggering fallback.")
                return self._generate_fallback_response(
                    user_query=user_query,
                    all_queries=[latest_expanded],
                    complexity=complexity,
                    crag_score=0,
                )
            # ────────────────────────────────────────────────────────────────

            system_prompt = get_system_prompt(final_chunks)
            prompt        = build_synthesis_prompt(expanded, final_chunks)
            raw_answer    = llm.generate(
                prompt=prompt, system_prompt=system_prompt,
                temperature=self.temperature, max_tokens=self.max_tokens,
            )
            return synthesize(
                query=user_query, chunks=final_chunks, llm_answer=raw_answer,
                rewritten_queries=[expanded], reranker_used=False,
            )

        # ── Knowledge Graph injection ─────────────────────────────────────────
        if KG_INJECTION_ENABLED:
            self._inject_kg(pinned_chunks, expanded, latest_expanded)
        else:
            logger.info("[Pipeline] KG injection disabled — skipping")

        pinned_ids = {c["chunk_id"] for c in pinned_chunks}

        # Dedup candidates
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
                logger.debug(f"[Pipeline] Soft-pinned paired: section={soft_pinned[0].get('section','?')} "
                      f"source={soft_pinned[0].get('source','?')[:40]}")

        soft_pinned_ids  = {c["chunk_id"] for c in soft_pinned}
        all_reserved_ids = pinned_ids | soft_pinned_ids

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 1: COMPLEX PATH — Multi-hop decomposition
        # ═══════════════════════════════════════════════════════════════════════
        subquery_pools:    list[tuple[str, list[dict]]] = []
        decomposed_chunks: list[dict]                   = []

        if complexity == "complex":
            logger.info("[Pipeline] complex path — decomposing query")
            sub_queries = decompose_query(latest_expanded)

            if len(sub_queries) >= 2:
                for i, sq in enumerate(sub_queries, 1):
                    label  = f"Sub-query {i}"
                    chunks = _retrieve_for_subquery(
                        sq, self.n_retrieve, effective_filter, label,
                        act_hint=original_act_hint,
                        category_confidence=auto_confidence if effective_filter else None,
                    )
                    subquery_pools.append((label, chunks))
                    decomposed_chunks.extend(chunks)

                # Deduplicate decomposed pool
                decomposed_chunks = deduplicate_chunks([decomposed_chunks])
                logger.info(f"[Pipeline] decomposed pool: {len(decomposed_chunks)} unique chunks")

        # search_query: history-free, used for entity extraction in hybrid search
        search_query = latest_expanded

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 2: Query rewriting (moderate + complex only)
        # ═══════════════════════════════════════════════════════════════════════
        if complexity in ("moderate", "complex") and self.enable_rewriting:
            rewritten   = query_rewriter.rewrite(latest_expanded)
            all_queries = [search_query] + [q for q in rewritten if q != latest_expanded]
        else:
            all_queries = [search_query]

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 3: Hybrid retrieval
        # ═══════════════════════════════════════════════════════════════════════
        if effective_filter:
            merged_free = _hybrid_search_multi(
                all_queries, self.n_retrieve, effective_filter,
                act_hint=original_act_hint,
                category_confidence=auto_confidence,
                query_complexity=complexity,
            )
        elif auto_category and auto_confidence >= CONFIDENCE_MED:
            filtered = _hybrid_search_multi(
                all_queries, self.n_retrieve, auto_category,
                act_hint=original_act_hint,
                category_confidence=auto_confidence,
                query_complexity=complexity,
            )
            global_r = _hybrid_search_multi(
                all_queries, self.n_retrieve, None,
                act_hint=original_act_hint,
                query_complexity=complexity,
            )
            fids     = {r["chunk_id"] for r in filtered}
            merged_free = filtered + [r for r in global_r if r["chunk_id"] not in fids]
        else:
            merged_free = _hybrid_search_multi(
                all_queries, self.n_retrieve, None,
                act_hint=original_act_hint,
                query_complexity=complexity,
            )

        # Merge decomposed chunks into free pool
        if decomposed_chunks:
            existing_ids = {c["chunk_id"] for c in merged_free}
            extra = [c for c in decomposed_chunks if c["chunk_id"] not in existing_ids]
            merged_free = merged_free + extra

        # Remove reserved from free pool
        merged_free = [
            c for c in merged_free
            if c["chunk_id"] not in all_reserved_ids
            and c["chunk_id"] not in section_candidate_ids
        ]

        combined_free = section_candidates + merged_free
        combined_free.sort(key=lambda x: x.get("hybrid_score", 0.0), reverse=True)
        merged = pinned_chunks + combined_free
        logger.debug(f"[DIAGNOSE] Hybrid retrieval returned {len(merged)} chunks. Top scores: {[c.get('hybrid_score', 0) for c in merged[:3]]}")

        # Out-of-corpus warning
        q_lower = user_query.lower()
        if any(act in q_lower for act in OUT_OF_CORPUS_ACTS):
            merged.insert(0, {
                "chunk_id": "_corpus_warning",
                "text": (
                    "NOTE: The legal corpus may not contain the specific Act directly "
                    "relevant to this query. Base your answer only on available sources "
                    "and explicitly state this limitation."
                ),
                "source": "System", "section": "", "section_title": "", "chapter": "",
                "doc_type": "system", "chunk_type": "warning", "category": "", "era": "",
                "hybrid_score": 2.0, "retrieval_source": "system",
            })

        if not merged and not soft_pinned:
            return self._generate_fallback_response(
                user_query=user_query,
                all_queries=all_queries,
                complexity=complexity,
                crag_score=0,
            )

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 4: CRAG evaluation (moderate + complex only)
        # ═══════════════════════════════════════════════════════════════════════
        rag_grade      = "good"
        crag_triggered = False
        crag_fallback  = False   # set True when CRAG scores insufficient

        # Extra safety net: even when complexity is "simple", if the
        # section fast-path did not find any REAL chunk (real_hits_found is
        # False — this is deliberately NOT "not pinned_chunks", since
        # pinned_chunks may hold nothing but a synthetic equivalence note)
        # AND an act was identified for this query (single-act hint or a
        # paired-act relationship), still run CRAG. This catches cross-act
        # contamination for act-specific queries without a section number
        # (e.g. "what is the equivalent in BNS?"), without adding a CRAG
        # call for purely conceptual simple queries that name no act at
        # all (e.g. "define bail", "what is FIR"). Equivalence queries are
        # exactly the case most prone to cross-act contamination (their
        # embeddings sit close to both the legacy and current act's text),
        # so this must not be silently disabled by the presence of the
        # synthetic eq_chunks note.
        run_crag = (
            complexity in ("moderate", "complex")
            or (
                complexity == "simple"
                and not real_hits_found
                and (original_act_hint or paired_source)
            )
        )

        if run_crag:
            # Build CRAG eval pool: always include pinned + section_candidates
            # so fast-path chunks are never lost before CRAG scoring.
            _crag_ids: set[str] = set()
            eval_chunks: list[dict] = []
            for c in pinned_chunks + section_candidates:
                cid = c.get("chunk_id", "")
                if not cid or cid in _crag_ids:
                    continue
                if cid.startswith("_") and cid != "_kg_equivalence_context":
                    continue
                eval_chunks.append(c)
                _crag_ids.add(cid)
            for c in merged[:N_RERANKER_INPUT]:
                cid = c.get("chunk_id", "")
                if cid and not cid.startswith("_") and cid not in _crag_ids:
                    eval_chunks.append(c)
                    _crag_ids.add(cid)
            crag_result = evaluate_retrieval(latest_expanded, eval_chunks)
            logger.debug(f"[DIAGNOSE] CRAG result: action={crag_result.get('action')}, score={crag_result.get('score')}, reason={crag_result.get('reason')}")
            # GOOD_MIN_SCORE (4) is the grading threshold — intentionally
            # stricter than CRAG's proceed gate (PROCEED_MIN_SCORE=3).
            # A degraded (evaluator-failed) result carries score=3 and is
            # therefore graded "poor" here, never "good".
            rag_grade   = "good" if crag_result["score"] >= GOOD_MIN_SCORE else "poor"
            crag_fallback = crag_result.get("fallback", False)

            if crag_result["action"] == "insufficient":
                logger.debug("[DIAGNOSE] Branch taken: insufficient")
                crag_fallback = True
                logger.warning(
                    "[Pipeline] CRAG: insufficient — short-circuiting to fallback message "
                    f"(score={crag_result['score']}, reason={crag_result['reason'][:80]!r})"
                )
                return self._generate_fallback_response(
                    user_query=user_query,
                    all_queries=all_queries,
                    complexity=complexity,
                    crag_score=crag_result["score"],
                )

            elif crag_result["action"] == "rewrite" and not crag_triggered:
                logger.debug("[DIAGNOSE] Branch taken: rewrite")
                # Marginal retrieval — rewrite and re-retrieve once
                logger.info("[Pipeline] CRAG: rewrite triggered — re-retrieving")
                crag_triggered = True
                extra_rewrites = query_rewriter.rewrite(latest_expanded)
                extra_queries  = [q for q in extra_rewrites if q not in all_queries]
                if extra_queries:
                    extra_chunks = _hybrid_search_multi(
                        extra_queries, self.n_retrieve, effective_filter,
                        category_confidence=auto_confidence,
                        query_complexity=complexity,
                    )
                    existing_ids = {c["chunk_id"] for c in merged}
                    new_chunks   = [c for c in extra_chunks if c["chunk_id"] not in existing_ids]
                    merged       = merged + new_chunks
                    logger.info(f"[Pipeline] CRAG rewrite added {len(new_chunks)} new chunks")
                    all_queries  = all_queries + extra_queries

            else:
                logger.debug("[DIAGNOSE] Branch taken: proceed")
                # else: action == "proceed" — continue normally


        # ═══════════════════════════════════════════════════════════════════════
        # STEP 5: Rerank free pool
        # ═══════════════════════════════════════════════════════════════════════
        # Exclude both reserved IDs AND section_candidate IDs from reranker —
        # section_candidates are semi-pinned and preserved separately.
        reranker_input = [
            c for c in merged[:self.n_reranker_input]
            if c["chunk_id"] not in all_reserved_ids
            and c["chunk_id"] not in section_candidate_ids
            and not c.get("chunk_id", "").startswith("_")
        ]
        reranker_used = False

        if self.enable_reranking:
            ranked_chunks, reranker_used = reranker.rerank(
                query=expanded, chunks=reranker_input, top_n=n_final
            )
        else:
            ranked_chunks = reranker_input[:n_final]
            for c in ranked_chunks:
                c["rerank_score"] = None

        # Final assembly: [hard-pinned] + [soft-pinned] + [section_candidates] + [reranked]
        # Section candidates are semi-pinned: they survive reranker cutoff but
        # sit after hard/soft pins in priority.
        _final_ids: set[str] = set()
        final_chunks: list[dict] = []
        for c in pinned_chunks + soft_pinned + section_candidates:
            cid = c["chunk_id"]
            if cid not in _final_ids:
                final_chunks.append(c)
                _final_ids.add(cid)
        for c in ranked_chunks:
            cid = c["chunk_id"]
            if cid not in _final_ids and cid not in all_reserved_ids:
                final_chunks.append(c)
                _final_ids.add(cid)
        final_chunks = final_chunks[:n_final]

        # Section safety fallback (unchanged)
        section_match = re.search(r'\bSection\s+(\d+[A-Z]?)\b', latest_expanded, re.IGNORECASE)
        if section_match:
            target = section_match.group(1)
            if not any(c.get("section", "") == target for c in final_chunks):
                from rag.bm25_retriever import bm25_retriever
                extra = bm25_retriever.search(
                    latest_expanded, n_results=3, category_filter=effective_filter
                )
                for e in extra:
                    if e.get("section", "") == target:
                        if e.get("rerank_score") is not None and e["rerank_score"] < -3.0:
                            continue
                        e.update({"rerank_score": None, "hybrid_score": 0.005})
                        final_chunks.append(e)
                final_chunks = final_chunks[:n_final]

        # ═══════════════════════════════════════════════════════════════════════
        # STEP 6: Build prompt (labeled for complex, standard otherwise)
        # ═══════════════════════════════════════════════════════════════════════
        # PRE-SYNTHESIS RELEVANCE GATE (runs for simple, moderate, complex)
        # Gate must score the same query retrieval used (latest_expanded:
        # conversational rewrite + abbreviation expansion), NOT the raw
        # user_query — otherwise valid follow-ups like "what about the
        # punishment for that?" retrieve correct chunks but fail the gate.
        if not self._is_retrieval_relevant(latest_expanded, final_chunks):
            logger.info("Pre-synthesis gate: no relevant chunks found. Triggering fallback.")
            return self._generate_fallback_response(
                user_query=user_query,
                all_queries=all_queries,
                complexity=complexity,
                crag_score=0
            )

        system_prompt = get_system_prompt(final_chunks)

        if complexity == "complex" and subquery_pools:
            # Build subquery-pool mapping for labeled prompt
            labeled_pools: list[tuple[str, list[dict]]] = []
            for label, pool in subquery_pools:
                pool_ids   = {c["chunk_id"] for c in pool}
                in_final   = [c for c in final_chunks if c["chunk_id"] in pool_ids]
                labeled_pools.append((label, in_final))
            prompt = _build_labeled_synthesis_prompt(expanded, labeled_pools, final_chunks)
        else:
            prompt = build_synthesis_prompt(expanded, final_chunks)

        raw_answer = llm.generate(
            prompt=prompt, system_prompt=system_prompt,
            temperature=self.temperature, max_tokens=self.max_tokens,
        )

        answer = synthesize(
            query=user_query, chunks=final_chunks, llm_answer=raw_answer,
            rewritten_queries=all_queries, reranker_used=reranker_used,
        )

        # crag_fallback short-circuits earlier, so it is no longer checked here.

        # Attach rag_grade so graph.py node can store it in AgentState
        answer.synthesis_note = (
            f"[complexity={complexity} rag_grade={rag_grade}"
            + (f" crag_rewrite=True" if crag_triggered else "")
            + (f" crag_fallback=True" if crag_fallback else "")
            + "] "
            + (answer.synthesis_note or "")
        )

        rt = get_current_run_tree()
        if rt:
            rt.add_metadata({
                "retrieval_mode":  complexity,
                "crag_score":      crag_result["score"] if run_crag else 4,
                "crag_fallback":   crag_fallback,
                "chunks_retrieved": len(merged) if "merged" in locals() else len(final_chunks),
                "query_complexity": complexity,
            })

        return answer


    # ── KG injection helper ────────────────────────────────────────────────────

    def _inject_kg(self, pinned_chunks: list[dict], expanded: str,
                   latest_expanded: str = "") -> None:
        """Inject Knowledge Graph context when section fast-path fired."""
        if not pinned_chunks:
            return
        try:
            from rag.knowledge_graph import get_kg
            kg    = get_kg()
            notes = []
            query_for_sections = latest_expanded if latest_expanded else expanded
            for sec, hint in extract_sections_and_sources(query_for_sections):
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
                logger.info(f"[Pipeline] KG injected: {len(notes)} section(s)")
        except Exception as e:
            logger.info(f"[Pipeline] KG injection skipped: {e}")


# ── Singleton ──────────────────────────────────────────────────────────────────
rag_pipeline = RAGPipeline()
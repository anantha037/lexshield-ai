"""
LexShield AI — RAG Pipeline  (Week 2, Day 3)
============================================
Unchanged from Day 2 up to reranking.
Day 3 change: replaces bare llm.generate() call with full synthesis flow.

  query → preprocess → rewrite → hybrid×N → dedup → rerank
        → build_synthesis_prompt → llm.generate → synthesize()
        → LegalAnswer  (structured citations + grounding check)
"""

import os
import re
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from rag.llm            import llm
from rag.hybrid_search  import hybrid_searcher
from rag.query_rewriter import query_rewriter
from rag.reranker       import reranker
from rag.synthesizer    import (
    build_synthesis_prompt,
    SYNTHESIS_SYSTEM_PROMPT,
    synthesize,
    LegalAnswer,
)

# ── Constants ─────────────────────────────────────────────────────────────────
N_RETRIEVE_PER_QUERY = 8
N_RERANKER_INPUT     = 10
N_FINAL_CONTEXT      = 5

# ── Query preprocessor ────────────────────────────────────────────────────────
QUERY_EXPANSIONS: dict[str, str] = {
    r'\bipc\b':      'Indian Penal Code',
    r'\bbnss?\b':    'Bharatiya Nyaya Sanhita',
    r'\bcrpc\b':     'Code of Criminal Procedure',
    r'\bpil\b':      'Public Interest Litigation',
    r'\bfir\b':      'First Information Report',
    r'\bnbw\b':      'non-bailable warrant',
    r'\bbw\b':       'bailable warrant',
    r'\bsc\b':       'Supreme Court',
    r'\bhc\b':       'High Court',
    r'\bdb\b':       'Division Bench',
    r'\bsb\b':       'Single Bench',
    r'\brt[ia]\b':   'Right to Information Act',
    r'\bmv\s?act\b': 'Motor Vehicles Act',
    r'\bndps\b':     'Narcotic Drugs and Psychotropic Substances Act',
    r'\bpocso\b':    'Protection of Children from Sexual Offences Act',
    r'\bpca\b':      'Prevention of Corruption Act',
}

SECTION_RE = re.compile(
    r'\b(\d{1,4}[A-Z]?)\s*(ipc|bnss?|crpc|cpc|it\s?act|consumer|mv\s?act)\b',
    re.IGNORECASE,
)


def preprocess_query(query: str) -> str:
    q = query.strip()
    def expand_section(m: re.Match) -> str:
        num = m.group(1)
        act = m.group(2).upper()
        expansions = {
            'IPC': 'Indian Penal Code', 'BNS': 'Bharatiya Nyaya Sanhita',
            'BNSS': 'Bharatiya Nyaya Sanhita', 'CRPC': 'Code of Criminal Procedure',
            'CPC': 'Code of Civil Procedure',
        }
        return f"Section {num} {expansions.get(act, act)}"
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


# ── Pipeline ──────────────────────────────────────────────────────────────────

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

    def query(self, user_query: str, n_results: Optional[int] = None) -> LegalAnswer:
        """Run the full pipeline. Never raises — errors return a LegalAnswer with warning."""
        try:
            return self._run(user_query, n_results or self.n_final)
        except Exception as e:
            import traceback; traceback.print_exc()
            return LegalAnswer(
                answer_text       = "An internal error occurred. Please try again.",
                sources_consulted = 0,
                synthesis_note    = "Pipeline error.",
                grounding_warning = str(e),
            )

    def _run(self, user_query: str, n_final: int) -> LegalAnswer:

        # Step 0: Preprocess
        expanded = preprocess_query(user_query)

        # Step 1: Rewrite
        all_queries = (
            query_rewriter.rewrite(expanded)
            if self.enable_rewriting else [expanded]
        )

        # Step 2: Multi-query hybrid search
        all_results = [
            hybrid_searcher.search(q, n_results=self.n_retrieve)
            for q in all_queries
        ]

        # Step 3: Deduplicate
        merged = deduplicate_chunks(all_results)

        if not merged:
            return LegalAnswer(
                answer_text=(
                    "The retrieved legal sections do not contain sufficient information "
                    "to answer this question. Please consult a qualified legal professional."
                ),
                sources_consulted  = 0,
                synthesis_note     = "No sources retrieved.",
                grounding_warning  = "No chunks matched the query.",
                rewritten_queries  = all_queries,
                reranker_used      = False,
            )

        # Step 4: Rerank
        reranker_input = merged[:self.n_reranker_input]
        reranker_used  = False

        if self.enable_reranking:
            final_chunks, reranker_used = reranker.rerank(
                query=expanded, chunks=reranker_input, top_n=n_final,
            )
        else:
            final_chunks = reranker_input[:n_final]
            for c in final_chunks:
                c["rerank_score"] = None

        # Step 5: Section-number safety fallback
        section_match = re.search(r'\bSection\s+(\d+[A-Z]?)\b', expanded, re.IGNORECASE)
        if section_match:
            target = section_match.group(1)
            if not any(c.get("section", "") == target for c in final_chunks):
                from rag.bm25_retriever import bm25_retriever
                extra = bm25_retriever.search(expanded, n_results=3)
                for e in extra:
                    if e.get("section", "") == target:
                        e.update({"rerank_score": None, "hybrid_score": 0.005})
                        final_chunks.append(e)
                final_chunks = final_chunks[:n_final]

        # Step 6: Synthesize
        prompt     = build_synthesis_prompt(expanded, final_chunks)
        raw_answer = llm.generate(
            prompt        = prompt,
            system_prompt = SYNTHESIS_SYSTEM_PROMPT,
            temperature   = self.temperature,
            max_tokens    = self.max_tokens,
        )

        return synthesize(
            query             = user_query,
            chunks            = final_chunks,
            llm_answer        = raw_answer,
            rewritten_queries = all_queries,
            reranker_used     = reranker_used,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
rag_pipeline = RAGPipeline()
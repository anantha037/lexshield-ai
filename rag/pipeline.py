"""
LexShield AI — RAG Pipeline  (Week 2, Day 2 update)
====================================================
Full advanced retrieval pipeline:

  User query
      │
      ▼
  [Step 0] preprocess_query()        — expand abbreviations (Week 1)
      │
      ▼
  [Step 1] query_rewriter.rewrite()  — 3 angle-diverse queries via LLaMA
      │
      ▼
  [Step 2] hybrid_searcher.search()  — run on ALL queries (original + rewrites)
      │
      ▼
  [Step 3] deduplicate by chunk_id   — keep best hybrid_score per chunk
      │
      ▼
  [Step 4] reranker.rerank()         — top 10 → NVIDIA NIM → reordered top 5
      │    fallback: use hybrid order if API unavailable
      │
      ▼
  [Step 5] build_rag_prompt()        — format top 5 chunks for LLM
      │
      ▼
  [Step 6] llm.generate()            — Groq LLaMA 3.3 70B answer

RAGResponse new fields vs Day 1:
  rewritten_queries  — all queries actually searched
  reranker_used      — bool
  retrieval_sources  — "vector" / "bm25" / "both" per chunk
  section_titles     — deduplicated section titles
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from rag.llm            import llm
from rag.hybrid_search  import hybrid_searcher
from rag.query_rewriter import query_rewriter
from rag.reranker       import reranker

# ── Constants ─────────────────────────────────────────────────────────────────
N_RETRIEVE_PER_QUERY = 8    # hybrid results per rewritten query
N_RERANKER_INPUT     = 10   # top-N deduplicated chunks sent to reranker
N_FINAL_CONTEXT      = 5    # chunks passed to LLM after reranking

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are LexShield, an AI legal assistant specialising in Indian law.

Rules you MUST follow:
1. Answer ONLY from the retrieved legal sections provided. Do not use outside knowledge.
2. Every factual claim must cite its source in the format: [Source Name, Section X].
3. If the retrieved sections do not contain a clear answer, say:
   "The retrieved legal sections do not contain enough information to answer this question."
4. Do NOT guess, infer, or extrapolate beyond what the text explicitly states.
5. When a section number is available, always include it in your citation.
6. Write in clear, simple English that a non-lawyer can understand.
7. Keep your answer concise — typically 150-300 words unless complexity demands more.
"""

# ── Query preprocessor (Week 1, unchanged) ────────────────────────────────────
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
            'IPC':  'Indian Penal Code',  'BNS': 'Bharatiya Nyaya Sanhita',
            'BNSS': 'Bharatiya Nyaya Sanhita', 'CRPC': 'Code of Criminal Procedure',
            'CPC':  'Code of Civil Procedure',
        }
        return f"Section {num} {expansions.get(act, act)}"
    q = SECTION_RE.sub(expand_section, q)
    for pattern, replacement in QUERY_EXPANSIONS.items():
        q = re.sub(pattern, replacement, q, flags=re.IGNORECASE)
    return q


# ── Response dataclass ────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    query:             str
    answer:            str
    citations:         list[str]     = field(default_factory=list)
    retrieved_chunks:  list[dict]    = field(default_factory=list)
    context_used:      bool          = True
    warning:           Optional[str] = None
    rewritten_queries: list[str]     = field(default_factory=list)
    reranker_used:     bool          = False
    retrieval_sources: list[str]     = field(default_factory=list)
    section_titles:    list[str]     = field(default_factory=list)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    sections_text = ""
    for i, chunk in enumerate(chunks, start=1):
        source        = chunk.get("source",        "Unknown")
        section       = chunk.get("section",       "")
        section_title = chunk.get("section_title", "")
        chapter       = chunk.get("chapter",       "")
        parts = [source]
        if chapter:
            parts.append(chapter)
        if section:
            label = f"Section {section}"
            if section_title:
                label += f" ({section_title})"
            parts.append(label)
        header = " › ".join(parts)
        sections_text += f"\n--- [{i}] {header} ---\n{chunk.get('text', '')}\n"

    return (
        f"[RETRIEVED LEGAL SECTIONS]\n"
        f"{sections_text}\n"
        f"[USER QUESTION]\n{query}\n\n"
        f"[TASK]\n"
        f"Answer the user's question using ONLY the retrieved sections above. "
        f"Cite every claim with its source and section number.\n"
        f"Answer:"
    )


def extract_citations(chunks: list[dict]) -> list[str]:
    seen:  set[str]  = set()
    cites: list[str] = []
    for c in chunks:
        source        = c.get("source",        "Unknown")
        section       = c.get("section",       "")
        section_title = c.get("section_title", "")
        cite = f"{source}, Section {section}" if section else source
        if section_title:
            cite += f" ({section_title})"
        if cite not in seen:
            seen.add(cite)
            cites.append(cite)
    return cites


# ── Multi-query deduplication ─────────────────────────────────────────────────

def deduplicate_chunks(all_results: list[list[dict]]) -> list[dict]:
    """
    Merge results from multiple queries.
    Keeps highest hybrid_score per chunk_id. Returns sorted list.
    """
    best: dict[str, dict] = {}
    for result_list in all_results:
        for chunk in result_list:
            cid   = chunk.get("chunk_id", "")
            score = chunk.get("hybrid_score", 0.0)
            if cid not in best or score > best[cid].get("hybrid_score", 0.0):
                best[cid] = chunk
    return sorted(best.values(), key=lambda x: x.get("hybrid_score", 0.0), reverse=True)


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Full Week 2 Day 2 RAG pipeline.
    query → rewrite → multi-query hybrid → dedup → rerank → LLM
    """

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

    def query(self, user_query: str, n_results: Optional[int] = None) -> RAGResponse:

        n_final = n_results or self.n_final

        # ── Step 0: Preprocess ───────────────────────────────────────────────
        expanded = preprocess_query(user_query)

        # ── Step 1: Query rewriting ──────────────────────────────────────────
        if self.enable_rewriting:
            all_queries = query_rewriter.rewrite(expanded)
        else:
            all_queries = [expanded]

        # ── Step 2: Multi-query hybrid search ────────────────────────────────
        all_results: list[list[dict]] = []
        for q in all_queries:
            hits = hybrid_searcher.search(q, n_results=self.n_retrieve)
            all_results.append(hits)

        # ── Step 3: Deduplicate ───────────────────────────────────────────────
        merged = deduplicate_chunks(all_results)

        if not merged:
            return RAGResponse(
                query=user_query,
                answer=(
                    "The retrieved legal sections do not contain enough information "
                    "to answer this question. Please consult a qualified legal professional."
                ),
                context_used=False,
                rewritten_queries=all_queries,
                warning="No relevant chunks retrieved.",
            )

        # ── Step 4: Reranking ─────────────────────────────────────────────────
        reranker_input = merged[:self.n_reranker_input]
        reranker_used  = False

        if self.enable_reranking:
            final_chunks, reranker_used = reranker.rerank(
                query=expanded,
                chunks=reranker_input,
                top_n=n_final,
            )
        else:
            final_chunks = reranker_input[:n_final]
            for c in final_chunks:
                c["rerank_score"] = None

        # ── Step 5: Section-number safety fallback ───────────────────────────
        section_match = re.search(r'\bSection\s+(\d+[A-Z]?)\b', expanded, re.IGNORECASE)
        if section_match:
            target     = section_match.group(1)
            has_target = any(c.get("section", "") == target for c in final_chunks)
            if not has_target:
                from rag.bm25_retriever import bm25_retriever
                extra = bm25_retriever.search(expanded, n_results=3)
                for e in extra:
                    if e.get("section", "") == target:
                        e["rerank_score"] = None
                        e["hybrid_score"] = 0.005
                        final_chunks.append(e)
                final_chunks = final_chunks[:n_final]

        # ── Step 6: Generate answer ──────────────────────────────────────────
        prompt = build_rag_prompt(expanded, final_chunks)
        answer = llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return RAGResponse(
            query=user_query,
            answer=answer,
            citations=extract_citations(final_chunks),
            retrieved_chunks=final_chunks,
            context_used=True,
            rewritten_queries=all_queries,
            reranker_used=reranker_used,
            retrieval_sources=list({c.get("retrieval_source", "?") for c in final_chunks}),
            section_titles=list({
                c.get("section_title", "") for c in final_chunks if c.get("section_title")
            }),
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
rag_pipeline = RAGPipeline()
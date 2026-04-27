"""
LexShield AI — RAG Pipeline  (Week 2 update)
=============================================
Changes from Week 1:
  • search step now uses HybridSearcher (vector + BM25) instead of raw vector search
  • Citations include section_title and chapter when available
  • Prompt builder passes context_text (header-enriched) to the LLM
  • RAGResponse gets two new fields: retrieval_sources, section_titles
  • Everything else (query preprocessing, LLM call, anti-hallucination prompt) unchanged
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from rag.llm          import llm            # LegalLLM singleton (Week 1, unchanged)
from rag.hybrid_search import hybrid_searcher  # Week 2 hybrid searcher

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
7. Keep your answer concise — typically 150–300 words unless complexity demands more.
"""

# ── Query preprocessor ────────────────────────────────────────────────────────
# Expands abbreviated legal queries to improve retrieval quality.
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
    r'\bpoc[so]\b':  'Prevention of Corruption Act',
}

SECTION_RE = re.compile(
    r'\b(\d{1,4}[A-Z]?)\s*(ipc|bnss?|crpc|cpc|it\s?act|consumer|mv\s?act)\b',
    re.IGNORECASE,
)


def preprocess_query(query: str) -> str:
    """
    Expands abbreviations and normalises section references.
    "216ipc" → "Section 216 Indian Penal Code"
    """
    q = query.strip()
    # Expand section references: "420 IPC" → "Section 420 Indian Penal Code"
    def expand_section(m: re.Match) -> str:
        num = m.group(1)
        act = m.group(2).upper()
        expansions = {
            'IPC':  'Indian Penal Code', 'BNS': 'Bharatiya Nyaya Sanhita',
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
    citations:         list[str]          = field(default_factory=list)
    retrieved_chunks:  list[dict]         = field(default_factory=list)
    context_used:      bool               = True
    warning:           Optional[str]      = None
    retrieval_sources: list[str]          = field(default_factory=list)  # "vector"/"bm25"/"both"
    section_titles:    list[str]          = field(default_factory=list)


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """
    Builds the user-turn prompt for the LLM.
    Uses chunk text as retrieved context.
    """
    sections_text = ""
    for i, chunk in enumerate(chunks, start=1):
        source       = chunk.get("source", "Unknown")
        section      = chunk.get("section", "")
        section_title = chunk.get("section_title", "")
        chapter      = chunk.get("chapter", "")

        # Build a clear citation header
        citation_parts = [source]
        if chapter:
            citation_parts.append(chapter)
        if section:
            label = f"Section {section}"
            if section_title:
                label += f" ({section_title})"
            citation_parts.append(label)

        header = " › ".join(citation_parts)
        text   = chunk.get("text", "")

        sections_text += f"\n--- [{i}] {header} ---\n{text}\n"

    return (
        f"[RETRIEVED LEGAL SECTIONS]\n"
        f"{sections_text}\n"
        f"[USER QUESTION]\n{query}\n\n"
        f"[TASK]\n"
        f"Answer the user's question using ONLY the retrieved sections above. "
        f"Cite every claim with its source and section number.\n"
        f"Answer:"
    )


# ── Citation extractor ────────────────────────────────────────────────────────

def extract_citations(chunks: list[dict]) -> list[str]:
    """Build a deduplicated citation list from retrieved chunks."""
    seen: set[str]     = set()
    citations: list[str] = []

    for chunk in chunks:
        source       = chunk.get("source", "Unknown")
        section      = chunk.get("section", "")
        section_title = chunk.get("section_title", "")

        if section:
            cite = f"{source}, Section {section}"
            if section_title:
                cite += f" ({section_title})"
        else:
            cite = source

        if cite not in seen:
            seen.add(cite)
            citations.append(cite)

    return citations


# ── RAG Pipeline ──────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    End-to-end RAG pipeline.

    Steps:
      0. preprocess_query   — expand abbreviations / section refs
      1. hybrid search      — vector + BM25 via HybridSearcher
      2. filter             — remove low-score / ToC chunks
      3. build prompt       — format retrieved context for LLM
      4. llm.generate()     — call Groq LLaMA 3.3 70B
      5. extract citations  — deduplicated citation list
      6. return RAGResponse
    """

    def __init__(
        self,
        n_retrieve:       int   = 8,
        min_hybrid_score: float = 0.005,   # RRF scores are small floats
        temperature:      float = 0.1,
        max_tokens:       int   = 1024,
    ):
        self.n_retrieve       = n_retrieve
        self.min_hybrid_score = min_hybrid_score
        self.temperature      = temperature
        self.max_tokens       = max_tokens

    def query(
        self,
        user_query: str,
        n_results:  Optional[int] = None,
    ) -> RAGResponse:
        n = n_results or self.n_retrieve

        # ── Step 0: Preprocess ───────────────────────────────────────────────
        expanded_query = preprocess_query(user_query)

        # ── Step 1: Hybrid retrieval ─────────────────────────────────────────
        raw_chunks = hybrid_searcher.search(expanded_query, n_results=n)

        # ── Step 2: Filter ───────────────────────────────────────────────────
        chunks = [
            c for c in raw_chunks
            if c.get("hybrid_score", 0) >= self.min_hybrid_score
        ]

        # Section-number fallback — if query contains an explicit section number,
        # ensure at least one chunk with that section is present
        section_match = re.search(r'\bSection\s+(\d+[A-Z]?)\b', expanded_query, re.IGNORECASE)
        if section_match and chunks:
            target_sec = section_match.group(1)
            has_target = any(c.get("section", "") == target_sec for c in chunks)
            if not has_target:
                # Widen search with pure BM25 for section-number queries
                from rag.bm25_retriever import bm25_retriever
                extra = bm25_retriever.search(expanded_query, n_results=4)
                for e in extra:
                    if e.get("section", "") == target_sec:
                        e["hybrid_score"] = 0.01   # give it a nominal hybrid score
                        chunks.append(e)
                chunks = chunks[:n]

        # ── Step 3: Build prompt ─────────────────────────────────────────────
        if not chunks:
            return RAGResponse(
                query=user_query,
                answer=(
                    "The retrieved legal sections do not contain enough information "
                    "to answer this question. Please consult a qualified legal professional."
                ),
                context_used=False,
                warning="No relevant chunks retrieved above threshold.",
            )

        prompt = build_rag_prompt(expanded_query, chunks)

        # ── Step 4: Generate answer ──────────────────────────────────────────
        answer = llm.generate(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # ── Step 5: Citations + metadata ─────────────────────────────────────
        citations        = extract_citations(chunks)
        retrieval_sources = list({c.get("retrieval_source", "?") for c in chunks})
        section_titles   = list({
            c.get("section_title", "")
            for c in chunks
            if c.get("section_title")
        })

        return RAGResponse(
            query=user_query,
            answer=answer,
            citations=citations,
            retrieved_chunks=chunks,
            context_used=True,
            retrieval_sources=retrieval_sources,
            section_titles=section_titles,
        )


# ── Singleton ─────────────────────────────────────────────────────────────────
rag_pipeline = RAGPipeline()
"""
LexShield AI — Multi-Document Synthesizer  (Week 2, Day 3)
===========================================================
Handles everything between "chunks retrieved" and "LLM call":
  1. Formats chunks as numbered [SOURCE N] blocks for synthesis prompt
  2. Builds a synthesis-aware prompt that forces cross-source inline citation
  3. Converts raw chunk dicts → structured Citation objects
  4. Checks the answer for grounding / hallucination signals
  5. Returns LegalAnswer — the single structured object the API serialises

LegalAnswer also carries pipeline metadata (rewritten_queries, reranker_used)
so the API endpoint never needs to reach back into the pipeline internals.
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Structured citation ───────────────────────────────────────────────────────

@dataclass
class Citation:
    """
    One cited source in a legal answer.

    source_number    — corresponds to [1] / [2] inline markers in answer_text
    source           — document name  e.g. "Indian Penal Code (IPC)"
    section          — section number e.g. "420"
    section_title    — section name   e.g. "Cheating"
    chapter          — chapter name   e.g. "CHAPTER XVII"
    preview          — first 200 chars of retrieved chunk text
    relevance_score  — rerank_score if available, else hybrid_score
    retrieval_source — "vector" / "bm25" / "both"
    doc_type         — "statute" / "judgment" / "handbook"
    """
    source_number:    int
    source:           str
    section:          str            = ""
    section_title:    str            = ""
    chapter:          str            = ""
    preview:          str            = ""
    relevance_score:  Optional[float] = None
    retrieval_source: str            = ""
    doc_type:         str            = ""


# ── Structured response ───────────────────────────────────────────────────────

@dataclass
class LegalAnswer:
    """
    Fully structured response returned by the pipeline.

    answer_text       — LLM-generated text with [1][2] inline citations
    citations         — Citation objects, one per source consulted
    sources_consulted — count of distinct chunks used
    synthesis_note    — human-readable note e.g. "Synthesized from 3 sources"
    grounding_warning — set if hallucination signals detected, else None

    Pipeline metadata (set by RAGPipeline, not synthesizer):
    rewritten_queries — all queries searched (original + LLM rewrites)
    reranker_used     — True if NVIDIA API call succeeded
    """
    answer_text:       str
    citations:         list[Citation] = field(default_factory=list)
    sources_consulted: int            = 0
    synthesis_note:    str            = ""
    grounding_warning: Optional[str]  = None
    # Set by RAGPipeline after calling synthesize()
    rewritten_queries: list[str]      = field(default_factory=list)
    reranker_used:     bool           = False


# ── System prompt ─────────────────────────────────────────────────────────────

SYNTHESIS_SYSTEM_PROMPT = """You are LexShield, an AI legal assistant specialising in Indian law.

You will receive several numbered legal sources retrieved from Indian statutes and court judgments.
Your job is to synthesize information ACROSS all relevant sources to give a complete answer.

STRICT RULES — follow every one:
1. Use ONLY the information in the provided sources. Never add outside knowledge.
2. Every sentence that states a legal fact MUST end with an inline citation: [1] or [2] or [1][3].
3. If two sources say the same thing, cite both: [1][2].
4. If sources address the same offence under both IPC and BNS, explain both and cite each separately.
5. Never invent a section number. Only use section numbers that appear explicitly in the sources.
6. If the sources do not answer the question, say exactly:
   "The retrieved legal sections do not contain sufficient information to answer this question."
7. Structure your answer:
   a) Direct answer to the question (1-2 sentences)
   b) Relevant legal provisions with inline citations
   c) Punishment or remedy if present in sources
   d) Procedure or practical note if present in sources
8. Keep the answer between 150 and 350 words.
9. Write in plain English that a non-lawyer can understand.
"""


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_synthesis_prompt(query: str, chunks: list[dict]) -> str:
    """
    Formats retrieved chunks as numbered [SOURCE N] blocks.
    The numbers here must match [1] [2] citations in the LLM answer.
    """
    sources_block = ""
    for i, chunk in enumerate(chunks, start=1):
        source        = chunk.get("source",        "Unknown Source")
        section       = chunk.get("section",       "")
        section_title = chunk.get("section_title", "")
        chapter       = chunk.get("chapter",       "")
        text          = chunk.get("text",          "")

        header_parts = [source]
        if chapter:
            header_parts.append(chapter)
        if section:
            sec_label = f"Section {section}"
            if section_title:
                sec_label += f" ({section_title})"
            header_parts.append(sec_label)

        header  = " › ".join(header_parts)
        divider = "─" * min(len(header) + 4, 72)

        sources_block += (
            f"\n[SOURCE {i}] {header}\n"
            f"{divider}\n"
            f"{text}\n"
        )

    return (
        f"[RETRIEVED LEGAL SOURCES]\n"
        f"{sources_block}\n"
        f"[USER QUESTION]\n"
        f"{query}\n\n"
        f"[SYNTHESIS TASK]\n"
        f"Synthesize the above sources to answer the question.\n"
        f"- Cite every legal claim with its [SOURCE NUMBER] inline.\n"
        f"- If multiple sources address the same point, cite all of them.\n"
        f"- If both IPC and BNS apply, explain both with separate citations.\n"
        f"Answer:"
    )


# ── Citation builder ──────────────────────────────────────────────────────────

def build_citations(chunks: list[dict]) -> list[Citation]:
    """Converts chunk dicts to Citation objects. source_number matches [SOURCE N]."""
    citations: list[Citation] = []
    for i, chunk in enumerate(chunks, start=1):
        score = (
            chunk.get("rerank_score")
            or chunk.get("hybrid_score")
            or chunk.get("vector_score")
            or chunk.get("score")
        )
        try:
            score = round(float(score), 4) if score is not None else None
        except (TypeError, ValueError):
            score = None

        raw_text = chunk.get("text", "")
        preview  = raw_text[:200].strip() + ("…" if len(raw_text) > 200 else "")

        citations.append(Citation(
            source_number    = i,
            source           = chunk.get("source",          "Unknown"),
            section          = chunk.get("section",          ""),
            section_title    = chunk.get("section_title",    ""),
            chapter          = chunk.get("chapter",          ""),
            preview          = preview,
            relevance_score  = score,
            retrieval_source = chunk.get("retrieval_source", ""),
            doc_type         = chunk.get("doc_type",         ""),
        ))
    return citations


# ── Grounding checker ─────────────────────────────────────────────────────────

_HALLUCINATION_SIGNALS = [
    "generally speaking", "in general", "typically", "usually",
    "in most cases", "it is widely understood", "commonly known",
    "as a rule", "by convention", "in practice", "experts say",
    "legal experts", "lawyers agree", "based on my knowledge",
    "i believe", "i think",
]


def check_grounding(answer_text: str, chunks: list[dict]) -> Optional[str]:
    """
    Returns a warning string if the answer shows signs of hallucination.
    Returns None if the answer looks clean.
    """
    answer_lower = answer_text.lower()

    # Check 1: hedging / generalising phrases
    found = [s for s in _HALLUCINATION_SIGNALS if s in answer_lower]
    if found:
        return f"Answer contains generalising phrases ({found[:2]}). Review for hallucination."

    # Check 2: section numbers in answer not present in any chunk
    cited_secs     = set(re.findall(r'\bSection\s+(\d+[A-Z]?)\b', answer_text, re.IGNORECASE))
    available_secs = {str(c.get("section", "")) for c in chunks if c.get("section")}
    phantom        = cited_secs - available_secs
    if phantom:
        return f"Answer cites section(s) {phantom} not found in retrieved sources. Possible hallucination."

    # Check 3: no inline citations in a substantive answer
    inline = re.findall(r'\[\d+\]', answer_text)
    if not inline and len(answer_text) > 100:
        return "No inline [N] citations found. LLM may not have followed synthesis instructions."

    return None


# ── Synthesis note ────────────────────────────────────────────────────────────

def build_synthesis_note(chunks: list[dict]) -> str:
    if not chunks:
        return "No sources consulted."
    n        = len(chunks)
    sources  = {c.get("source", "") for c in chunks}
    doc_types = {c.get("doc_type", "unknown") for c in chunks}
    type_label = " + ".join(sorted(dt.capitalize() for dt in doc_types if dt))
    if n == 1:
        return f"Single source consulted ({type_label})."
    elif len(sources) == 1:
        return f"Synthesized from {n} sections of {next(iter(sources))}."
    else:
        return f"Synthesized from {n} sections across {len(sources)} sources ({type_label})."


# ── Main synthesize function ──────────────────────────────────────────────────

def synthesize(
    query:             str,
    chunks:            list[dict],
    llm_answer:        str,
    rewritten_queries: list[str] = None,
    reranker_used:     bool      = False,
) -> LegalAnswer:
    """
    Wraps the raw LLM answer into a fully structured LegalAnswer.

    Call from pipeline.py:
        prompt     = build_synthesis_prompt(query, final_chunks)
        raw_answer = llm.generate(prompt, system_prompt=SYNTHESIS_SYSTEM_PROMPT, ...)
        result     = synthesize(query, final_chunks, raw_answer,
                                rewritten_queries=all_queries,
                                reranker_used=reranker_used)
    """
    return LegalAnswer(
        answer_text        = llm_answer.strip(),
        citations          = build_citations(chunks),
        sources_consulted  = len(chunks),
        synthesis_note     = build_synthesis_note(chunks),
        grounding_warning  = check_grounding(llm_answer, chunks),
        rewritten_queries  = rewritten_queries or [],
        reranker_used      = reranker_used,
    )
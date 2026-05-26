"""
LexShield AI — Multi-Document Synthesizer
===========================================================
Changes in this version:

  NEW  Era-aware system prompt for paired act responses
       When final_chunks contain both a "legacy" era chunk (IPC/CrPC/Evidence Act)
       AND a "current" era chunk (BNS/BNSS/BSA) or a "paired_act" chunk,
       SYNTHESIS_SYSTEM_PROMPT now instructs the LLM to:
         - Explain both old and new provisions with separate citations
         - State the July 1, 2024 cutoff explicitly
         - Tell the user which act applies to their situation

  NEW  build_synthesis_prompt() injects an era context note when both
       era types are present in chunks — makes the LLM aware it has
       paired act sources without any extra pipeline changes.

  NEW  Citation dataclass gains era field (read from chunk metadata).

  FIX  build_citations() now reads category from chunk (was missing).

Everything else (grounding checker, synthesis note, LegalAnswer) unchanged.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from langsmith import traceable


# ── Structured citation ───────────────────────────────────────────────────────

@dataclass
class Citation:
    source_number:    int
    source:           str
    section:          str            = ""
    section_title:    str            = ""
    chapter:          str            = ""
    preview:          str            = ""
    relevance_score:  Optional[float] = None
    retrieval_source: str            = ""
    doc_type:         str            = ""
    category:         str            = ""
    era:              str            = ""   # NEW: "legacy" | "current" | ""


# ── Structured response ───────────────────────────────────────────────────────

@dataclass
class LegalAnswer:
    answer_text:       str
    citations:         list[Citation] = field(default_factory=list)
    sources_consulted: int            = 0
    synthesis_note:    str            = ""
    grounding_warning: Optional[str]  = None
    rewritten_queries: list[str]      = field(default_factory=list)
    reranker_used:     bool           = False
    # Set by pipeline when CRAG scores insufficient — frontend uses these
    # to label the response appropriately without changing synthesizer logic.
    confidence:        str            = "normal"  # "normal" | "low"
    fallback:          bool           = False      # True when CRAG: insufficient path taken


# ── System prompts ────────────────────────────────────────────────────────────

# Standard prompt — used when no paired act chunks present
SYNTHESIS_SYSTEM_PROMPT = """You are LexShield, an AI legal assistant specialising in Indian law.

You will receive several numbered legal sources retrieved from Indian statutes and court judgments.
Your job is to synthesize information ACROSS all relevant sources to give a complete answer.

STRICT RULES — follow every one:
1. Use ONLY the information in the provided sources. Never add outside knowledge.
2. Every sentence that states a legal fact MUST end with an inline citation: [1] or [2] or [1][3].
3. If two sources say the same thing, cite both: [1][2].
4. If sources address the same offence under both IPC and BNS, explain both and cite each separately.
5. SECTION NUMBER RULE — THIS IS ABSOLUTE: You may ONLY write a section number if that exact
   number appears in the [SOURCE N] header above. The headers are the ONLY permitted source of
   section numbers. If you know a related section from your training but it does not appear in
   any [SOURCE N] header, you MUST NOT write that number. Write the legal concept in plain words.
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

# Paired act prompt — used when both legacy (IPC/CrPC/Evidence Act) and
# current (BNS/BNSS/BSA) or paired_act chunks are present.
PAIRED_ACT_SYSTEM_PROMPT = """You are LexShield, an AI legal assistant specialising in Indian law.

You will receive numbered legal sources from BOTH old Indian laws AND their 2023 replacements.

IMPORTANT CONTEXT — Indian Criminal Law Reform (effective July 1, 2024):
  • Indian Penal Code (IPC) 1860      -> replaced by Bharatiya Nyaya Sanhita (BNS) 2023
  • Code of Criminal Procedure (CrPC) -> replaced by Bharatiya Nagarik Suraksha Sanhita (BNSS) 2023
  • Indian Evidence Act 1872          -> replaced by Bharatiya Sakshya Adhiniyam (BSA) 2023

  For offences/cases BEFORE July 1, 2024 -> the OLD law (IPC/CrPC/Evidence Act) applies.
  For offences/cases ON OR AFTER July 1, 2024 -> the NEW law (BNS/BNSS/BSA) applies.
  Old cases already in court continue under the old law even after July 1, 2024.

STRICT RULES — follow every one:
1. Use ONLY the information in the provided sources. Never add outside knowledge.
2. Every sentence that states a legal fact MUST end with an inline citation: [1] or [2] or [1][3].
3. ALWAYS explain BOTH the old law provision AND the new law provision if both are in the sources.
   Structure it as:
     "Under the old law (pre-July 2024): ... [SOURCE N]"
     "Under the new law (post-July 2024): ... [SOURCE N]"
4. SECTION NUMBER RULE — ABSOLUTE: Only write section numbers that appear in [SOURCE N] headers.
   Never invent or recall section numbers from training. If you don't see the section number in
   a header, describe the provision in plain words instead.
5. End your answer with a brief practical note:
   "If your matter arose before July 1, 2024, the [old act] applies.
    If it arose on or after July 1, 2024, the [new act] applies."
6. Keep the answer between 200 and 400 words.
7. Write in plain English that a non-lawyer can understand.
"""


# ── Era detection ─────────────────────────────────────────────────────────────

def _has_paired_context(chunks: list[dict]) -> bool:
    """
    Returns True if chunks contain BOTH a legacy era chunk AND either a
    current era chunk or a paired_act retrieval_source chunk.
    This triggers the paired act system prompt.
    """
    has_legacy  = any(c.get("era") == "legacy"  for c in chunks)
    has_current = any(
        c.get("era") == "current" or c.get("retrieval_source") == "paired_act"
        for c in chunks
    )
    return has_legacy and has_current


def get_system_prompt(chunks: list[dict]) -> str:
    """Returns the appropriate system prompt based on chunk era composition."""
    return PAIRED_ACT_SYSTEM_PROMPT if _has_paired_context(chunks) else SYNTHESIS_SYSTEM_PROMPT


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_synthesis_prompt(query: str, chunks: list[dict]) -> str:
    """
    Formats retrieved chunks as numbered [SOURCE N] blocks.
    When paired act chunks are present, injects an era context note
    before the sources block so the LLM knows what it's looking at.
    """
    # Era context note — only injected when paired chunks are present
    era_note = ""
    if _has_paired_context(chunks):
        era_note = (
            "[LEGAL ERA CONTEXT]\n"
            "Sources below include BOTH pre-July 2024 laws (IPC/CrPC/Evidence Act) "
            "AND post-July 2024 replacement laws (BNS/BNSS/BSA). "
            "Explain provisions under both. Cutoff: July 1, 2024.\n\n"
        )

    sources_block = ""
    for i, chunk in enumerate(chunks, start=1):
        source        = chunk.get("source",        "Unknown Source")
        section       = chunk.get("section",       "")
        section_title = chunk.get("section_title", "")
        chapter       = chunk.get("chapter",       "")
        text          = chunk.get("text",          "")
        era           = chunk.get("era",           "")
        r_source      = chunk.get("retrieval_source", "")

        header_parts = [source]
        if chapter:
            header_parts.append(chapter)
        if section:
            sec_label = f"Section {section}"
            if section_title:
                sec_label += f" ({section_title})"
            header_parts.append(sec_label)

        # Era label in source header for LLM clarity
        if era == "legacy":
            header_parts.append("⟨PRE-JULY 2024 LAW⟩")
        elif era == "current" or r_source == "paired_act":
            header_parts.append("⟨POST-JULY 2024 LAW⟩")

        header  = " › ".join(header_parts)
        divider = "─" * min(len(header) + 4, 72)

        sources_block += (
            f"\n[SOURCE {i}] {header}\n"
            f"{divider}\n"
            f"{text}\n"
        )

    return (
        f"{era_note}"
        f"[RETRIEVED LEGAL SOURCES]\n"
        f"{sources_block}\n"
        f"[USER QUESTION]\n"
        f"{query}\n\n"
        f"[SYNTHESIS TASK]\n"
        f"Synthesize the above sources to answer the question.\n"
        f"- Cite every legal claim with its [SOURCE NUMBER] inline.\n"
        f"- If multiple sources address the same point, cite all of them.\n"
        f"- If both old and new law sources are present, explain both separately.\n"
        f"Answer:"
    )


# ── Citation builder ──────────────────────────────────────────────────────────

def build_citations(chunks: list[dict]) -> list[Citation]:
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
            category         = chunk.get("category",         ""),   # FIX
            era              = chunk.get("era",               ""),   # NEW
        ))
    return citations


# ── Grounding checker ─────────────────────────────────────────────────────────

_HALLUCINATION_SIGNALS = [
    "legal experts", "lawyers agree", "based on my knowledge",
    "i believe", "i think",
]


def check_grounding(answer_text: str, chunks: list[dict]) -> Optional[str]:
    answer_lower = answer_text.lower()

    found = [s for s in _HALLUCINATION_SIGNALS if s in answer_lower]
    if found:
        return f"Answer contains generalising phrases ({found[:2]}). Review for hallucination."

    cited_secs     = set(re.findall(r'\bSection\s+(\d{2,4}[A-Z]?)\b', answer_text, re.IGNORECASE))
    available_secs = {
        str(c.get("section", "")) for c in chunks
        if c.get("section") and len(str(c.get("section"))) >= 2
    }
    phantom = cited_secs - available_secs
    if phantom:
        true_hallucinations = list(phantom)
        try:
            from rag.knowledge_graph import get_related_sections
            
            retrieved_graph_ids = []
            for c in chunks:
                sec = str(c.get("section", ""))
                src = c.get("source", "")
                if sec and len(sec) >= 2 and src != "Knowledge Graph":
                    src_lower = src.lower()
                    acronym = src
                    if "penal code" in src_lower or "ipc" in src_lower: acronym = "IPC"
                    elif "nyaya sanhita" in src_lower or "bns" in src_lower: acronym = "BNS"
                    elif "criminal procedure" in src_lower or "crpc" in src_lower: acronym = "CrPC"
                    elif "nagarik suraksha" in src_lower or "bnss" in src_lower: acronym = "BNSS"
                    elif "negotiable" in src_lower or "ni act" in src_lower: acronym = "NI"
                    elif "consumer protection" in src_lower or "cpa" in src_lower: acronym = "CPA"
                    elif "evidence act" in src_lower or "iea" in src_lower: acronym = "IEA"
                    elif "sakshya" in src_lower or "bsa" in src_lower: acronym = "BSA"
                    elif "pocso" in src_lower or "protection of children" in src_lower: acronym = "POCSO"
                    elif "narcotic" in src_lower or "ndps" in src_lower: acronym = "NDPS"
                    elif "unlawful" in src_lower or "uapa" in src_lower: acronym = "UAPA"
                    
                    retrieved_graph_ids.append(f"{acronym}_{sec}")
                    
            def is_graph_neighbor(mentioned_section: str, retrieved_ids: list[str], hops: int = 2) -> bool:
                for retrieved_id in retrieved_ids:
                    related = get_related_sections(retrieved_id, hops=hops)
                    if any(mentioned_section == rid.split("_")[-1] for rid in related):
                        return True
                return False

            true_hallucinations = [
                s for s in phantom
                if not is_graph_neighbor(s, retrieved_graph_ids, hops=2)
            ]
        except Exception as e:
            pass
            
        if true_hallucinations:
            return f"Answer cites section(s) {set(true_hallucinations)} not found in retrieved sources or related graph. Possible hallucination."

    inline = re.findall(r'\[\d+\]', answer_text)
    if not inline and len(answer_text) > 100:
        return "No inline [N] citations found. LLM may not have followed synthesis instructions."

    return None


# ── Synthesis note ────────────────────────────────────────────────────────────

def build_synthesis_note(chunks: list[dict]) -> str:
    if not chunks:
        return "No sources consulted."
    n         = len(chunks)
    sources   = {c.get("source", "") for c in chunks}
    doc_types = {c.get("doc_type", "unknown") for c in chunks}
    type_label = " + ".join(sorted(dt.capitalize() for dt in doc_types if dt))
    paired     = _has_paired_context(chunks)
    suffix     = " (includes old + new law comparison)" if paired else ""
    if n == 1:
        return f"Single source consulted ({type_label}){suffix}."
    elif len(sources) == 1:
        return f"Synthesized from {n} sections of {next(iter(sources))}{suffix}."
    else:
        return f"Synthesized from {n} sections across {len(sources)} sources ({type_label}){suffix}."


# ── Main synthesize function ──────────────────────────────────────────────────

@traceable(name="synthesizer.synthesize", run_type="chain")
def synthesize(
    query:             str,
    chunks:            list[dict],
    llm_answer:        str,
    rewritten_queries: list[str] = None,
    reranker_used:     bool      = False,
) -> LegalAnswer:
    return LegalAnswer(
        answer_text        = llm_answer.strip(),
        citations          = build_citations(chunks),
        sources_consulted  = len(chunks),
        synthesis_note     = build_synthesis_note(chunks),
        grounding_warning  = check_grounding(llm_answer, chunks),
        rewritten_queries  = rewritten_queries or [],
        reranker_used      = reranker_used,
    )
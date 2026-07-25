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

import logging

logger = logging.getLogger(__name__)


# ── Citation-tag mismatch validator ───────────────────────────────────────────
#
# Short and long forms of every act name that may appear in citation tags OR in
# the surrounding sentence text.  Both directions must be covered:
#   short -> long forms so "CrPC" matches "Code of Criminal Procedure"
#   long fragments -> short so the citation tag text hits the same bucket.
# Format: each tuple is a group of synonyms treated as the same act.
_CITATION_ACT_ALIASES: list[tuple[str, ...]] = [
    # IPC / BNS
    ("indian penal code", "ipc"),
    ("bharatiya nyaya sanhita", "bns"),
    # CrPC / BNSS
    ("code of criminal procedure", "crpc"),
    ("bharatiya nagarik suraksha sanhita", "bnss"),
    # Evidence Act / BSA
    ("indian evidence act", "evidence act", "iea"),
    ("bharatiya sakshya adhiniyam", "bsa"),
    # Others
    ("negotiable instruments act", "ni act"),
    ("information technology act", "it act"),
    ("code of civil procedure", "cpc"),
    ("protection of children from sexual offences", "pocso"),
    ("prevention of money laundering", "pmla"),
    ("narcotic drugs and psychotropic substances", "ndps"),
    ("insolvency and bankruptcy code", "ibc"),
    # System / generic — these never mismatch anything
    ("system",),
]

# Pre-build: token -> canonical group index, for O(1) group lookup
_ALIAS_INDEX: dict[str, int] = {}
for _gidx, _group in enumerate(_CITATION_ACT_ALIASES):
    for _alias in _group:
        _ALIAS_INDEX[_alias.lower()] = _gidx


def _citation_act_group(text: str) -> Optional[int]:
    """
    Return the alias-group index for the first recognised act name found in
    *text* (case-insensitive substring search through alias table).
    Returns None if no recognised act is found.
    """
    lower = text.lower()
    # Longest aliases first so "code of criminal procedure" beats "code"
    for alias, gidx in sorted(_ALIAS_INDEX.items(), key=lambda x: -len(x[0])):
        if alias in lower:
            return gidx
    return None


# Regex that matches a replaced citation tag like:
#   [Act Name, Section 154]   [System]   [Act Name (Short) 2023]
_REPLACED_TAG_RE = re.compile(
    r'\[([^\[\]]+?)\]'
)


# Caveat-sentence detector.  Sentences that describe the equivalence
# pairing itself (e.g. "has not been independently verified") are always
# sourced from the System note, never from either statute's real text.
# They must be forced to [System] regardless of which act names appear in
# the sentence body.  Anchored to the exact phrasing that synthesizer.py's
# unverified_note instruction produces so the two stay in sync.
_CAVEAT_RE = re.compile(
    r"\[STATUS:\s*UNVERIFIED\]"
    r"|not been independently verified"
    r"|section pairing has not been"
    r"|pairing is unverified",
    re.IGNORECASE,
)


def validate_citation_tags(text: str) -> str:
    """
    Post-generation mechanical validation pass.

    Splits *text* into sentence-like segments, then for each segment that
    ends with a replaced citation tag checks whether the act named in the
    tag matches an act name mentioned in the sentence body.  When they
    don't match, the tag is downgraded to a generic safe fallback
    ('[Statute]' or '[System]') rather than leaving a provably wrong
    specific citation in place.

    Rules:
    - Only checks tags whose text contains a recognised act alias.
      Unrecognised or purely numeric tags are left untouched.
    - '[System]' tags are never flagged as mismatched.
    - A sentence that mentions NO recognised act in its body is left
      untouched (no body context to compare against → no false positive).
    - Downgrades are logged at INFO level.
    """
    # Split on sentence boundaries: period/exclamation/question followed by
    # whitespace + capital, OR a newline.  We keep the delimiter attached
    # to the preceding segment via a lookahead so we don't lose it.
    segments = re.split(r'(?<=[.!?])(?=\s+[A-Z])|(?=\n)', text)

    result_parts: list[str] = []
    for seg in segments:
        # Find all citation tags in this segment
        tags = _REPLACED_TAG_RE.findall(seg)
        if not tags:
            result_parts.append(seg)
            continue

        # Determine act group of the BODY (segment text minus the tags)
        body = _REPLACED_TAG_RE.sub("", seg).strip()
        body_group = _citation_act_group(body)

        # ── Caveat-sentence override ──────────────────────────────────────────
        # A sentence describing the equivalence pairing itself ("not been
        # independently verified", "[STATUS: UNVERIFIED]", etc.) is always
        # sourced from the System note — never from either statute's text.
        # Force all specific-act tags to [System] regardless of body content.
        if _CAVEAT_RE.search(seg):
            new_seg = seg
            for tag_content in tags:
                tag_group = _citation_act_group(tag_content)
                if tag_group is None:
                    continue
                if tag_content.strip().lower() == "system":
                    continue
                old_tag = f"[{tag_content}]"
                logger.info(
                    "[CitationValidator] Caveat sentence — forcing %r -> '[System]' | "
                    "body preview: %r",
                    old_tag, body[:80],
                )
                new_seg = new_seg.replace(old_tag, "[System]", 1)
            result_parts.append(new_seg)
            continue   # skip normal act-match check for this segment
        # ─────────────────────────────────────────────────────────────────────

        # If body mentions no recognisable act → nothing to compare, leave as-is
        if body_group is None:
            result_parts.append(seg)
            continue

        # For each tag, check whether its act group matches the body group
        new_seg = seg
        for tag_content in tags:
            tag_group = _citation_act_group(tag_content)
            # Tags with no recognised act (e.g. bare numeric remnants) → skip
            if tag_group is None:
                continue
            # System group never mismatches
            if tag_content.strip().lower() == "system":
                continue
            # Mismatch detected
            if tag_group != body_group:
                # Decide fallback: 'System' if tag was already a system note,
                # otherwise generic 'Statute'
                fallback = "System" if "system" in tag_content.lower() else "Statute"
                old_tag  = f"[{tag_content}]"
                new_tag  = f"[{fallback}]"
                logger.info(
                    "[CitationValidator] Mismatch downgraded: %r -> %r | "
                    "sentence body mentions act-group %d, tag was act-group %d | "
                    "body preview: %r",
                    old_tag, new_tag, body_group, tag_group, body[:80],
                )
                new_seg = new_seg.replace(old_tag, new_tag, 1)
        result_parts.append(new_seg)

    return "".join(result_parts)


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
5. Always cite the specific section number when it appears in the source header 
   (e.g., write "Section 9 of the Wildlife Protection Act" not just "the Act"). 
   Section numbers are provided in the source headers — use them.
   Do not cite section numbers that do not appear in the provided sources.
6. If the sources do not answer the question, say exactly:
   "The retrieved legal sections do not contain sufficient information to answer this question."
7. Keep the answer between 150 and 350 words.
8. Write in plain English that a non-lawyer can understand.
9. If any source is marked [STATUS: UNVERIFIED], you MUST explicitly tell the user that this specific
   section pairing has not been independently verified and recommend they confirm it against the
   official bare act text. Do NOT state it as settled fact.
10. Before citing [N], re-check that SOURCE N's own text actually supports the specific sentence you
    are citing it for. Do not reuse a source number out of habit when discussing content from a
    different source block — cite that other block's actual number instead. Citing the wrong source
    number is a factual error, the same as citing the wrong section.
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
3. First check whether the old-law source and the new-law source actually
   describe the SAME legal provision or offence (same subject matter), or
   whether they only happen to share a section number by coincidence — read
   both source texts to judge this, do not assume from the section number alone.
     - If they ARE the same provision (renumbered): explain both and structure
       it as:
         "Under the old law (pre-July 2024): ... [SOURCE N]"
         "Under the new law (post-July 2024): ... [SOURCE N]"
     - If they are DIFFERENT, unrelated provisions that merely share a section
       number: explicitly state that the shared number is coincidental and the
       provisions are unrelated, then explain each one separately under its own
       act name. Do NOT imply one replaced the other.
4. Always cite the specific section number when it appears in the source header 
   (e.g., write "Section 9 of the Wildlife Protection Act" not just "the Act"). 
   Section numbers are provided in the source headers — use them.
   Do not cite section numbers that do not appear in the provided sources.
5. Only if the sources describe the SAME provision (per rule 3), end with a
   brief practical note:
   "If your matter arose before July 1, 2024, the [old act] applies.
    If it arose on or after July 1, 2024, the [new act] applies."
   If the provisions are unrelated (per rule 3), omit this note entirely —
   do not imply either one supersedes the other.
6. Keep the answer between 200 and 400 words.
7. Write in plain English that a non-lawyer can understand.
8. If any source is marked [STATUS: UNVERIFIED], you MUST explicitly tell the user that this specific
   section pairing has not been independently verified and recommend they confirm it against the
   official bare act text. Do NOT state it as settled fact.
9. Before citing [N], re-check that SOURCE N's own text actually supports the specific sentence you
   are citing it for. Do not reuse a source number out of habit when discussing content from a
   different source block — cite that other block's actual number instead. Citing the wrong source
   number is a factual error, the same as citing the wrong section.
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


def _order_chunks_for_synthesis(chunks: list[dict]) -> list[dict]:
    """
    When paired era context is present (both legacy and current-era
    chunks), reorder so legacy-era chunks come first. This keeps SOURCE
    numbering — and therefore citation numbering, since build_citations()
    must use this same ordering — consistent with the "old law first,
    new law second" narrative PAIRED_ACT_SYSTEM_PROMPT requires,
    regardless of which act the user mentioned first in their query.
    Stable sort: relative order within each era group is preserved.
    No-op when paired context isn't present.
    """
    if not _has_paired_context(chunks):
        return chunks
    era_rank = {"legacy": 0}
    return sorted(chunks, key=lambda c: era_rank.get(c.get("era", ""), 1))


def get_system_prompt(chunks: list[dict]) -> str:
    """Returns the appropriate system prompt based on chunk era composition."""
    return PAIRED_ACT_SYSTEM_PROMPT if _has_paired_context(chunks) else SYNTHESIS_SYSTEM_PROMPT


# ── Prompt builder ────────────────────────────────────────────────────────────

def build_synthesis_prompt(query: str, chunks: list[dict], intent: str = "legal_query") -> str:
    """
    Formats retrieved chunks as numbered [SOURCE N] blocks.
    When paired act chunks are present, injects an era context note
    before the sources block so the LLM knows what it's looking at.
    """
    chunks = _order_chunks_for_synthesis(chunks)
    logger.debug("[DIAGNOSE] SYNTHESIS TRIGGERED")

    # Unverified equivalence note — injected when system note chunk is marked UNVERIFIED
    has_unverified = any(
        "[STATUS: UNVERIFIED]" in c.get("text", "") for c in chunks
    )
    unverified_note = ""
    if has_unverified:
        unverified_note = (
            "[UNVERIFIED EQUIVALENCE WARNING]\n"
            "One or more sources contain a section pairing marked [STATUS: UNVERIFIED]. "
            "This means the equivalence has NOT been independently verified. "
            "You MUST tell the user explicitly that this specific pairing is unverified "
            "and recommend they confirm it against the official bare act text. "
            "Do NOT present the pairing as settled fact.\n\n"
        )

    # Equivalence priority note — injected when a system equivalence chunk is present.
    # Parses the target act/section from the SYSTEM NOTE text and identifies which
    # source slots actually contain the target section's statute text, then tells
    # the LLM explicitly so it cannot ignore them.
    equivalence_priority_note = ""
    eq_chunk = next(
        (c for c in chunks if c.get("chunk_id") == "_kg_equivalence_context"), None
    )
    if eq_chunk:
        import re as _re
        eq_text = eq_chunk.get("text", "")
        # Parse "X corresponds to Y ACT SECTION" pattern from the system note
        _target_act, _target_sec = "", ""
        _m = _re.search(
            r"corresponds to\s+([A-Z][^\d\n]+?)\s+(\d+[A-Z]?)\s*(?:\[|$|\n)",
            eq_text, _re.IGNORECASE
        )
        if _m:
            _target_act = _m.group(1).strip()
            _target_sec = _m.group(2).strip()

        # After _order_chunks_for_synthesis runs (called just above), find which
        # 1-indexed source slots contain the target section.
        _ordered = _order_chunks_for_synthesis(chunks)
        _target_source_nums = [
            str(i + 1)
            for i, c in enumerate(_ordered)
            if c.get("section", "").upper() == _target_sec.upper()
            and c.get("chunk_id") != "_kg_equivalence_context"
            and (not _target_act or _target_act.upper().split()[0]
                 in c.get("source", "").upper())
        ]

        if _target_act and _target_sec:
            _src_hint = (
                f" Its full text is in SOURCE{'S' if len(_target_source_nums) > 1 else ''} "
                f"{', '.join(_target_source_nums)}."
                if _target_source_nums else
                " Its full text was not retrieved — state the section number anyway."
            )
            equivalence_priority_note = (
                "[EQUIVALENCE ANSWER \u2014 AUTHORITATIVE]\n"
                f"The SYSTEM NOTE below has already identified the answer: "
                f"the equivalent section is {_target_act} Section {_target_sec}."
                f"{_src_hint}\n"
                "You MUST state this section number as the direct answer. "
                "Do NOT write that the equivalent 'cannot be confirmed', "
                "'is not provided in context', or that you 'cannot determine' it "
                "\u2014 the SYSTEM NOTE is the authoritative source for this mapping "
                "and overrides any such conclusion.\n"
                "Cite the retrieved sources for the actual text of each provision.\n\n"
            )
        else:
            # Fallback: no parseable target — use the generic advisory
            equivalence_priority_note = (
                "[EQUIVALENCE ANSWER \u2014 AUTHORITATIVE]\n"
                "A SYSTEM NOTE below states the correct cross-act section "
                "correspondence for this query. Treat it as authoritative. "
                "Do NOT conclude no equivalent exists when the SYSTEM NOTE "
                "has identified the pairing.\n\n"
            )


    # Era context note — only injected when paired chunks are present
    era_note = ""
    if _has_paired_context(chunks):
        era_note = (
            "[LEGAL ERA CONTEXT]\n"
            "Sources below include BOTH pre-July 2024 laws (IPC/CrPC/Evidence Act) "
            "AND post-July 2024 replacement laws (BNS/BNSS/BSA). "
            "These sources are NOT guaranteed to be the same provision renumbered \u2014 "
            "verify from the source text itself whether they cover the same subject "
            "matter before treating one as the successor of the other. "
            "Cutoff for the old->new law transition where applicable: July 1, 2024.\n\n"
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
        
    if intent == "legal_query":
        structure_prompt = (
            "Structure your answer using these sections:\n"
            "**Relevant Law:** State the act and section number directly\n"
            "**What it says:** Explain the provision in plain language  \n"
            "**Answer:** Direct answer to the question asked\n"
            "**Punishment/Remedy:** State explicitly if present, or state \"No specific punishment is prescribed for this provision\"\n"
            "**Note:** Only include if directly relevant — omit otherwise\n"
        )
    elif intent == "risk_check":
        structure_prompt = (
            "Structure your answer using these sections:\n"
            "**Legal Risk:** State the applicable law and section\n"
            "**Consequences:** Penalties, imprisonment, fines if present in sources\n"
            "**Procedure:** What authorities are involved\n"
            "**Advice:** Practical next step\n"
        )
    elif intent == "rights_check":
        structure_prompt = (
            "Structure your answer using these sections:\n"
            "**Your Rights:** List rights from retrieved sources only\n"
            "**Relevant Law:** Act and section for each right\n"
            "**What to do:** Practical steps\n"
        )
    else:
        structure_prompt = ""

    return (
        f"{equivalence_priority_note}"
        f"{unverified_note}"
        f"{era_note}"
        f"[RETRIEVED LEGAL SOURCES]\n"
        f"{sources_block}\n"
        f"[USER QUESTION]\n"
        f"{query}\n\n"
        f"[SYNTHESIS TASK]\n"
        f"Synthesize the above sources to answer the question.\n"
        f"- Cite every legal claim with its [SOURCE NUMBER] inline.\n"
        f"- If multiple sources address the same point, cite all of them.\n"
        f"- If both old and new law sources are present, explain both separately.\n\n"
        f"IMPORTANT: Only mention acts and sections that appear in the provided sources below. \n"
        f"Do not introduce any act, section, or legal concept not present in the source material.\n"
        f"Cite section numbers whenever they appear in the source headers.\n\n"
        f"{structure_prompt}"
        f"Answer:"
    )


# ── Citation builder ──────────────────────────────────────────────────────────

def build_citations(chunks: list[dict]) -> list[Citation]:
    chunks = _order_chunks_for_synthesis(chunks)
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

    # Act name verification
    def _act_matches(mention: str, available: set) -> bool:
        mention_words = set(mention.lower().split())
        for avail in available:
            avail_words = set(avail.lower().split())
            if len(mention_words & avail_words) >= 2:
                return True
        return False

    act_mentions = set(re.findall(r'\b(?:[A-Z][a-zA-Z]+\s+)+(?:Act|Code|Sanhita|Adhiniyam)\b', answer_text))
    available_acts = {str(c.get("source", "")).strip() for c in chunks if c.get("source")}
    
    phantom_acts = set()
    for act in act_mentions:
        if not _act_matches(act, available_acts):
            phantom_acts.add(act)
            
    if phantom_acts:
        return f"Answer mentions act(s) {phantom_acts} not found in retrieved sources. Possible hallucination."

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
    citations = build_citations(chunks)
    
    # Run grounding check on original LLM output before placeholder replacement
    grounding_warning = check_grounding(llm_answer.strip(), chunks)

    # Then do citation replacement on processed_answer
    processed_answer = llm_answer.strip()
    
    for cit in citations:
        idx = cit.source_number
        act = cit.source
        sec = cit.section
        
        if act and act != "Unknown":
            if sec:
                replacement = f"[{act}, Section {sec}]"
            else:
                replacement = f"[{act}]"
                
            processed_answer = re.sub(rf'\[{idx}\]', replacement, processed_answer)
            processed_answer = re.sub(rf'\[SOURCE {idx}\]', replacement, processed_answer, flags=re.IGNORECASE)

    # ── Citation-tag mismatch validation pass ────────────────────────────────
    # Mechanical check: for each sentence in processed_answer whose trailing
    # citation tag names a specific act, verify the act name also appears in
    # the sentence body.  Mismatches are downgraded to [Statute]/[System]
    # rather than left as provably wrong specific citations.
    processed_answer = validate_citation_tags(processed_answer)

    return LegalAnswer(
        answer_text        = processed_answer,
        citations          = citations,
        sources_consulted  = len(chunks),
        synthesis_note     = build_synthesis_note(chunks),
        grounding_warning  = grounding_warning,
        rewritten_queries  = rewritten_queries or [],
        reranker_used      = reranker_used,
    )
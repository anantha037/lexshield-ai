"""
LexShield AI — Query Rewriter  (Week 2, Day 2 — updated Week 3)
================================================================
Changes in this version:
  + decompose_query(query) added for complex multi-act queries
    Called by pipeline.py when adaptive_router returns "complex".
    Breaks the query into 2-3 independent sub-queries via Groq.
    Returns list[str] — each sub-query focuses on one act / concept.

Everything else is unchanged from the Week 2 version.

Takes one user query -> generates 3 rewritten queries covering different
legal angles -> caller runs hybrid search on all 4 (original + 3 rewrites)
-> deduplicated result pool goes to reranker.

Design:
  • Uses Groq LLaMA 3.3 70B (same as answer generation — no extra API)
  • Prompts for angle diversity: statutory text / punishment / procedure
  • Low temperature (0.3) for consistent but varied outputs
  • Strict JSON output -> robust parser with line-by-line fallback
  • Whole step fails gracefully — returns [original_query] on any error
  • Adds legal context injection (IPC/BNS parallel queries auto-generated)
"""

import os
import re
import json

import logging

logger = logging.getLogger(__name__)

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from rag.llm import llm

# ── Rewriter system prompt ────────────────────────────────────────────────────
REWRITER_SYSTEM = """You are a legal search query optimizer specializing in Indian law.
Your job is to take a user's legal question and generate alternative search queries
that will retrieve the most relevant legal provisions from a database of Indian statutes
and court judgments.

Rules:
1. Generate exactly 3 alternative queries.
2. Each query must approach the same legal issue from a DIFFERENT angle.
3. Use specific legal terminology when possible.
4. Keep each query under 20 words.
5. Return ONLY a JSON array of 3 strings. No explanation, no markdown.

Example output:
["Section 420 IPC cheating ingredients elements", "dishonest inducement delivery property deception punishment", "cheating criminal breach trust fraud Indian Penal Code"]
"""

REWRITER_USER_TEMPLATE = """Original query: {query}

Generate 3 alternative search queries covering:
1. The specific statutory provision or section number
2. The legal elements, ingredients, or definition
3. The punishment, procedure, or remedy

Return ONLY a JSON array of 3 strings."""

# ── Decomposer system prompt ──────────────────────────────────────────────────
_DECOMPOSE_SYSTEM = (
    "You are an Indian legal query decomposer. "
    "Your job is to break a complex multi-act legal question into independent "
    "sub-queries that can each be retrieved separately from a legal database. "
    "Each sub-query must focus on exactly one act or one legal concept. "
    "Return ONLY a valid JSON array of 2-3 strings. "
    "No markdown, no explanation, no preamble."
)

_DECOMPOSE_USER_TEMPLATE = """\
Complex legal query: {query}

Break this into 2-3 independent sub-queries.
Rules:
- Each sub-query must focus on ONE act or ONE legal concept
- Sub-queries must together cover the full intent of the original query
- Use specific section numbers and act names where present in the original
- Keep each sub-query under 25 words

Return ONLY a JSON array:
["sub-query about act 1 / concept 1", "sub-query about act 2 / concept 2"]
"""


# ── Query angle injectors ─────────────────────────────────────────────────────
BNS_IPC_PAIRS: dict[str, str] = {
    "murder":      "Section 302 IPC Section 101 BNS punishment",
    "cheating":    "Section 420 IPC Section 318 BNS fraud",
    "theft":       "Section 378 IPC Section 303 BNS stealing",
    "assault":     "Section 351 IPC Section 130 BNS hurt grievous",
    "rape":        "Section 376 IPC Section 63 BNS sexual assault",
    "kidnapping":  "Section 359 IPC Section 137 BNS abduction",
    "extortion":   "Section 383 IPC Section 308 BNS coercion",
    "defamation":  "Section 499 IPC Section 356 BNS reputation",
    "sedition":    "Section 124A IPC Section 152 BNS",
    "forgery":     "Section 463 IPC Section 334 BNS",
    "bribery":     "Prevention of Corruption Act Section 7 bribe",
    "eviction":    "tenant eviction grounds notice period procedure",
    "bail":        "bail non-bailable bailable Section 437 CrPC",
    "fir":         "First Information Report Section 154 CrPC registration",
    "consumer":    "Consumer Protection Act complaint forum redressal",
}


# ── Act-presence detector ───────────────────────────────────────────────────
# Mirrors the act patterns from agents/graph.py _ACT_RE so that follow-up
# queries like "what about section 8?" are correctly identified as act-free.
_ACT_PRESENT_RE = re.compile(
    r'\b(?:Indian Penal Code|Bharatiya Nyaya Sanhita'
    r'|Code of Criminal Procedure|Bharatiya Nagarik Suraksha Sanhita'
    r'|Indian Evidence Act|Bharatiya Sakshya Adhiniyam'
    r'|Negotiable Instruments Act|Protection of Children from Sexual Offences Act'
    r'|Consumer Protection Act|Information Technology Act'
    r'|Motor Vehicles Act|Transfer of Property Act'
    r'|Indian Contract Act|Prevention of Corruption Act'
    r'|Narcotic Drugs and Psychotropic Substances Act'
    r'|Unlawful Activities \(Prevention\) Act'
    r'|IPC|BNS|CrPC|BNSS|NI\s*Act|BSA|POCSO|NDPS|UAPA)\b',
    re.IGNORECASE,
)


def _query_has_act(query: str) -> bool:
    """Return True if the query explicitly names any Indian act or abbreviation."""
    return bool(_ACT_PRESENT_RE.search(query))


# ── JSON parser (shared) ──────────────────────────────────────────────────────

def _parse_json_array(raw: str) -> list[str]:
    """
    Parses LLM output into a list of strings.
    Tries JSON first, falls back to line-by-line extraction.
    Used by both rewrite() and decompose_query().
    """
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()]
    except json.JSONDecodeError:
        pass

    array_match = re.search(r"\[.*?\]", cleaned, re.DOTALL)
    if array_match:
        try:
            parsed = json.loads(array_match.group())
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()]
        except json.JSONDecodeError:
            pass

    # Line-by-line fallback
    queries = []
    for line in cleaned.splitlines():
        line = re.sub(r"^[\d\.\-\•\*]+\s*", "", line.strip())
        quoted = re.findall(r'"([^"]{10,})"', line)
        if quoted:
            queries.extend(quoted)
        elif 10 < len(line) < 120:
            queries.append(line)
    return [q.strip() for q in queries if q.strip()]


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY DECOMPOSER  (new — Week 3)
# ═══════════════════════════════════════════════════════════════════════════════

def decompose_query(query: str) -> list[str]:
    """
    Breaks a complex multi-act query into 2-3 independent sub-queries.

    Called by pipeline.py ONLY when adaptive_router returns "complex".
    Each sub-query is retrieved independently via hybrid_search, then the
    chunk pools are merged and de-duplicated before the synthesizer.

    Args:
        query: The original (possibly preprocessed) user query.

    Returns:
        list[str] of 2-3 sub-queries.  Always returns at least [query]
        on failure so the pipeline always has something to retrieve with.

    Token cost: one Groq call (~150 tokens) — only on complex queries.
    """
    query = query.strip()
    if not query:
        return [query]

    try:
        prompt = _DECOMPOSE_USER_TEMPLATE.format(query=query)
        raw    = llm.generate(
            prompt=prompt,
            system_prompt=_DECOMPOSE_SYSTEM,
            temperature=0.1,   # low temp — want deterministic decomposition
            max_tokens=200,
        )
        sub_queries = _parse_json_array(raw)

        # Filter: must be non-trivial (>10 chars) and different from original
        original_lower = query.lower()
        cleaned = [
            sq for sq in sub_queries
            if len(sq) > 10 and sq.lower() != original_lower
        ]

        if len(cleaned) < 2:
            # Decomposition returned garbage — fall back to original
            logger.info(f"[QueryDecomposer] Decomposition insufficient ({cleaned!r}) — using original")
            return [query]

        logger.info(f"[QueryDecomposer] Decomposed into {len(cleaned)} sub-queries:")
        for i, sq in enumerate(cleaned, 1):
            logger.debug(f"  [{i}] {sq}")

        return cleaned[:3]   # cap at 3 sub-queries

    except Exception as exc:
        logger.info(f"[QueryDecomposer] Failed ({exc}) — returning original query")
        return [query]


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY REWRITER  (unchanged from Week 2)
# ═══════════════════════════════════════════════════════════════════════════════


def _get_statutory_hint(query: str) -> str:
    """
    Extract a statutory hint from the query by matching against BNS_IPC_PAIRS.

    Scans the query for known legal concept keywords (murder, cheating, bail, etc.)
    and returns the corresponding pre-built statutory search string that covers
    both legacy (IPC/CrPC) and current (BNS/BNSS) act references.

    Returns an empty string when no keyword matches — the caller treats this as
    falsy and skips hint injection.
    """
    q_lower = query.lower()
    for keyword, statutory_query in BNS_IPC_PAIRS.items():
        if keyword in q_lower:
            return statutory_query
    return ""


class QueryRewriter:

    """
    LLM-based query rewriter for legal retrieval.

    Usage:
        from rag.query_rewriter import query_rewriter
        all_queries = query_rewriter.rewrite(user_query)
        # -> [original, rewrite1, rewrite2, rewrite3, (optional statutory hint)]
    """

    def __init__(self, temperature: float = 0.3, max_tokens: int = 200):
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def rewrite(self, query: str) -> list[str]:
        """
        Returns list of queries: [original] + up to 3 LLM rewrites + optional hint.
        Always returns at least [original] even on complete failure.

        Act-context injection:
          When the query has no explicit act reference (acts=[]) and the session
          has a persisted last_act, the query is augmented with
          "under {last_act}" before being sent to the LLM rewriter and returned
          as the lead query.  This anchors ambiguous follow-ups ("what about
          section 8?") to the correct act without touching the pipeline.

          If the query explicitly names a different act, set_last_act is updated
          so the new act takes precedence from this turn onwards.
        """
        query = query.strip()
        if not query:
            return [query]

        # Act-context injection removed. This responsibility now belongs
        # solely to rewrite_for_retrieval() in this module, which is called
        # earlier in rag/pipeline.py with full 2-turn conversation context
        # instead of a single persisted last_act string. Two independent
        # follow-up-resolution heuristics caused the section fast-path to
        # fire on stale entities; rewrite() now only generates angle-diverse
        # rewrites of whatever query it is given.

        all_queries: list[str] = [query]
        hint = _get_statutory_hint(query)


        try:
            prompt = REWRITER_USER_TEMPLATE.format(query=query)
            raw    = llm.generate(
                prompt=prompt,
                system_prompt=REWRITER_SYSTEM,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            rewrites = _parse_json_array(raw)

            seen = {query.lower()}
            for r in rewrites:
                if r.lower() not in seen and len(r) > 5:
                    all_queries.append(r)
                    seen.add(r.lower())

        except Exception as e:
            logger.info(f"[QueryRewriter] LLM call failed: {e} — using original query only.")

        if hint and hint.lower() not in {q.lower() for q in all_queries}:
            all_queries.append(hint)

        return all_queries

    def rewrite_explain(self, query: str) -> None:
        queries = self.rewrite(query)
        logger.debug(f"\n[QueryRewriter] Input: '{query}'")
        logger.info(f"[QueryRewriter] Generated {len(queries)} queries:")
        for i, q in enumerate(queries):
            tag = "original" if i == 0 else f"rewrite {i}"
            logger.debug(f"  [{tag}]  {q}")


# ═══════════════════════════════════════════════════════════════════════════════
# CONVERSATIONAL QUERY REFORMULATOR  (replaces orchestrator entity injection)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Section-number detection ──────────────────────────────────────────────────
_SECTION_NUM_RE = re.compile(r'\bsection\s+\d+', re.IGNORECASE)

# ── Broad act-name detection (covers all major Indian statutes) ───────────────
_REWRITE_ACT_RE = re.compile(
    r'\b(?:IPC|BNS|CrPC|BNSS|NDPS|IT\s*Act|Companies\s+Act'
    r'|Hindu\s+Marriage\s+Act|CPC|IEA|BSA|POCSO|PMLA|UAPA'
    r'|RERA|FEMA|SEBI|POSH|GST|CGST|IGST'
    r'|Indian\s+Penal\s+Code|Bharatiya\s+Nyaya\s+Sanhita'
    r'|Code\s+of\s+Criminal\s+Procedure|Indian\s+Evidence\s+Act'
    r'|Consumer\s+Protection\s+Act|Motor\s+Vehicles?\s+Act'
    r'|Negotiable\s+Instruments?\s+Act|Information\s+Technology\s+Act'
    r'|Narcotic\s+Drugs|Transfer\s+of\s+Property\s+Act'
    r'|Indian\s+Contract\s+Act|Insolvency\s+and\s+Bankruptcy'
    r'|Prevention\s+of\s+Corruption|Domestic\s+Violence'
    r'|Right\s+to\s+Information|NI\s*Act|MV\s*Act|DV\s*Act)\b',
    re.IGNORECASE,
)

_REWRITE_PROMPT = """You are a legal query reformulator.
Given a conversation history and a follow-up question, rewrite the follow-up as a complete, self-contained legal search query.
Output ONLY the rewritten query. No explanation. No preamble.

Conversation history:
{last_2_turns}

Follow-up question: {query}

Rewritten query:"""


def _extract_last_2_turns(context_block: str) -> str:
    """Extract the last 2 user/assistant turn pairs from context_block."""
    lines = context_block.strip().splitlines()
    turns: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.startswith(("User:", "Assistant:", "[USER]", "[ASSISTANT]")):
            if current:
                turns.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        turns.append("\n".join(current))
    # last 2 pairs = up to 4 entries (user+assistant, user+assistant)
    return "\n".join(turns[-4:])


def rewrite_for_retrieval(query: str, context_block: str) -> str:
    """
    Rewrite a vague follow-up query into a self-contained search query
    using the last 2 conversation turns from context_block.

    Conditions for rewriting (ALL must be true):
      1. context_block is non-empty (there is prior conversation)
      2. query has no section number (no digits after "section")
      3. query has no act name from the broad coverage list

    Returns the original query unchanged if any condition is false,
    or if the LLM call fails for any reason — a rewriter failure must
    never block retrieval.

    Token cost: one Groq call with max_tokens=100.
    """
    logger.debug(f"[QueryRewriter] rewrite_for_retrieval called: query={query!r}, "
                 f"context_empty={not bool(context_block and context_block.strip())}")

    # Condition 1: must have conversation history
    if not context_block or not context_block.strip():
        logger.debug("[QueryRewriter] rewrite_for_retrieval: no context — skipping")
        return query

    # Condition 2: skip if query has section number
    if _SECTION_NUM_RE.search(query):
        logger.debug("[QueryRewriter] rewrite_for_retrieval: section number found — skipping")
        return query

    # Condition 3: skip if query has an act name
    if _REWRITE_ACT_RE.search(query):
        logger.debug("[QueryRewriter] rewrite_for_retrieval: act name found — skipping")
        return query

    # All conditions met — rewrite via LLM
    try:
        last_2 = _extract_last_2_turns(context_block)
        prompt = _REWRITE_PROMPT.format(last_2_turns=last_2, query=query)
        rewritten = llm.generate(
            prompt=prompt,
            system_prompt="",
            temperature=0.0,
            max_tokens=100,
        )
        rewritten = rewritten.strip().strip('"').strip("'").strip()

        # ── Output validation guard ──────────────────────────────────────
        # The LLM occasionally echoes the prompt or context_block instead of
        # producing a clean rewrite (observed: full [CONVERSATION HISTORY]
        # block returned verbatim, poisoning downstream section extraction).
        _invalid_markers = (
            "[CONVERSATION HISTORY]", "[USER PROFILE]", "[END HISTORY]",
            "[END PROFILE]", "[CONVERSATION SUMMARY]",
            "Rewritten query:", "Follow-up question:", "Conversation history:",
        )
        is_invalid = (
            not rewritten
            or len(rewritten) <= 5
            or len(rewritten) > max(len(query) * 4, 200)
            or any(marker in rewritten for marker in _invalid_markers)
        )
        if is_invalid:
            logger.warning(
                f"[QueryRewriter] rewrite_for_retrieval: rejected invalid output "
                f"(len={len(rewritten)}) — using original query"
            )
            return query

        logger.info(f"[QueryRewriter] rewrite_for_retrieval: {query!r} -> {rewritten!r}")
        return rewritten
    except Exception as exc:
        logger.warning(f"[QueryRewriter] rewrite_for_retrieval failed ({exc}) — using original")
        return query


# ── Singletons ─────────────────────────────────────────────────────────────────
query_rewriter = QueryRewriter()
"""
LexShield AI — Query Rewriter  (Week 2, Day 2)
===============================================
Takes one user query → generates 3 rewritten queries covering different
legal angles → caller runs hybrid search on all 4 (original + 3 rewrites)
→ deduplicated result pool goes to reranker.

Design:
  • Uses Groq LLaMA 3.3 70B (same as answer generation — no extra API)
  • Prompts for angle diversity: statutory text / punishment / procedure
  • Low temperature (0.3) for consistent but varied outputs
  • Strict JSON output → robust parser with line-by-line fallback
  • Whole step fails gracefully — returns [original_query] on any error
  • Adds legal context injection (IPC/BNS parallel queries auto-generated)
"""

import os
import re
import json

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from rag.llm import llm  # LegalLLM singleton (Week 1, unchanged)

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


# ── Query angle injectors ─────────────────────────────────────────────────────
# For known patterns, add a guaranteed statutory angle even before LLM call.
# This ensures BNS/IPC parallel coverage for criminal law queries.

BNS_IPC_PAIRS: dict[str, str] = {
    "murder":          "Section 302 IPC Section 101 BNS punishment",
    "cheating":        "Section 420 IPC Section 318 BNS fraud",
    "theft":           "Section 378 IPC Section 303 BNS stealing",
    "assault":         "Section 351 IPC Section 130 BNS hurt grievous",
    "rape":            "Section 376 IPC Section 63 BNS sexual assault",
    "kidnapping":      "Section 359 IPC Section 137 BNS abduction",
    "extortion":       "Section 383 IPC Section 308 BNS coercion",
    "defamation":      "Section 499 IPC Section 356 BNS reputation",
    "sedition":        "Section 124A IPC Section 152 BNS",
    "forgery":         "Section 463 IPC Section 334 BNS",
    "bribery":         "Prevention of Corruption Act Section 7 bribe",
    "eviction":        "tenant eviction grounds notice period procedure",
    "bail":            "bail non-bailable bailable Section 437 CrPC",
    "fir":             "First Information Report Section 154 CrPC registration",
    "consumer":        "Consumer Protection Act complaint forum redressal",
}


def _get_statutory_hint(query: str) -> str | None:
    """
    Returns a pre-built statutory angle string if query matches a known topic.
    Used as a guaranteed 4th query to ensure statutory coverage.
    """
    q_lower = query.lower()
    for keyword, hint in BNS_IPC_PAIRS.items():
        if keyword in q_lower:
            return hint
    return None


# ── JSON parser ───────────────────────────────────────────────────────────────

def _parse_rewritten_queries(raw: str) -> list[str]:
    """
    Parses LLM output into a list of query strings.
    Tries JSON first, falls back to line-by-line extraction.
    """
    # Strip markdown fences if present
    cleaned = re.sub(r'```(?:json)?', '', raw).strip()
    cleaned = cleaned.strip('`').strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [str(q).strip() for q in parsed if str(q).strip()]
    except json.JSONDecodeError:
        pass

    # Try to find a JSON array anywhere in the text
    array_match = re.search(r'\[.*?\]', cleaned, re.DOTALL)
    if array_match:
        try:
            parsed = json.loads(array_match.group())
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()]
        except json.JSONDecodeError:
            pass

    # Line-by-line fallback — extract quoted strings or numbered lines
    queries = []
    for line in cleaned.splitlines():
        line = line.strip()
        # Remove leading numbers/bullets: "1.", "•", "-"
        line = re.sub(r'^[\d\.\-\•\*]+\s*', '', line)
        # Extract content from quotes if present
        quoted = re.findall(r'"([^"]{10,})"', line)
        if quoted:
            queries.extend(quoted)
        elif len(line) > 10 and len(line) < 120:
            queries.append(line)

    return [q.strip() for q in queries if q.strip()][:3]


# ── Query rewriter class ──────────────────────────────────────────────────────

class QueryRewriter:
    """
    LLM-based query rewriter for legal retrieval.

    Usage:
        from rag.query_rewriter import query_rewriter
        all_queries = query_rewriter.rewrite(user_query)
        # → [original, rewrite1, rewrite2, rewrite3, (optional statutory hint)]
    """

    def __init__(self, temperature: float = 0.3, max_tokens: int = 200):
        self.temperature = temperature
        self.max_tokens  = max_tokens

    def rewrite(self, query: str) -> list[str]:
        """
        Returns list of queries: [original] + up to 3 LLM rewrites + optional hint.
        Always returns at least [original] even on complete failure.
        """
        query = query.strip()
        if not query:
            return [query]

        # Start with original
        all_queries: list[str] = [query]

        # Add statutory hint if topic is recognised (guaranteed coverage)
        hint = _get_statutory_hint(query)

        # Generate LLM rewrites
        try:
            prompt = REWRITER_USER_TEMPLATE.format(query=query)
            raw    = llm.generate(
                prompt=prompt,
                system_prompt=REWRITER_SYSTEM,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            rewrites = _parse_rewritten_queries(raw)

            # Deduplicate against original (case-insensitive)
            seen = {query.lower()}
            for r in rewrites:
                if r.lower() not in seen and len(r) > 5:
                    all_queries.append(r)
                    seen.add(r.lower())

        except Exception as e:
            print(f"[QueryRewriter] LLM call failed: {e} — using original query only.")

        # Add statutory hint if not already covered
        if hint and hint.lower() not in {q.lower() for q in all_queries}:
            all_queries.append(hint)

        return all_queries  # [original, rewrite1, rewrite2, rewrite3, (hint)]

    def rewrite_explain(self, query: str) -> None:
        """Debug: prints all generated queries."""
        queries = self.rewrite(query)
        print(f"\n[QueryRewriter] Input: '{query}'")
        print(f"[QueryRewriter] Generated {len(queries)} queries:")
        for i, q in enumerate(queries):
            tag = "original" if i == 0 else f"rewrite {i}"
            print(f"  [{tag}]  {q}")


# ── Singleton ─────────────────────────────────────────────────────────────────
query_rewriter = QueryRewriter()
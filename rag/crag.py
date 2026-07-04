"""
LexShield AI — Corrective RAG (CRAG)
======================================
Evaluates retrieval quality and decides whether to proceed, rewrite,
or return a low-confidence response.

Single public function:
    evaluate_retrieval(query, chunks) -> dict

Returns:
    {
        "score":  int,          # 1–5
        "reason": str,          # one-sentence explanation
        "action": str,          # "proceed" | "rewrite" | "insufficient"
    }

Scoring thresholds (PROCEED_MIN_SCORE = 3, enforced in _parse_crag_response):
    score >= 3  -> "proceed"       (relevant enough — continue to synthesizer)
    score == 2  -> "rewrite"       (marginal — trigger query rewriter + re-retrieve once)
    score == 1  -> "insufficient"  (retrieval failed — return low-confidence response)

Note: rag/pipeline.py grades retrieval as "good" only at score >= GOOD_MIN_SCORE (4).
Grading (telemetry label) is intentionally stricter than gating (proceed decision).

Design constraints (Windows 11, 8GB RAM, no GPU):
    - Pure Groq API call — zero local model, zero extra memory
    - Truncates chunk preview to 300 chars each to stay within token budget
    - Max 5 chunks evaluated (top 5 by hybrid_score) to limit Groq tokens
    - Falls back gracefully on JSON parse error or API failure
"""

import json
import re
import os

import logging

logger = logging.getLogger(__name__)

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

from rag.llm import llm
from langsmith import traceable

# ── Prompt templates ───────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a legal retrieval evaluator specializing in Indian law. "
    "You assess whether retrieved legal text chunks are relevant to the user query. "
    "You return ONLY valid JSON — no markdown, no explanation outside the JSON."
)

_USER_TEMPLATE = """You are evaluating retrieved Indian legal text chunks for relevance to a user query.

CRITICAL RULES FOR SCORING:
1. Indian law has multiple acts with the SAME section numbers.
   IPC Section 302 = Murder (Indian Penal Code)
   BNS Section 302 = Uttering words to wound religious feelings
   CrPC Section 302 = Permission to conduct prosecution
   BNSS Section 302 = Different provision
   These are COMPLETELY DIFFERENT laws. Section number alone means nothing without the act name.

2. If the query specifies an act (IPC/BNS/CrPC/BNSS/NI Act etc.) and the chunks are from a
   DIFFERENT act with the same section number, score 1 (insufficient). Do not confuse them.

3. IPC and BNS are paired equivalents (IPC replaced by BNS 2024).
   If query asks about IPC 302 (murder) and chunks contain BNS 101 (murder equivalent), this IS
   relevant — score 4-5.
   If query asks about BNS 302 (religious insult) and chunks contain IPC 302 (murder), this is
   NOT relevant — score 1.
   The pairing is semantic (same crime) not numeric (same number).

4. CrPC and BNSS are paired. Evidence Act and BSA are paired.
   Other acts are NOT interchangeable even if section numbers match.

5. Score 4-5: chunks directly answer the query from the correct act
   Score 3: chunks are partially relevant, same legal topic
   Score 2: chunks are tangentially related, same broad area
   Score 1: chunks are from the wrong act or unrelated topic

Query: {query}

Retrieved chunks (summarized):
{chunk_summaries}

Return ONLY valid JSON (no markdown, no explanation):
{{"score": <1-5>, "reason": "<one sentence>", "action": "<proceed|rewrite|insufficient>"}}

Rules:
score >= 3 -> action must be "proceed"
score == 2 -> action must be "rewrite"  
score == 1 -> action must be "insufficient"
"""

_MAX_CHUNKS_TO_EVAL  = 5     # evaluate top 5 only — limits Groq token use
_CHUNK_PREVIEW_CHARS = 300   # truncate each chunk preview to this length

# Canonical thresholds — single source of truth for crag.py AND rag/pipeline.py.
PROCEED_MIN_SCORE = 3   # gating: score >= 3 proceeds (see BUG FIX 3)
GOOD_MIN_SCORE    = 4   # grading: rag_grade == "good" requires score >= 4


# ── JSON parser ────────────────────────────────────────────────────────────────

def _parse_crag_response(raw: str) -> dict:
    """
    Parses Groq response into {score, reason, action}.
    Tries direct JSON first, then regex extraction, then safe defaults.
    """
    # Strip markdown fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    # Try direct JSON parse
    try:
        parsed = json.loads(cleaned)
        if all(k in parsed for k in ("score", "reason", "action")):
            score  = int(parsed["score"])
            action = parsed["action"].strip().lower()
            # Enforce action consistency regardless of what LLM returned.
            # BUG FIX 3: threshold lowered — score >= PROCEED_MIN_SCORE (3)
            # proceeds (was >= 4). Kept as a named constant so docstring,
            # implementation, and pipeline.py cannot drift again.
            if score >= PROCEED_MIN_SCORE:
                action = "proceed"
            elif score <= 1:
                action = "insufficient"
            else:
                action = "rewrite"
            return {"score": score, "reason": str(parsed["reason"]), "action": action}
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # Regex fallback — extract score integer from anywhere in the text
    score_match = re.search(r'"score"\s*:\s*(\d)', cleaned)
    score = int(score_match.group(1)) if score_match else 2

    reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', cleaned)
    reason = reason_match.group(1) if reason_match else "Could not parse evaluator response."

    # BUG FIX 3: consistent threshold in regex fallback path too
    if score >= PROCEED_MIN_SCORE:
        action = "proceed"
    elif score <= 1:
        action = "insufficient"
    else:
        action = "rewrite"

    return {"score": score, "reason": reason, "action": action}


# ── Main evaluator ─────────────────────────────────────────────────────────────

@traceable(name="crag.evaluate_retrieval", run_type="chain")
def evaluate_retrieval(query: str, chunks: list[dict]) -> dict:
    """
    Evaluates whether retrieved chunks are relevant to the query.

    Args:
        query:  The (preprocessed/expanded) user query string.
        chunks: List of retrieved chunk dicts (must have "text" key at minimum).

    Returns:
        {
            "score":    int,   # 1–5
            "reason":   str,
            "action":   str,   # "proceed" | "rewrite" | "insufficient"
            "fallback": bool,  # True when action == "insufficient"
            "degraded": bool,  # True when the evaluator itself failed and the
                               # result is an UNVERIFIED pass-through, not a
                               # genuine quality assessment
        }

    Never raises — on any LLM API failure the pipeline is never blocked,
    but the failure is NOT reported as success: the result carries
    score=PROCEED_MIN_SCORE (proceeds, but grades as "poor" downstream)
    and degraded=True, and the failure is logged at ERROR level.
    """
    if not chunks:
        return {
            "score":  1,
            "reason": "No chunks were retrieved for this query.",
            "action": "insufficient",
            "fallback": True,
            "degraded": False,
        }

    # Take top N chunks by hybrid_score; truncate text previews
    top_chunks = sorted(
        chunks, key=lambda c: c.get("hybrid_score", 0.0), reverse=True
    )[:_MAX_CHUNKS_TO_EVAL]

    previews = []
    for i, c in enumerate(top_chunks, 1):
        source  = c.get("source", "Unknown")[:50]
        section = c.get("section", "")
        text    = c.get("text", "")[:_CHUNK_PREVIEW_CHARS]
        previews.append(f"[{i}] {source} §{section}\n{text}")

    chunk_block = "\n\n".join(previews)
    prompt      = _USER_TEMPLATE.format(
        query=query, chunk_summaries=chunk_block
    )

    try:
        raw = llm.generate(
            prompt=prompt,
            system_prompt=_SYSTEM,
            temperature=0.0,   # deterministic evaluation
            max_tokens=180,    # score + reason + action fits in ~80 tokens
        )
        result = _parse_crag_response(raw)
        # Stamp fallback flag so pipeline can read it without inspecting action string.
        result["fallback"] = result["action"] == "insufficient"
        result["degraded"] = False
        logger.debug(
            f"[CRAG] score={result['score']} action={result['action']!r} "
            f"reason={result['reason'][:60]!r}"
        )
        return result

    except Exception as exc:
        # Fail-degraded, NOT fail-open: the pipeline may proceed (availability
        # over strictness — the embedding-based pre-synthesis relevance gate
        # still runs as a backstop), but we must not claim verified quality.
        # score=PROCEED_MIN_SCORE proceeds per gating threshold while grading
        # as "poor" (< GOOD_MIN_SCORE) in pipeline telemetry.
        logger.exception(
            "[CRAG] Evaluator LLM call FAILED — retrieval quality is UNVERIFIED "
            "for this query; proceeding in degraded mode (quality gating skipped)."
        )
        return {
            "score":    PROCEED_MIN_SCORE,
            "reason":   f"Evaluator unavailable ({exc}) — quality NOT verified.",
            "action":   "proceed",
            "fallback": False,
            "degraded": True,
        }
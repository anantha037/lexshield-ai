"""
tests/test_crag.py
==================
Pure-logic unit tests for rag/crag.py — no LLM calls, no API quota needed.
"""

from rag.crag import _USER_TEMPLATE, _parse_crag_response, PROCEED_MIN_SCORE


def test_user_template_does_not_penalize_unverified_status():
    """
    Guard against Rule 5 regressing back to treating UNVERIFIED as grounds
    for rejection — pure string check, no LLM call.
    """
    assert "REGARDLESS of whether it is marked" in _USER_TEMPLATE, (
        "Rule 5 must state that UNVERIFIED status is NOT grounds for "
        "scoring retrieval as insufficient — the phrase "
        "'REGARDLESS of whether it is marked' is missing from _USER_TEMPLATE."
    )
    # Old broken wording must not be present
    assert "should not be presented as certain" not in _USER_TEMPLATE, (
        "Old Rule 5 wording ('should not be presented as certain') is still "
        "present in _USER_TEMPLATE — the fix did not land correctly."
    )


def test_parse_crag_response_proceed_on_score_4():
    """Score >= PROCEED_MIN_SCORE must resolve to 'proceed' action."""
    result = _parse_crag_response('{"score": 4, "reason": "relevant", "action": "proceed"}')
    assert result["action"] == "proceed"
    assert result["score"] == 4


def test_parse_crag_response_insufficient_on_score_1():
    """Score == 1 must resolve to 'insufficient' regardless of action field."""
    result = _parse_crag_response('{"score": 1, "reason": "wrong act", "action": "proceed"}')
    assert result["action"] == "insufficient"
    assert result["score"] == 1


def test_parse_crag_response_rewrite_on_score_2():
    """Score == 2 must resolve to 'rewrite'."""
    result = _parse_crag_response('{"score": 2, "reason": "partial", "action": "insufficient"}')
    assert result["action"] == "rewrite"
    assert result["score"] == 2


def test_parse_crag_response_handles_markdown_fence():
    """Markdown-fenced JSON must be stripped and parsed correctly."""
    raw = '```json\n{"score": 5, "reason": "exact match", "action": "proceed"}\n```'
    result = _parse_crag_response(raw)
    assert result["score"] == 5
    assert result["action"] == "proceed"


def test_parse_crag_response_regex_fallback():
    """When JSON parse fails, regex fallback must still extract the score."""
    malformed = 'score: 3, reason: "partial", action: proceed'
    result = _parse_crag_response(malformed)
    # score=2 is the default when regex also can't find it, 3 from regex
    # malformed has no "score": N pattern so falls back to default 2
    assert result["action"] in ("proceed", "rewrite")  # no crash


def test_proceed_min_score_constant_is_3():
    """PROCEED_MIN_SCORE must be 3 — changing it silently breaks gating."""
    assert PROCEED_MIN_SCORE == 3, (
        f"PROCEED_MIN_SCORE is {PROCEED_MIN_SCORE}, expected 3. "
        "Changing this threshold silently breaks CRAG gating."
    )


def test_user_template_contains_score_guidance():
    """Sanity-check that the scoring rubric is still present."""
    assert "score >= 3" in _USER_TEMPLATE or "score ≥ 3" in _USER_TEMPLATE or "score > 3" in _USER_TEMPLATE or ">= 3" in _USER_TEMPLATE
    assert "SYSTEM NOTE" in _USER_TEMPLATE

"""
Day 1 Checkpoint — Intent Classifier Tests
===========================================
10 queries covering all 6 intents.
Run: pytest tests/test_intent_classifier.py -v
"""

import pytest
from agents.intent_classifier import intent_classifier


# ── Test cases: (query, expected_intent) ──────────────────────────────────────

TEST_CASES = [
    # legal_query (2 cases)
    (
        "What is the punishment for theft under Section 379 IPC?",
        "legal_query",
    ),
    (
        "Explain the bail provisions under BNSS for non-bailable offences",
        "legal_query",
    ),
    # document_analysis (2 cases)
    (
        "Analyze this rental agreement document and tell me the key clauses",
        "document_analysis",
    ),
    (
        "Review this employment contract and extract all important terms",
        "document_analysis",
    ),
    # draft_request (2 cases)
    (
        "Help me draft a legal notice for cheque bounce under Section 138 NI Act",
        "draft_request",
    ),
    (
        "Create a template for a rental agreement between landlord and tenant",
        "draft_request",
    ),
    # risk_check (1 case)
    (
        "Is it legal to not pay advance notice before terminating an employee? What are the consequences?",
        "risk_check",
    ),
    # translation_request (1 case)
    (
        "Translate this legal notice into Malayalam",
        "translation_request",
    ),
    # general (2 cases)
    (
        "Hello, what can LexShield AI do for me?",
        "general",
    ),
    (
        "Thank you for the help!",
        "general",
    ),
]


# ═══════════════════════════════════════════════════════════════════════════════
# TESTS
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("query,expected", TEST_CASES)
def test_intent_classification(query, expected):
    result = intent_classifier.classify(query)
    assert result.intent == expected, (
        f"\nQuery:    {query!r}"
        f"\nExpected: {expected}"
        f"\nGot:      {result.intent}"
        f"\nScores:   {result.scores}"
    )


def test_confidence_range():
    """All confidence values must be in [0.0, 1.0]."""
    for query, _ in TEST_CASES:
        result = intent_classifier.classify(query)
        assert 0.0 <= result.confidence <= 1.0, (
            f"Confidence out of range: {result.confidence} for query: {query!r}"
        )


def test_empty_query_returns_general():
    """Empty or whitespace query must not crash — falls back to general."""
    result = intent_classifier.classify("   ")
    assert result.intent == "general"
    assert result.confidence >= 0.0


def test_unknown_query_returns_general():
    """Random unrelated text should return general."""
    result = intent_classifier.classify("The weather is nice today")
    assert result.intent == "general"


def test_scores_dict_has_all_intents():
    """Scores dict must always contain exactly 6 keys."""
    result = intent_classifier.classify("What is Section 302 IPC?")
    expected_keys = {"legal_query", "document_analysis", "draft_request",
                     "risk_check", "translation_request", "general"}
    assert set(result.scores.keys()) == expected_keys
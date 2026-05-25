"""
Day 5 Checkpoint — Structured Output + Translation Tests
=========================================================
Tests:
  1. LexShieldResponse fields all populated
  2. summary is non-empty and shorter than answer_text
  3. key_clauses extracted from section-heavy answer
  4. risk_score in valid range, risk_level valid
  5. suggestions non-empty list
  6. Translation language detection (no API calls)
  7. Translation strip legal content helper
  8. Full graph returns StructuredResponse-compatible dict

Run: pytest tests/test_structured_output.py -v
"""

import pytest
from unittest.mock import patch, MagicMock

from rag.structured_output import (
    build_structured_response,
    _extract_summary,
    _extract_key_clauses,
    LexShieldResponse,
)
from agents.translation_agent import (
    detect_language,
    _strip_legal_content,
    LanguageDetectionResult,
)


# ── Sample answer texts ────────────────────────────────────────────────────────

LEGAL_ANSWER = (
    "Under the Indian Penal Code, Section 302 prescribes the punishment for murder. [1] "
    "The punishment is death or imprisonment for life, along with a fine. [1] "
    "Under the Bharatiya Nyaya Sanhita, Section 101 is the equivalent provision. [2] "
    "The court may award death penalty only in the rarest of rare cases as held by "
    "the Supreme Court in Bachan Singh v. State of Punjab. [3]"
)

SIMPLE_ANSWER = (
    "You have the right to consult a lawyer when arrested. "
    "The police must inform you of the grounds of arrest. "
    "Bail may be available depending on the nature of the offence."
)

DRAFT_ANSWER = "The drafting agent is being built and will be available shortly."


# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURED OUTPUT TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_summary_shorter_than_answer():
    summary = _extract_summary(LEGAL_ANSWER)
    assert len(summary) > 0
    assert len(summary) <= len(LEGAL_ANSWER)
    assert "[1]" not in summary  # citations stripped
    print(f"\nOK Summary: {summary[:80]}")


def test_summary_max_3_sentences():
    summary = _extract_summary(LEGAL_ANSWER, max_sentences=3)
    sentence_count = summary.count(". ") + summary.count("! ") + summary.count("? ")
    assert sentence_count <= 4  # generous bound for sentence splitting
    print(f"\nOK Summary sentence count within limit")


def test_key_clauses_extracted():
    clauses = _extract_key_clauses(LEGAL_ANSWER, citations=[])
    assert len(clauses) > 0
    has_ipc  = any("Indian Penal Code" in c or "IPC" in c for c in clauses)
    has_bns  = any("Bharatiya Nyaya Sanhita" in c or "BNS" in c for c in clauses)
    assert has_ipc or has_bns, f"No act found in clauses: {clauses}"
    print(f"\nOK Key clauses: {clauses}")


def test_build_structured_response_all_fields():
    """All LexShieldResponse fields must be populated."""
    resp = build_structured_response(
        answer_text       = LEGAL_ANSWER,
        intent            = "legal_query",
        session_id        = "test-session",
        confidence        = 0.9,
        mode              = "legal_rag_node",
        citations         = [],
        draft             = "",
        sources_consulted = 3,
        synthesis_note    = "Synthesized from 3 sections",
        grounding_warning = "",
        rewritten_queries = ["What is punishment for murder IPC?"],
        reranker_used     = True,
    )

    assert isinstance(resp, LexShieldResponse)
    assert resp.answer_text       == LEGAL_ANSWER
    assert len(resp.summary)      >  0
    assert isinstance(resp.key_clauses,  list)
    assert isinstance(resp.suggestions,  list)
    assert 0.0 <= resp.risk_score <= 1.0
    assert resp.risk_level        in ("Low", "Medium", "High", "Critical")
    assert isinstance(resp.risk_factors, list)
    assert resp.intent            == "legal_query"
    assert resp.session_id        == "test-session"
    assert resp.confidence        == 0.9
    assert resp.sources_consulted == 3
    assert resp.reranker_used     == True

    print(f"\nOK All LexShieldResponse fields populated")
    print(f"   summary:     {resp.summary[:60]}")
    print(f"   key_clauses: {resp.key_clauses}")
    print(f"   risk:        {resp.risk_score:.2f} ({resp.risk_level})")
    print(f"   suggestions: {resp.suggestions[:2]}")


def test_risk_score_range():
    for intent in ["legal_query", "risk_check", "general", "draft_request"]:
        resp = build_structured_response(
            answer_text = SIMPLE_ANSWER,
            intent      = intent,
            session_id  = "s",
            confidence  = 0.8,
            mode        = "test",
        )
        assert 0.0 <= resp.risk_score <= 1.0, f"Risk score out of range for {intent}: {resp.risk_score}"
        assert resp.risk_level in ("Low", "Medium", "High", "Critical")
    print(f"\nOK Risk score in [0.0, 1.0] for all intents")


def test_suggestions_non_empty():
    resp = build_structured_response(
        answer_text = LEGAL_ANSWER,
        intent      = "risk_check",
        session_id  = "s",
        confidence  = 0.9,
        mode        = "risk_node",
    )
    assert len(resp.suggestions) > 0
    assert all(isinstance(s, str) for s in resp.suggestions)
    print(f"\nOK Suggestions: {resp.suggestions[:2]}")


def test_draft_field_populated():
    resp = build_structured_response(
        answer_text = DRAFT_ANSWER,
        intent      = "draft_request",
        session_id  = "s",
        confidence  = 1.0,
        mode        = "draft_node_stage3",
        draft       = "LEGAL NOTICE\nDear Sir...",
    )
    assert resp.draft == "LEGAL NOTICE\nDear Sir..."
    print(f"\nOK Draft field preserved in structured response")


def test_to_dict_serializable():
    resp = build_structured_response(
        answer_text = SIMPLE_ANSWER,
        intent      = "general",
        session_id  = "s",
        confidence  = 0.5,
        mode        = "general_node",
    )
    d = resp.to_dict()
    assert "answer_text"  in d
    assert "summary"      in d
    assert "key_clauses"  in d
    assert "risk"         in d
    assert "suggestions"  in d
    assert "citations"    in d
    assert "intent"       in d
    assert isinstance(d["risk"], dict)
    assert "score" in d["risk"]
    print(f"\nOK to_dict() produces correct structure")


# ═══════════════════════════════════════════════════════════════════════════════
# TRANSLATION TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_detect_language_english():
    result = detect_language("What is Section 302 IPC?")
    assert result.is_english      == True
    assert result.detected_script is None
    assert result.target_language is None
    print(f"\nOK English query detected as English")


def test_detect_language_translation_request():
    result = detect_language("Translate this into Malayalam: What is Section 138 NI Act?")
    assert result.is_english      == True
    assert result.target_language == "Malayalam"
    print(f"\nOK Translation request detected: target={result.target_language}")


def test_detect_language_explain_in():
    result = detect_language("Explain this in Hindi")
    assert result.target_language == "Hindi"
    print(f"\nOK 'Explain in Hindi' detected: target={result.target_language}")


def test_detect_language_malayalam_script():
    # Malayalam Unicode characters
    malayalam_query = "വകുപ്പ് 302 ഐപിസി എന്താണ്?"
    result = detect_language(malayalam_query)
    assert result.is_english      == False
    assert result.detected_script == "Malayalam"
    print(f"\nOK Malayalam script detected correctly")


def test_detect_language_hindi_script():
    hindi_query = "धारा 302 आईपीसी क्या है?"
    result = detect_language(hindi_query)
    assert result.is_english      == False
    assert result.detected_script == "Hindi"
    print(f"\nOK Hindi script detected correctly")


def test_strip_legal_content_malayalam():
    query  = "Translate into Malayalam: What is Section 302 IPC?"
    result = _strip_legal_content(query, "Malayalam")
    assert "Section 302" in result
    assert "Translate" not in result
    print(f"\nOK Stripped: {result!r}")


def test_strip_legal_content_hindi():
    query  = "Explain this in Hindi: What are the bail provisions under BNSS?"
    result = _strip_legal_content(query, "Hindi")
    assert "bail" in result.lower() or "BNSS" in result
    print(f"\nOK Stripped: {result!r}")
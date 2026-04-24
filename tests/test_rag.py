import pytest
from rag.pipeline import rag_pipeline, RAGResponse, preprocess_query
 
 
def test_rag_returns_response_object():
    result = rag_pipeline.answer("What is IPC section 420?")
    assert isinstance(result, RAGResponse)
    assert len(result.answer) > 50
 
 
def test_rag_answer_has_citations():
    result = rag_pipeline.answer("What is the punishment for murder in India?")
    assert result.context_used is True
    assert len(result.citations) > 0
    for c in result.citations:
        assert "source" in c
        assert 0 <= c["score"] <= 1
 
 
def test_rag_answer_mentions_law():
    result      = rag_pipeline.answer("What is cheating under Indian law?")
    answer_lower = result.answer.lower()
    assert any(kw in answer_lower for kw in ["ipc", "bns", "section", "act", "penal"]), \
        "Answer did not reference any law"
 
 
def test_rag_statute_filter():
    result = rag_pipeline.answer(
        "What are the rights of a tenant?", doc_type_filter="statute"
    )
    for c in result.citations:
        assert c["doc_type"] == "statute"
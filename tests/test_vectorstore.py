"""Tests for the embedding and vector store pipeline."""

import pytest


def test_embedder_vector_shape():
    """Embedder should return 384-dimensional vectors."""
    from rag.embedder import embedder
    vectors = embedder.embed(["Indian Penal Code section 420 cheating"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384


def test_embedder_batch():
    """Embedder should handle multiple texts."""
    from rag.embedder import embedder
    texts = [
        "tenant security deposit refund",
        "bail application anticipatory bail",
        "consumer complaint deficiency of service",
    ]
    vectors = embedder.embed(texts)
    assert len(vectors) == 3
    assert all(len(v) == 384 for v in vectors)


def test_vectorstore_count_positive():
    """ChromaDB should have chunks after ingestion."""
    from rag.vectorstore import vectorstore
    count = vectorstore.count()
    assert count > 0, "ChromaDB is empty — run: python rag/ingest.py"


def test_vectorstore_search_returns_results():
    """Search should return results with required fields."""
    from rag.vectorstore import vectorstore
    if vectorstore.count() == 0:
        pytest.skip("ChromaDB empty — run ingest.py first")

    results = vectorstore.search("cheque bounce punishment", n_results=5)
    assert len(results) > 0
    for r in results:
        assert "text"     in r
        assert "source"   in r
        assert "score"    in r
        assert "doc_type" in r
        assert 0 <= r["score"] <= 1


def test_vectorstore_search_relevance():
    """Top result for a clear IPC query should be a statute."""
    from rag.vectorstore import vectorstore
    if vectorstore.count() == 0:
        pytest.skip("ChromaDB empty — run ingest.py first")

    results = vectorstore.search("section 420 cheating punishment IPC", n_results=5)
    doc_types = [r["doc_type"] for r in results]
    assert "statute" in doc_types, "Expected at least one statute in top-5 results"


def test_vectorstore_filter_by_doc_type():
    """doc_type_filter should restrict results correctly."""
    from rag.vectorstore import vectorstore
    if vectorstore.count() == 0:
        pytest.skip("ChromaDB empty — run ingest.py first")

    results = vectorstore.search(
        "punishment for theft", n_results=5, doc_type_filter="statute"
    )
    for r in results:
        assert r["doc_type"] == "statute"
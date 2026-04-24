import pytest
 
 
def test_embedder_vector_shape():
    from rag.embedder import embedder
    vectors = embedder.embed(["Indian Penal Code section 420 cheating"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 384
 
 
def test_embedder_batch():
    from rag.embedder import embedder
    texts   = ["tenant deposit", "bail application", "consumer complaint"]
    vectors = embedder.embed(texts)
    assert len(vectors) == 3
    assert all(len(v) == 384 for v in vectors)
 
 
def test_vectorstore_count_positive():
    from rag.vectorstore import vectorstore
    assert vectorstore.count() > 0, "ChromaDB empty — run: python rag/ingest.py"
 
 
def test_vectorstore_search_returns_results():
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
 
 
def test_vectorstore_filter_by_doc_type():
    from rag.vectorstore import vectorstore
    if vectorstore.count() == 0:
        pytest.skip("ChromaDB empty — run ingest.py first")
    results = vectorstore.search(
        "punishment for theft", n_results=5, doc_type_filter="statute"
    )
    for r in results:
        assert r["doc_type"] == "statute"
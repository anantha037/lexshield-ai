import pytest
from fastapi.testclient import TestClient
from api.main import app
 
client = TestClient(app)
 
LEGAL_QUERIES = [
    "What are my rights if my landlord refuses to return my deposit?",
    "What is the punishment for cheating under Indian law?",
    "Can I get anticipatory bail if I fear arrest?",
    "My employer has not paid my salary. What can I do?",
    "What are my rights as a consumer if a product is defective?",
]
 
 
@pytest.mark.parametrize("query", LEGAL_QUERIES)
def test_end_to_end_legal_query(query):
    """Each query should return 200 with an answer and at least one citation."""
    response = client.post("/api/v1/legal/query", json={"query": query})
    assert response.status_code == 200
    data = response.json()
    assert len(data["answer"]) > 50,   "Answer too short"
    assert isinstance(data["citations"], list)
 
 
def test_orchestrate_text_flow():
    response = client.post(
        "/api/v1/orchestrate/query",
        json={"query": "What is anticipatory bail?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["flow"] == "text_query"
    assert len(data["answer"]) > 50
 
 
def test_health_all_services_reachable():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    # At minimum ChromaDB should be ok
    assert "ok" in data["chromadb"]
 
 
def test_disclaimer_in_answer():
    """Every legal answer must contain the disclaimer."""
    response = client.post(
        "/api/v1/legal/query",
        json={"query": "What is the punishment for theft in India?"},
    )
    assert response.status_code == 200
    answer = response.json()["answer"].lower()
    assert any(phrase in answer for phrase in [
        "not legal advice",
        "consult a qualified lawyer",
        "legal information",
    ]), "Disclaimer missing from answer"

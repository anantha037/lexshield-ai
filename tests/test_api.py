from fastapi.testclient import TestClient
from api.main import app
 
client = TestClient(app)
 
 
def test_health_check_returns_200():
    response = client.get("/health")
    assert response.status_code == 200
 
 
def test_health_check_has_required_fields():
    response = client.get("/health")
    data = response.json()
    assert "service"  in data
    assert "chromadb" in data
    assert "llm"      in data
    assert "embedder" in data
    assert "overall"  in data
 
 
def test_legal_query_rejects_empty():
    response = client.post("/api/v1/legal/query", json={"query": ""})
    assert response.status_code == 400
 
 
def test_orchestrate_query_rejects_empty():
    response = client.post("/api/v1/orchestrate/query", json={"query": ""})
    assert response.status_code == 400
 
 
def test_document_analyze_rejects_invalid_type():
    response = client.post(
        "/api/v1/document/analyze",
        files={"file": ("test.txt", b"some content", "text/plain")},
    )
    assert response.status_code == 415
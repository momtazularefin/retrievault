from fastapi.testclient import TestClient
from retrievault.api import app

client = TestClient(app)

def test_health_check_returns_schema():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "qdrant" in data
    assert "model" in data
    assert "corpus" in data
    
    assert data["model"] == "claude-sonnet-4-6"
    assert data["corpus"]["repo"] == "fastapi/fastapi"
    assert data["corpus"]["commit_tag"] == "0.136.3"

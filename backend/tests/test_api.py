from fastapi.testclient import TestClient
import pytest

from retrievault.api import app
from retrievault.config import Settings


@pytest.fixture
def client(monkeypatch):
    test_settings = Settings(_env_file=None)
    monkeypatch.setattr("retrievault.api.get_settings", lambda: test_settings)
    return TestClient(app)


def test_health_check_returns_schema(client, monkeypatch):
    class MockQdrantClient:
        def __init__(self, url, api_key):
            assert url == "http://localhost:6333"
            assert api_key is None

        def get_collections(self):
            return []

    monkeypatch.setattr("retrievault.api.QdrantClient", MockQdrantClient)

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    
    assert "status" in data
    assert "qdrant" in data
    assert "model" in data
    assert "corpus" in data
    
    assert data["qdrant"] is True
    assert data["model"] == "claude-sonnet-4-6"
    assert data["corpus"]["repo"] == "fastapi/fastapi"
    assert data["corpus"]["commit_tag"] == "0.136.3"


def test_query_endpoint(client, monkeypatch):
    # Mock searcher
    class MockSearcher:
        def search(self, query):
            from retrievault.retrieve.hybrid_search import RetrievedChunk
            return [
                RetrievedChunk(
                    point_id="123",
                    score=0.9,
                    file_path="foo.py",
                    symbol_name="foo",
                    symbol_type="function",
                    start_line=1,
                    end_line=10,
                    code="def foo(): pass",
                    github_url="https://github.com/fastapi/fastapi/blob/0.136.3/foo.py"
                )
            ]
    monkeypatch.setattr("retrievault.api.get_searcher", lambda: MockSearcher())
    
    # Mock reranker
    def mock_rerank(query, chunks, top_k=None):
        return chunks[:top_k]

    monkeypatch.setattr("retrievault.api.rerank", mock_rerank)
    
    # Mock LangGraph
    class MockGraph:
        async def ainvoke(self, state):
            return {
                "answer": "This is a mocked answer.",
                "citations": [{"label": "[S1]", "file_path": "foo.py"}],
                "input_tokens": 100,
                "output_tokens": 50
            }
            
    monkeypatch.setattr("retrievault.api.get_graph", lambda: MockGraph())
    
    # Make the request
    response = client.post("/query", json={"question": "What is foo?", "top_k": 1})
    assert response.status_code == 200
    
    data = response.json()
    assert data["answer"] == "This is a mocked answer."
    assert len(data["citations"]) == 1

    meta = data["metadata"]
    assert meta["tokens"]["input"] == 100
    assert meta["tokens"]["output"] == 50
    assert "latency_ms" in meta
    assert "total" in meta["latency_ms"]
    assert meta["model"] == "claude-sonnet-4-6"
    assert meta["retrieved_chunk_ids"] == ["123"]
    assert meta["est_cost_usd"] > 0.001


def test_query_endpoint_accepts_legacy_query_field(client, monkeypatch):
    class MockSearcher:
        def search(self, query):
            assert query == "What is foo?"
            return []

    class MockGraph:
        async def ainvoke(self, state):
            return {"answer": "No context.", "citations": []}

    monkeypatch.setattr("retrievault.api.get_searcher", lambda: MockSearcher())
    monkeypatch.setattr("retrievault.api.rerank", lambda q, chunks, top_k=None: chunks)
    monkeypatch.setattr("retrievault.api.get_graph", lambda: MockGraph())

    response = client.post("/query", json={"query": "What is foo?"})
    assert response.status_code == 200

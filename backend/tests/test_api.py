import pytest
from fastapi.testclient import TestClient
from qdrant_client.http.exceptions import ResponseHandlingException

from retrievault import api
from retrievault.collection import MANIFEST_POINT_ID
from retrievault.config import Settings
from retrievault.retrieve.hybrid_search import RetrievedChunk


@pytest.fixture
def settings(monkeypatch):
    test_settings = Settings(_env_file=None)
    monkeypatch.setattr("retrievault.api.get_settings", lambda: test_settings)
    return test_settings


@pytest.fixture
def client(settings):
    return TestClient(api.app)


def fake_qdrant(points: int, manifest: dict | None):
    class Record:
        payload = manifest

    class FakeQdrantClient:
        def __init__(self, url, api_key, timeout):
            assert url == "http://localhost:6333"
            assert api_key is None

        def get_collections(self):
            return []

        def collection_exists(self, name):
            return True

        def count(self, name, exact):
            return type("Count", (), {"count": points})()

        def retrieve(self, name, ids):
            assert ids == [MANIFEST_POINT_ID]
            return [Record()] if manifest else []

    return FakeQdrantClient


MANIFEST = {"repo": "fastapi/fastapi", "commit_tag": "0.136.3", "chunk_count": 3, "chunker_version": "2"}


def test_health_is_ok_when_the_manifest_matches_the_point_count(client, monkeypatch):
    monkeypatch.setattr(api, "QdrantClient", fake_qdrant(points=4, manifest=MANIFEST))

    data = client.get("/health").json()

    assert data["status"] == "ok"
    assert data["qdrant"] is True and data["index_complete"] is True
    assert data["model"] == "claude-sonnet-4-6"
    assert data["corpus"]["commit_tag"] == "0.136.3"
    assert data["corpus"]["chunk_count"] == 3
    assert data["corpus"]["chunker_version"] == "2"


def test_health_is_degraded_for_a_partial_index(client, monkeypatch):
    monkeypatch.setattr(api, "QdrantClient", fake_qdrant(points=2, manifest=None))

    data = client.get("/health").json()

    assert data["status"] == "degraded"
    assert data["qdrant"] is True and data["index_complete"] is False
    assert data["corpus"]["chunk_count"] is None


def retrieved(code="def foo(): pass"):
    return RetrievedChunk(
        point_id="123",
        score=0.9,
        file_path="fastapi/foo.py",
        symbol_name="foo",
        symbol_type="function",
        start_line=1,
        end_line=10,
        code=code,
        github_url="https://github.com/fastapi/fastapi/blob/0.136.3/fastapi/foo.py#L1-L10",
    )


def install_pipeline(monkeypatch, graph_result, chunks=None):
    class MockSearcher:
        def search(self, query):
            return [retrieved()] if chunks is None else chunks

    class MockGraph:
        async def ainvoke(self, state):
            self.state = state
            return graph_result

    graph = MockGraph()
    monkeypatch.setattr(api, "get_searcher", lambda: MockSearcher())
    monkeypatch.setattr(api, "rerank", lambda q, c, top_k=None: [dict(x, rerank_score=0.8) for x in c])
    monkeypatch.setattr(api, "get_graph", lambda: graph)
    return graph


def test_query_returns_answer_grounding_and_cost_from_all_token_types(client, monkeypatch):
    graph = install_pipeline(
        monkeypatch,
        {
            "answer": "Foo [S1].",
            "citations": [{"label": "[S1]", "file_path": "fastapi/foo.py"}],
            "refused": False,
            "grounding": "grounded",
            "invalid_labels": [],
            "retries": 0,
            "llm_calls": 1,
            "input_tokens": 1_000_000,
            "output_tokens": 100_000,
            "cache_creation_input_tokens": 1_000_000,
            "cache_read_input_tokens": 1_000_000,
        },
    )

    response = client.post("/query", json={"question": "  What is foo?  ", "top_k": 1})

    assert response.status_code == 200
    data = response.json()
    assert graph.state["question"] == "What is foo?"
    assert "text" in graph.state["chunks"][0], "reranked chunks carry the header text"
    assert data["answer"] == "Foo [S1]."
    assert data["refused"] is False
    assert data["grounding"] == {"status": "grounded", "invalid_labels": [], "retries": 0}
    meta = data["metadata"]
    assert meta["tokens"] == {
        "input": 1_000_000,
        "output": 100_000,
        "cache_creation": 1_000_000,
        "cache_read": 1_000_000,
    }
    # $3 input + $1.50 output + $3.75 cache writes (1.25x) + $0.30 cache reads (0.1x).
    assert meta["est_cost_usd"] == pytest.approx(8.55)
    assert meta["retrieved_chunk_ids"] == ["123"]
    assert meta["retrieved_chunks"][0]["rerank_score"] == 0.8
    assert set(meta["latency_ms"]) == {"retrieve", "rerank", "synthesize", "total"}


def test_query_accepts_legacy_query_field(client, monkeypatch):
    graph = install_pipeline(monkeypatch, {"answer": "x", "citations": []}, chunks=[])

    response = client.post("/query", json={"query": "What is foo?"})

    assert response.status_code == 200
    assert graph.state["question"] == "What is foo?"


@pytest.mark.parametrize(
    "body",
    [{"question": "   "}, {"question": "x" * 2001}, {"question": "ok", "top_k": 0}, {"question": "ok", "top_k": 21}, {}],
)
def test_query_rejects_invalid_requests(client, body):
    assert client.post("/query", json=body).status_code == 422


def test_query_maps_an_unreachable_vector_store_to_503(client, monkeypatch):
    class DownSearcher:
        def search(self, query):
            raise ResponseHandlingException(ConnectionError("refused"))

    monkeypatch.setattr(api, "get_searcher", lambda: DownSearcher())

    assert client.post("/query", json={"question": "What is foo?"}).status_code == 503


def test_rerank_can_be_switched_off_and_the_fused_top_k_is_used(client, settings, monkeypatch):
    settings.rerank_enabled = False
    settings.top_k_rerank = 1
    graph = install_pipeline(monkeypatch, {"answer": "a", "citations": []}, chunks=[retrieved(), retrieved("def bar(): pass")])
    monkeypatch.setattr(api, "rerank", lambda *a, **k: pytest.fail("reranker must not run"))

    response = client.post("/query", json={"question": "What is foo?"})

    assert response.status_code == 200
    assert len(graph.state["chunks"]) == 1


def test_retrieved_chunks_expose_the_text_the_model_received(client, monkeypatch):
    install_pipeline(monkeypatch, {"answer": "a", "citations": []})

    chunk = client.post("/query", json={"question": "What is foo?"}).json()["metadata"][
        "retrieved_chunks"
    ][0]

    assert chunk["content"] == "def foo(): pass"
    assert chunk["text"] == "fastapi/foo.py :: foo\ndef foo(): pass"

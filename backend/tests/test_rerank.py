from retrievault.rerank.reranker import rerank

def test_reranker_reorders_known_case(monkeypatch):
    query = "How to declare a path parameter in FastAPI?"

    class FakeReranker:
        def predict(self, pairs):
            return [0.1, 0.95, 0.2]

    monkeypatch.setattr("retrievault.rerank.reranker.get_reranker", lambda: FakeReranker())

    # doc1 is irrelevant, doc2 is highly relevant
    doc1 = {"id": "1", "code": "def solve_math_problem(a, b):\n    return a + b"}
    doc2 = {
        "id": "2",
        "code": (
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n"
            "@app.get('/items/{item_id}')\n"
            "def read_item(item_id: int):\n"
            "    return item_id"
        ),
    }
    doc3 = {"id": "3", "code": "print('Hello world!')"}

    # Pass them in a suboptimal order
    chunks = [doc1, doc2, doc3]

    reranked = rerank(query, chunks, top_k=2)

    assert len(reranked) == 2
    # The relevant doc should be pulled to the top
    assert reranked[0]["id"] == "2"
    assert "rerank_score" in reranked[0]
    assert "rerank_score" not in doc2


def test_reranker_zero_top_k_returns_no_chunks(monkeypatch):
    class FakeReranker:
        def predict(self, pairs):
            raise AssertionError("reranker should not run when top_k is zero")

    monkeypatch.setattr("retrievault.rerank.reranker.get_reranker", lambda: FakeReranker())

    chunks = [{"id": "1", "code": "def route(): pass"}]

    assert rerank("How does routing work?", chunks, top_k=0) == []

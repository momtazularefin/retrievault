import pytest

from retrievault.rerank.reranker import _sigmoid, rerank


class FakeCrossEncoder:
    def __init__(self, logits):
        self.logits = logits
        self.documents = None

    def rerank(self, query, documents):
        self.documents = list(documents)
        return iter(self.logits)


def install(monkeypatch, logits):
    fake = FakeCrossEncoder(logits)
    monkeypatch.setattr("retrievault.rerank.reranker.get_reranker", lambda: fake)
    return fake


def test_reranker_orders_by_score_and_keeps_top_k(monkeypatch):
    install(monkeypatch, [-2.0, 3.0, 0.5])
    chunks = [{"id": "1", "code": "a"}, {"id": "2", "code": "b"}, {"id": "3", "code": "c"}]

    reranked = rerank("query", chunks, top_k=2)

    assert [c["id"] for c in reranked] == ["2", "3"]
    assert 0 < reranked[1]["rerank_score"] < reranked[0]["rerank_score"] < 1
    assert "rerank_score" not in chunks[1], "input chunks are not mutated"


def test_reranker_scores_header_text_when_present(monkeypatch):
    fake = install(monkeypatch, [1.0, 1.0])
    chunks = [{"code": "def a(): ...", "text": "fastapi/a.py :: a\ndef a(): ..."}, {"code": "b"}]

    rerank("query", chunks, top_k=2)

    assert fake.documents == ["fastapi/a.py :: a\ndef a(): ...", "b"]


def test_reranker_zero_top_k_does_not_run_the_model(monkeypatch):
    class Exploding:
        def rerank(self, query, documents):
            raise AssertionError("reranker should not run when top_k is zero")

    monkeypatch.setattr("retrievault.rerank.reranker.get_reranker", lambda: Exploding())

    assert rerank("query", [{"code": "x"}], top_k=0) == []
    assert rerank("query", [], top_k=3) == []


@pytest.mark.parametrize("logit", [-1000.0, -5.0, 0.0, 5.0, 1000.0])
def test_sigmoid_is_stable_and_bounded(logit):
    value = _sigmoid(logit)
    assert 0.0 <= value <= 1.0
    assert _sigmoid(0.0) == 0.5

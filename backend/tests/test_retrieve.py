from unittest.mock import MagicMock

import pytest
from qdrant_client.http import models

from retrievault.config import get_settings
from retrievault.retrieve.hybrid_search import HybridSearcher
from retrievault.retrieve.query_encoder import EncodedQuery, QueryEncoder


@pytest.fixture(autouse=True)
def force_no_acceleration(monkeypatch):
    monkeypatch.setenv("ACCELERATION", "none")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_query_encoding_uses_query_side_bm25_weights():
    encoded = QueryEncoder().encode("route route routing in FastAPI")

    assert len(encoded.dense) == 768
    assert len(encoded.sparse_indices) == len(encoded.sparse_values) > 0
    # Query-side BM25 is term presence: repeated words are one term with weight 1.
    assert len(set(encoded.sparse_indices)) == len(encoded.sparse_indices)
    assert set(encoded.sparse_values) == {1.0}


def searcher_with(encoded: EncodedQuery, **kwargs):
    client = MagicMock()
    client.query_points.return_value = MagicMock(points=[])
    encoder = MagicMock(spec=QueryEncoder)
    encoder.encode.return_value = encoded
    return client, HybridSearcher(client=client, encoder=encoder, **kwargs)


def test_hybrid_search_fuses_dense_and_sparse_prefetches_with_rrf():
    client, searcher = searcher_with(
        EncodedQuery(dense=[0.1] * 768, sparse_indices=[1, 2], sparse_values=[1.0, 1.0]),
        collection_name="test_col",
    )

    searcher.search("APIRouter route matching", limit=5)

    call = client.query_points.call_args.kwargs
    assert call["collection_name"] == "test_col"
    assert call["limit"] == 5
    assert isinstance(call["query"], models.FusionQuery)
    assert call["query"].fusion == models.Fusion.RRF
    assert [p.using for p in call["prefetch"]] == ["dense", "bm25"]


def test_hybrid_search_skips_sparse_prefetch_when_query_has_no_terms():
    client, searcher = searcher_with(EncodedQuery(dense=[0.1] * 768, sparse_indices=[], sparse_values=[]))

    searcher.search("???")

    assert [p.using for p in client.query_points.call_args.kwargs["prefetch"]] == ["dense"]


def test_dense_only_search_uses_dense_vector():
    dense = [0.2] * 768
    client, searcher = searcher_with(EncodedQuery(dense=dense, sparse_indices=[3], sparse_values=[1.0]))

    searcher.search_dense_only("dependency injection", limit=3)

    call = client.query_points.call_args.kwargs
    assert call["using"] == "dense" and call["query"] == dense and call["limit"] == 3


def test_sparse_only_search_uses_sparse_vector():
    client, searcher = searcher_with(
        EncodedQuery(dense=[0.1] * 768, sparse_indices=[7, 9], sparse_values=[1.0, 1.0])
    )

    searcher.search_sparse_only("OAuth2PasswordBearer", limit=4)

    call = client.query_points.call_args.kwargs
    assert call["using"] == "bm25"
    assert call["query"].indices == [7, 9]


def test_retrieved_chunk_reads_older_payloads_and_builds_header_text():
    point = MagicMock(
        id="abc",
        score=0.5,
        payload={
            "file_path": "fastapi/routing.py",
            "symbol_name": "APIRouter.get",
            "symbol_type": "method",
            "start_line": 10,
            "end_line": 20,
            "code": "    def get(self):",
        },
    )

    chunk = HybridSearcher._to_chunk(point)

    assert (chunk.context, chunk.part, chunk.part_count) == ("", 1, 1)
    assert chunk.to_dict()["text"] == "fastapi/routing.py :: APIRouter.get\n    def get(self):"


@pytest.fixture(scope="module")
def seeded_qdrant():
    import hashlib
    import os

    from fastembed import SparseTextEmbedding, TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import (
        Distance,
        Modifier,
        PointStruct,
        SparseVectorParams,
        VectorParams,
    )

    from retrievault.chunker import embedding_text

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        client = QdrantClient(url=url, timeout=5)
        client.get_collections()
    except Exception:  # noqa: BLE001 - any connection failure means "skip"
        pytest.skip("Qdrant not reachable")

    collection = "retrievault_test_m3"
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
        sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
    )

    chunks = [
        {
            "file_path": "fastapi/routing.py",
            "symbol_name": "APIRouter",
            "symbol_type": "class",
            "start_line": 10,
            "end_line": 40,
            "code": (
                "class APIRouter:\n"
                "    def add_api_route(self, path, endpoint, methods=None):\n"
                "        pass\n"
            ),
        },
        {
            "file_path": "fastapi/utils.py",
            "symbol_name": "generate_unique_id",
            "symbol_type": "function",
            "start_line": 50,
            "end_line": 60,
            "code": "def generate_unique_id(route):\n    return route.name\n",
        },
        {
            "file_path": "fastapi/security/oauth2.py",
            "symbol_name": "OAuth2PasswordBearer",
            "symbol_type": "class",
            "start_line": 100,
            "end_line": 130,
            "code": (
                "class OAuth2PasswordBearer:\n"
                "    def __init__(self, tokenUrl: str):\n"
                "        self.tokenUrl = tokenUrl\n"
            ),
        },
    ]
    texts = [embedding_text(c["file_path"], c["symbol_name"], "", c["code"]) for c in chunks]
    dense_vectors = list(TextEmbedding(model_name="BAAI/bge-base-en-v1.5").embed(texts))
    sparse_vectors = list(SparseTextEmbedding(model_name="Qdrant/bm25").embed(texts))

    points = []
    for chunk, dense, sparse in zip(chunks, dense_vectors, sparse_vectors, strict=True):
        stable = f"{chunk['file_path']}:{chunk['start_line']}:{chunk['end_line']}:0.136.3"
        points.append(
            PointStruct(
                id=hashlib.md5(stable.encode()).hexdigest(),
                vector={
                    "dense": dense.tolist(),
                    "bm25": {"indices": sparse.indices.tolist(), "values": sparse.values.tolist()},
                },
                payload=chunk,
            )
        )
    client.upsert(collection_name=collection, points=points)
    yield client, collection
    client.delete_collection(collection)


@pytest.mark.integration
def test_sparse_probe_surfaces_exact_symbol(seeded_qdrant):
    client, collection = seeded_qdrant
    results = HybridSearcher(client=client, collection_name=collection).search_sparse_only(
        "OAuth2PasswordBearer", limit=3
    )
    assert results[0].symbol_name == "OAuth2PasswordBearer"


@pytest.mark.integration
def test_dense_probe_surfaces_semantic_match(seeded_qdrant):
    client, collection = seeded_qdrant
    results = HybridSearcher(client=client, collection_name=collection).search_dense_only(
        "how are HTTP routes registered on a router object", limit=3
    )
    assert results[0].symbol_name == "APIRouter"


@pytest.mark.integration
def test_hybrid_fusion_prefers_both_signals(seeded_qdrant):
    client, collection = seeded_qdrant
    results = HybridSearcher(client=client, collection_name=collection).search(
        "APIRouter add_api_route", limit=3
    )
    assert results[0].symbol_name == "APIRouter"
    assert results[0].score >= results[-1].score

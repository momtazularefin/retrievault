from unittest.mock import MagicMock

import pytest
from qdrant_client.http import models

from retrievault.retrieve.hybrid_search import HybridSearcher
from retrievault.retrieve.query_encoder import EncodedQuery, QueryEncoder


def test_encoded_query_has_expected_shapes():
    encoder = QueryEncoder()
    encoded = encoder.encode("how does routing work in FastAPI")

    assert len(encoded.dense) == 768
    assert len(encoded.sparse_indices) == len(encoded.sparse_values)
    assert len(encoded.sparse_indices) > 0


def test_hybrid_search_uses_rrf_prefetch():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response

    encoder = MagicMock(spec=QueryEncoder)
    encoder.encode.return_value = EncodedQuery(
        dense=[0.1] * 768,
        sparse_indices=[1, 2],
        sparse_values=[0.5, 0.3],
    )

    searcher = HybridSearcher(client=mock_client, encoder=encoder, collection_name="test_col")
    searcher.search("APIRouter route matching", limit=5)

    mock_client.query_points.assert_called_once()
    call_kwargs = mock_client.query_points.call_args.kwargs
    assert call_kwargs["collection_name"] == "test_col"
    assert call_kwargs["limit"] == 5
    assert isinstance(call_kwargs["query"], models.FusionQuery)
    assert call_kwargs["query"].fusion == models.Fusion.RRF
    assert len(call_kwargs["prefetch"]) == 2
    assert call_kwargs["prefetch"][0].using == "dense"
    assert call_kwargs["prefetch"][1].using == "bm25"


def test_dense_only_search_uses_dense_vector():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response

    encoder = MagicMock(spec=QueryEncoder)
    dense = [0.2] * 768
    encoder.encode.return_value = EncodedQuery(
        dense=dense,
        sparse_indices=[3],
        sparse_values=[0.9],
    )

    searcher = HybridSearcher(client=mock_client, encoder=encoder)
    searcher.search_dense_only("dependency injection", limit=3)

    call_kwargs = mock_client.query_points.call_args.kwargs
    assert call_kwargs["using"] == "dense"
    assert call_kwargs["query"] == dense
    assert call_kwargs["limit"] == 3


def test_sparse_only_search_uses_sparse_vector():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.points = []
    mock_client.query_points.return_value = mock_response

    encoder = MagicMock(spec=QueryEncoder)
    encoder.encode.return_value = EncodedQuery(
        dense=[0.1] * 768,
        sparse_indices=[7, 9],
        sparse_values=[0.4, 0.6],
    )

    searcher = HybridSearcher(client=mock_client, encoder=encoder)
    searcher.search_sparse_only("OAuth2PasswordBearer", limit=4)

    call_kwargs = mock_client.query_points.call_args.kwargs
    assert call_kwargs["using"] == "bm25"
    sparse = call_kwargs["query"]
    assert sparse.indices == [7, 9]
    assert sparse.values == [0.4, 0.6]


@pytest.fixture(scope="module")
def seeded_qdrant():
    import hashlib
    import os

    from fastembed import SparseTextEmbedding, TextEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, Modifier, PointStruct, SparseVectorParams, VectorParams

    url = os.getenv("QDRANT_URL", "http://localhost:6333")
    try:
        client = QdrantClient(url=url, timeout=5)
        client.get_collections()
    except Exception:
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
            "github_url": "https://github.com/fastapi/fastapi/blob/0.136.3/fastapi/routing.py#L10-L40",
        },
        {
            "file_path": "fastapi/utils.py",
            "symbol_name": "generate_unique_id",
            "symbol_type": "function",
            "start_line": 50,
            "end_line": 60,
            "code": "def generate_unique_id(route):\n    return route.name\n",
            "github_url": "https://github.com/fastapi/fastapi/blob/0.136.3/fastapi/utils.py#L50-L60",
        },
        {
            "file_path": "fastapi/security/oauth2.py",
            "symbol_name": "OAuth2PasswordBearer",
            "symbol_type": "class",
            "start_line": 100,
            "end_line": 130,
            "code": "class OAuth2PasswordBearer:\n    def __init__(self, tokenUrl: str):\n        self.tokenUrl = tokenUrl\n",
            "github_url": "https://github.com/fastapi/fastapi/blob/0.136.3/fastapi/security/oauth2.py#L100-L130",
        },
    ]

    dense_model = TextEmbedding(model_name="BAAI/bge-base-en-v1.5")
    sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")

    texts = [c["code"] for c in chunks]
    dense_vectors = list(dense_model.embed(texts))
    sparse_vectors = list(sparse_model.embed(texts))

    points = []
    for i, chunk in enumerate(chunks):
        stable = f"{chunk['file_path']}:{chunk['start_line']}:{chunk['end_line']}:0.136.3"
        point_id = hashlib.md5(stable.encode()).hexdigest()
        sparse = sparse_vectors[i]
        points.append(
            PointStruct(
                id=point_id,
                vector={
                    "dense": dense_vectors[i].tolist(),
                    "bm25": {
                        "indices": sparse.indices.tolist(),
                        "values": sparse.values.tolist(),
                    },
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
    searcher = HybridSearcher(client=client, collection_name=collection)

    results = searcher.search_sparse_only("OAuth2PasswordBearer", limit=3)
    assert results
    assert results[0].symbol_name == "OAuth2PasswordBearer"


@pytest.mark.integration
def test_dense_probe_surfaces_semantic_match(seeded_qdrant):
    client, collection = seeded_qdrant
    searcher = HybridSearcher(client=client, collection_name=collection)

    results = searcher.search_dense_only(
        "how are HTTP routes registered on a router object", limit=3
    )
    assert results
    assert results[0].symbol_name == "APIRouter"


@pytest.mark.integration
def test_hybrid_fusion_prefers_both_signals(seeded_qdrant):
    client, collection = seeded_qdrant
    searcher = HybridSearcher(client=client, collection_name=collection)

    results = searcher.search("APIRouter add_api_route", limit=3)
    assert results
    assert results[0].symbol_name == "APIRouter"
    assert results[0].score >= results[-1].score

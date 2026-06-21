from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from retrievault.collection import COLLECTION_NAME
from retrievault.config import get_settings
from retrievault.retrieve.query_encoder import EncodedQuery, QueryEncoder


@dataclass(frozen=True)
class RetrievedChunk:
    point_id: str
    score: float
    file_path: str
    symbol_name: str
    symbol_type: str
    start_line: int
    end_line: int
    code: str
    github_url: str


class HybridSearcher:
    def __init__(
        self,
        client: QdrantClient | None = None,
        encoder: QueryEncoder | None = None,
        collection_name: str = COLLECTION_NAME,
    ):
        settings = get_settings()
        self._client = client or QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key
        )
        self._encoder = encoder or QueryEncoder()
        self._collection = collection_name
        self._prefetch_limit = settings.prefetch_limit
        self._top_n = settings.top_n_fusion

    def search(self, query: str, limit: int | None = None) -> list[RetrievedChunk]:
        encoded = self._encoder.encode(query)
        return self._search_encoded(encoded, limit=limit or self._top_n, fusion=True)

    def search_dense_only(self, query: str, limit: int = 10) -> list[RetrievedChunk]:
        encoded = self._encoder.encode(query)
        return self._search_encoded(encoded, limit=limit, fusion=False, vector_name="dense")

    def search_sparse_only(self, query: str, limit: int = 10) -> list[RetrievedChunk]:
        encoded = self._encoder.encode(query)
        return self._search_encoded(encoded, limit=limit, fusion=False, vector_name="bm25")

    def _search_encoded(
        self,
        encoded: EncodedQuery,
        limit: int,
        fusion: bool,
        vector_name: str | None = None,
    ) -> list[RetrievedChunk]:
        if fusion:
            response = self._client.query_points(
                collection_name=self._collection,
                prefetch=[
                    models.Prefetch(
                        query=encoded.dense,
                        using="dense",
                        limit=self._prefetch_limit,
                    ),
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=encoded.sparse_indices,
                            values=encoded.sparse_values,
                        ),
                        using="bm25",
                        limit=self._prefetch_limit,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
            if vector_name == "dense":
                query_vec: Any = encoded.dense
            else:
                query_vec = models.SparseVector(
                    indices=encoded.sparse_indices,
                    values=encoded.sparse_values,
                )
            response = self._client.query_points(
                collection_name=self._collection,
                query=query_vec,
                using=vector_name,
                limit=limit,
                with_payload=True,
            )

        return [self._to_chunk(point) for point in response.points]

    @staticmethod
    def _to_chunk(point) -> RetrievedChunk:
        payload = point.payload or {}
        return RetrievedChunk(
            point_id=str(point.id),
            score=point.score,
            file_path=payload["file_path"],
            symbol_name=payload["symbol_name"],
            symbol_type=payload["symbol_type"],
            start_line=payload["start_line"],
            end_line=payload["end_line"],
            code=payload["code"],
            github_url=payload.get("github_url", ""),
        )

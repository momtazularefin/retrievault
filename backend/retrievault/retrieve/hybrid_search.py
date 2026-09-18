from dataclasses import asdict, dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models

from retrievault.chunker import embedding_text
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
    context: str = ""
    part: int = 1
    part_count: int = 1

    @property
    def text(self) -> str:
        return embedding_text(self.file_path, self.symbol_name, self.context, self.code)

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "text": self.text}


class HybridSearcher:
    def __init__(
        self,
        client: QdrantClient | None = None,
        encoder: QueryEncoder | None = None,
        collection_name: str = COLLECTION_NAME,
    ):
        settings = get_settings()
        self._client = client or QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key_or_none,
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
        sparse = models.SparseVector(
            indices=encoded.sparse_indices,
            values=encoded.sparse_values,
        )
        if fusion:
            prefetch = [
                models.Prefetch(query=encoded.dense, using="dense", limit=self._prefetch_limit)
            ]
            # A query made only of punctuation or stopwords has no BM25 terms to search with.
            if encoded.sparse_indices:
                prefetch.append(
                    models.Prefetch(query=sparse, using="bm25", limit=self._prefetch_limit)
                )
            # Qdrant's RRF scores a point as the sum over prefetches of 1 / (k + rank), with
            # k = 2 and rank counted from 0, so it rewards agreement near the top of both lists.
            response = self._client.query_points(
                collection_name=self._collection,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
                with_payload=True,
            )
        else:
            query_vec: Any = encoded.dense if vector_name == "dense" else sparse
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
            context=payload.get("context", ""),
            part=payload.get("part", 1),
            part_count=payload.get("part_count", 1),
        )

from dataclasses import dataclass

from fastembed import SparseTextEmbedding, TextEmbedding

from retrievault.config import get_settings
from retrievault.encoders import load_dense_encoder, load_sparse_encoder


@dataclass(frozen=True)
class EncodedQuery:
    dense: list[float]
    sparse_indices: list[int]
    sparse_values: list[float]


class QueryEncoder:
    def __init__(
        self,
        dense_model: TextEmbedding | None = None,
        sparse_model: SparseTextEmbedding | None = None,
    ):
        settings = get_settings()
        self._dense = dense_model or load_dense_encoder(settings)
        self._sparse = sparse_model or load_sparse_encoder(settings)

    def encode(self, query: str) -> EncodedQuery:
        # query_embed, not embed: fastembed's BM25 weights the query side as plain term presence
        # (the document side carries the TF saturation and length normalisation), and query
        # embedding is the documented entry point for BGE queries.
        dense_vec = next(iter(self._dense.query_embed(query)))
        sparse_vec = next(iter(self._sparse.query_embed(query)))
        return EncodedQuery(
            dense=dense_vec.tolist(),
            sparse_indices=sparse_vec.indices.tolist(),
            sparse_values=sparse_vec.values.astype(float).tolist(),
        )

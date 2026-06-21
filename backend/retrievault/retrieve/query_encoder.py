from dataclasses import dataclass

from fastembed import SparseTextEmbedding, TextEmbedding

from retrievault.config import get_settings


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
        self._dense = dense_model or TextEmbedding(model_name=settings.embed_model)
        self._sparse = sparse_model or SparseTextEmbedding(model_name=settings.sparse_model)

    def encode(self, query: str) -> EncodedQuery:
        dense_vec = list(self._dense.embed([query]))[0]
        sparse_vec = list(self._sparse.embed([query]))[0]
        return EncodedQuery(
            dense=dense_vec.tolist(),
            sparse_indices=sparse_vec.indices.tolist(),
            sparse_values=sparse_vec.values.tolist(),
        )

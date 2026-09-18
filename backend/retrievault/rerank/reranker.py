import math
from functools import lru_cache
from typing import Any

from retrievault.config import get_settings
from retrievault.encoders import load_cross_encoder


def _sigmoid(logit: float) -> float:
    # Numerically stable in both directions; maps the cross-encoder logit onto (0, 1).
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exponent = math.exp(logit)
    return exponent / (1.0 + exponent)


@lru_cache
def get_reranker():
    """The BAAI/bge-reranker-base cross-encoder, loaded once from fastembed's ONNX export."""
    return load_cross_encoder(get_settings())


def rerank(query: str, chunks: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
    """Score (query, chunk) pairs with the cross-encoder and keep the best ``top_k``.

    Each chunk is scored on its ``text`` (location header, context, and code) when present,
    otherwise on ``code``. The returned chunks are copies carrying ``rerank_score`` in (0, 1).
    """
    if not chunks:
        return []
    settings = get_settings()
    top_k = settings.top_k_rerank if top_k is None else top_k
    if top_k <= 0:
        return []

    documents = [chunk.get("text") or chunk["code"] for chunk in chunks]
    logits = list(get_reranker().rerank(query, documents))

    scored = []
    for chunk, logit in zip(chunks, logits, strict=True):
        copy = dict(chunk)
        copy["rerank_score"] = _sigmoid(float(logit))
        scored.append(copy)
    scored.sort(key=lambda chunk: chunk["rerank_score"], reverse=True)
    return scored[:top_k]

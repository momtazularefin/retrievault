"""Loading of the three local models: dense embedder, BM25 sparse encoder, cross-encoder.

All three come from fastembed, so indexing and querying share one model cache and one
acceleration policy. BM25 is plain tokenisation and hashing with no ONNX session, so
acceleration applies only to the dense embedder and the cross-encoder.
"""

from fastembed import SparseTextEmbedding, TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder

from retrievault.acceleration import get_onnx_providers, verify_session
from retrievault.config import Settings


def _session_of(fastembed_model):
    """The ONNX Runtime session inside a fastembed model, or None if it has none."""
    inner = getattr(fastembed_model, "model", None)
    return getattr(inner, "model", None)


def _verified(fastembed_model, settings: Settings, model_name: str):
    session = _session_of(fastembed_model)
    if session is not None:
        verify_session(session, settings.acceleration, model_name)
    return fastembed_model


def load_dense_encoder(settings: Settings) -> TextEmbedding:
    providers = get_onnx_providers(settings.acceleration)
    model = TextEmbedding(model_name=settings.embed_model, providers=providers)
    return _verified(model, settings, settings.embed_model)


def load_sparse_encoder(settings: Settings) -> SparseTextEmbedding:
    return SparseTextEmbedding(model_name=settings.sparse_model)


def load_cross_encoder(settings: Settings) -> TextCrossEncoder:
    providers = get_onnx_providers(settings.acceleration)
    model = TextCrossEncoder(model_name=settings.rerank_model, providers=providers)
    return _verified(model, settings, settings.rerank_model)

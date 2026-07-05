import os
import math
import logging
from pathlib import Path
from typing import List, Dict, Any
from functools import lru_cache

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from retrievault.config import get_settings

logger = logging.getLogger(__name__)

class ONNXCrossEncoder:
    def __init__(self, model_dir: str | Path, acceleration: str):
        self.model_dir = str(model_dir)
        self.tokenizer = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=512)
        self.tokenizer.enable_padding()

        # Resolve ONNX providers based on acceleration config
        from retrievault.retrieve.query_encoder import get_onnx_providers
        providers = get_onnx_providers(acceleration)
        logger.info(f"ONNX Reranker loading with execution providers: {providers}")

        model_path = os.path.join(self.model_dir, "model.onnx")
        self.session = ort.InferenceSession(model_path, providers=providers)

    def predict(self, pairs: List[List[str]]) -> List[float]:
        encodings = self.tokenizer.encode_batch([(query, doc) for query, doc in pairs])
        input_names = {item.name for item in self.session.get_inputs()}
        onnx_inputs = {
            "input_ids": np.array([encoding.ids for encoding in encodings], dtype=np.int64),
            "attention_mask": np.array(
                [encoding.attention_mask for encoding in encodings], dtype=np.int64
            ),
        }
        if "token_type_ids" in input_names:
            onnx_inputs["token_type_ids"] = np.array(
                [encoding.type_ids for encoding in encodings], dtype=np.int64
            )
        onnx_inputs = {key: value for key, value in onnx_inputs.items() if key in input_names}

        outputs = self.session.run(None, onnx_inputs)
        logits = outputs[0].squeeze(-1)

        # Apply sigmoid to match SentenceTransformers CrossEncoder output scale [0, 1]
        if isinstance(logits, np.ndarray):
            if logits.ndim == 0:
                # Squeeze can reduce single-item array to 0-dim
                scores = [1 / (1 + math.exp(-float(logits)))]
            else:
                scores = [1 / (1 + math.exp(-float(x))) for x in logits]
        else:
            scores = [1 / (1 + math.exp(-float(logits)))]
        return scores


def _backend_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_model_dir(configured_model_dir: str) -> Path:
    configured = Path(configured_model_dir)
    if configured.is_absolute():
        return configured
    return _backend_root() / configured


def ensure_onnx_model(model_dir: str | Path | None = None) -> str:
    settings = get_settings()
    resolved_model_dir = Path(model_dir) if model_dir is not None else _resolve_model_dir(
        settings.rerank_model_dir
    )
    model_path = resolved_model_dir / "model.onnx"

    if not os.path.exists(model_path):
        logger.info(f"ONNX Reranker model not found at '{resolved_model_dir}'. Exporting via Optimum...")
        os.makedirs(resolved_model_dir, exist_ok=True)
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer

        # Export model from Hugging Face Hub
        model = ORTModelForSequenceClassification.from_pretrained(settings.rerank_model, export=True)
        tokenizer = AutoTokenizer.from_pretrained(settings.rerank_model)

        # Save to local cache
        model.save_pretrained(resolved_model_dir)
        tokenizer.save_pretrained(resolved_model_dir)
        logger.info("ONNX Reranker model exported and cached successfully.")

    return str(resolved_model_dir)


@lru_cache
def get_reranker() -> ONNXCrossEncoder:
    settings = get_settings()
    return ONNXCrossEncoder(ensure_onnx_model(), settings.acceleration)


def rerank(query: str, chunks: List[Dict[str, Any]], top_k: int | None = None) -> List[Dict[str, Any]]:
    """
    Rerank a list of retrieved chunks against the query using an ONNX cross-encoder.

    Args:
        query: The search query string.
        chunks: List of chunk payload dictionaries. Each must contain a "code" key.
        top_k: Number of top results to return. Defaults to settings.top_k_rerank.

    Returns:
        List of chunks sorted by rerank_score in descending order, truncated to top_k.
    """
    if not chunks:
        return []

    settings = get_settings()
    top_k = settings.top_k_rerank if top_k is None else top_k
    if top_k <= 0:
        return []

    reranker = get_reranker()

    # CrossEncoder expects a list of pairs: [(query, doc1), (query, doc2), ...]
    pairs = [[query, chunk["code"]] for chunk in chunks]

    # Get sigmoid-scaled scores
    scores = reranker.predict(pairs)

    scored_chunks = []
    for idx, chunk in enumerate(chunks):
        scored = dict(chunk)
        scored["rerank_score"] = float(scores[idx])
        scored_chunks.append(scored)

    chunks_sorted = sorted(scored_chunks, key=lambda x: x["rerank_score"], reverse=True)

    return chunks_sorted[:top_k]

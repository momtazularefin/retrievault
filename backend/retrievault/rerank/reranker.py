from typing import List, Dict, Any
from functools import lru_cache
import logging

from sentence_transformers import CrossEncoder

from retrievault.config import get_settings

logger = logging.getLogger(__name__)

@lru_cache
def get_reranker() -> CrossEncoder:
    settings = get_settings()
    logger.info(f"Loading cross-encoder: {settings.rerank_model}")
    return CrossEncoder(settings.rerank_model)

def rerank(query: str, chunks: List[Dict[str, Any]], top_k: int = None) -> List[Dict[str, Any]]:
    """
    Rerank a list of retrieved chunks against the query using a cross-encoder.
    
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
    top_k = top_k or settings.top_k_rerank
    
    reranker = get_reranker()
    
    # CrossEncoder expects a list of pairs: [(query, doc1), (query, doc2), ...]
    pairs = [[query, chunk["code"]] for chunk in chunks]
    
    # get scores
    scores = reranker.predict(pairs)
    
    # attach scores to chunks and sort
    for idx, chunk in enumerate(chunks):
        chunk["rerank_score"] = float(scores[idx])
        
    chunks_sorted = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
    
    return chunks_sorted[:top_k]

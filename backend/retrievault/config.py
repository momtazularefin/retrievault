import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""
    retrievault_synthesis_model: str = "claude-sonnet-4-6"
    
    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    
    # Corpus
    corpus_repo: str = "fastapi/fastapi"
    corpus_tag: str = "0.136.3"
    
    # Retrieval
    prefetch_limit: int = 50
    top_n_fusion: int = 30
    top_k_rerank: int = 6
    
    # Models
    embed_model: str = "BAAI/bge-base-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "BAAI/bge-reranker-base"

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "../../.env"), 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()

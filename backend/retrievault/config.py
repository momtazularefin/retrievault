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
    rerank_model_dir: str = "models/bge-reranker-onnx"
    execution_device: str = "cpu"  # "cpu", "gpu", or "npu"

    @property
    def qdrant_api_key_or_none(self) -> str | None:
        return self.qdrant_api_key.strip() or None


    # Eval defaults support clean CI/local imports without requiring `.env`.
    # The eval runner still validates provider-specific API keys before judging.
    eval_good_count: int = 3
    eval_refusal_count: int = 2
    eval_use_response_cache: bool = False
    eval_use_judge_cache: bool = False
    eval_max_workers: int = 30
    eval_concurrent_queries: int = 5
    eval_judge_provider: str = "claude"
    eval_judge_model: str = "claude-haiku-4-5-20251001"
    openai_api_key: str = ""

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(__file__), "../../.env"), 
        env_file_encoding="utf-8", 
        extra="ignore"
    )

@lru_cache
def get_settings() -> Settings:
    return Settings()

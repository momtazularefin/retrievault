from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from retrievault.collection import COLLECTION_NAME

# The repository-root .env, used for local runs; containers receive variables directly.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    # LLM
    anthropic_api_key: str = ""
    retrievault_synthesis_model: str = "claude-sonnet-4-6"
    synthesis_max_tokens: int = 2048
    # Published per-million-token prices for the synthesis model (claude-sonnet-4-6).
    synthesis_input_usd_per_mtok: float = 3.0
    synthesis_output_usd_per_mtok: float = 15.0

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = COLLECTION_NAME

    # Corpus
    corpus_repo: str = "fastapi/fastapi"
    corpus_tag: str = "0.136.3"

    # Retrieval
    prefetch_limit: int = 50
    # 12, not 30: the reranker scored no better on a deeper pool and cost 3.6x the time
    # (docs/evaluation.md, "Does the reranker earn its latency?").
    top_n_fusion: int = 12
    top_k_rerank: int = 6
    rerank_enabled: bool = True

    # Models
    embed_model: str = "BAAI/bge-base-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    rerank_model: str = "BAAI/bge-reranker-base"
    acceleration: str = "none"  # "none", "gpu", or "npu"

    # API
    cors_origins: str = "http://localhost:3000"

    # Eval defaults support clean CI/local imports without requiring `.env`.
    # The eval runner still validates provider-specific API keys before judging.
    eval_good_count: int = 45
    eval_refusal_count: int = 5
    eval_max_workers: int = 8
    eval_judge_provider: str = "claude"
    eval_judge_model: str = "claude-haiku-4-5-20251001"
    openai_api_key: str = ""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    @property
    def qdrant_api_key_or_none(self) -> str | None:
        return self.qdrant_api_key.strip() or None

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

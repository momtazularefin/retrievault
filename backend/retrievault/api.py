import hashlib
import threading
import time
from pathlib import Path
from typing import Any

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ApiException
from starlette.concurrency import run_in_threadpool

from retrievault.collection import MANIFEST_POINT_ID
from retrievault.config import get_settings
from retrievault.costing import synthesis_cost_usd
from retrievault.rerank.reranker import rerank
from retrievault.synthesize.graph import build_graph

STARTUP_TIME = time.time()
MAX_QUESTION_CHARS = 2000


def _compute_code_hash() -> str:
    """Short hash of the package's Python sources, shown in /health to identify the build."""
    hasher = hashlib.sha256()
    for file_path in sorted(Path(__file__).parent.glob("**/*.py")):
        hasher.update(file_path.read_text(encoding="utf-8").replace("\r\n", "\n").encode("utf-8"))
    return hasher.hexdigest()[:8]


BUILD_HASH = _compute_code_hash()

app = FastAPI(title="RetrieVault API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/health")
def health_check() -> dict[str, Any]:
    settings = get_settings()
    qdrant_ok = False
    manifest: dict[str, Any] | None = None
    points = None
    try:
        client = QdrantClient(
            url=settings.qdrant_url, api_key=settings.qdrant_api_key_or_none, timeout=3
        )
        client.get_collections()
        qdrant_ok = True
        if client.collection_exists(settings.qdrant_collection):
            points = client.count(settings.qdrant_collection, exact=True).count
            found = client.retrieve(settings.qdrant_collection, [MANIFEST_POINT_ID])
            manifest = found[0].payload if found else None
    except Exception:  # noqa: BLE001 - health reports failure instead of raising
        pass

    chunk_count = manifest.get("chunk_count") if manifest else None
    # The manifest is written last, so a complete index holds exactly chunk_count + 1 points.
    index_complete = chunk_count is not None and points == chunk_count + 1
    return {
        "status": "ok" if qdrant_ok and index_complete else "degraded",
        "qdrant": qdrant_ok,
        "index_complete": index_complete,
        "model": settings.retrievault_synthesis_model,
        "corpus": {
            "repo": manifest.get("repo") if manifest else settings.corpus_repo,
            "commit_tag": manifest.get("commit_tag") if manifest else settings.corpus_tag,
            "chunk_count": chunk_count,
            "chunker_version": manifest.get("chunker_version") if manifest else None,
            "indexed_at": manifest.get("indexed_at") if manifest else None,
        },
        "build_hash": BUILD_HASH,
        "startup_time": STARTUP_TIME,
    }


class QueryRequest(BaseModel):
    question: str | None = Field(default=None, max_length=MAX_QUESTION_CHARS)
    query: str | None = Field(default=None, max_length=MAX_QUESTION_CHARS)  # legacy field name
    top_k: int | None = Field(default=None, ge=1, le=20)

    @model_validator(mode="after")
    def normalize_question(self):
        if self.question is None and self.query is not None:
            self.question = self.query
        if not self.question or not self.question.strip():
            raise ValueError("question is required")
        return self


class QueryResponse(BaseModel):
    answer: str
    citations: list[dict[str, Any]]
    refused: bool
    grounding: dict[str, Any]
    metadata: dict[str, Any]


_lock = threading.Lock()
_searcher = None
_graph = None


def get_searcher():
    global _searcher
    with _lock:
        if _searcher is None:
            from retrievault.retrieve.hybrid_search import HybridSearcher

            _searcher = HybridSearcher(collection_name=get_settings().qdrant_collection)
    return _searcher


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _retrieve(question: str) -> list[dict[str, Any]]:
    return [chunk.to_dict() for chunk in get_searcher().search(question)]


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    settings = get_settings()
    question = req.question.strip()
    t_start = time.perf_counter()

    # Encoding, Qdrant's client, and ONNX inference are blocking; run them off the event loop
    # so one query's CPU work does not stall every other request.
    try:
        t0 = time.perf_counter()
        retrieved = await run_in_threadpool(_retrieve, question)
        t_retrieve = time.perf_counter() - t0

        t0 = time.perf_counter()
        top_k = req.top_k or settings.top_k_rerank
        if settings.rerank_enabled:
            reranked = await run_in_threadpool(rerank, question, retrieved, top_k)
        else:
            reranked = retrieved[:top_k]
        t_rerank = time.perf_counter() - t0
    except ApiException as exc:
        # Qdrant unreachable, or the collection is missing.
        raise HTTPException(status_code=503, detail="retrieval backend unavailable") from exc

    t0 = time.perf_counter()
    try:
        result = await get_graph().ainvoke({"question": question, "chunks": reranked})
    except anthropic.APIError as exc:
        raise HTTPException(status_code=502, detail="synthesis provider error") from exc
    t_synth = time.perf_counter() - t0
    t_total = time.perf_counter() - t_start

    tokens = {
        "input": result.get("input_tokens", 0),
        "output": result.get("output_tokens", 0),
        "cache_creation": result.get("cache_creation_input_tokens", 0),
        "cache_read": result.get("cache_read_input_tokens", 0),
    }
    cost = synthesis_cost_usd(
        settings,
        input_tokens=tokens["input"],
        output_tokens=tokens["output"],
        cache_creation_input_tokens=tokens["cache_creation"],
        cache_read_input_tokens=tokens["cache_read"],
    )
    metadata = {
        "latency_ms": {
            "retrieve": round(t_retrieve * 1000, 2),
            "rerank": round(t_rerank * 1000, 2),
            "synthesize": round(t_synth * 1000, 2),
            "total": round(t_total * 1000, 2),
        },
        "tokens": tokens,
        "llm_calls": result.get("llm_calls", 0),
        "model": settings.retrievault_synthesis_model,
        "est_cost_usd": cost,
        "retrieved_chunk_ids": [c["point_id"] for c in reranked],
        "retrieved_chunks": [
            {
                "point_id": c["point_id"],
                "file_path": c["file_path"],
                "symbol_name": c["symbol_name"],
                "start_line": c["start_line"],
                "end_line": c["end_line"],
                "rerank_score": c.get("rerank_score"),
                "content": c["code"],
                # The exact text this chunk contributed to the prompt, header and enclosing
                # signature included. The evaluation judges faithfulness against this, so that
                # the judge sees what the model saw rather than the code alone.
                "text": c.get("text") or c["code"],
            }
            for c in reranked
        ],
    }
    return QueryResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        refused=result.get("refused", False),
        grounding={
            "status": result.get("grounding", "uncited"),
            "invalid_labels": result.get("invalid_labels", []),
            "retries": result.get("retries", 0),
        },
        metadata=metadata,
    )

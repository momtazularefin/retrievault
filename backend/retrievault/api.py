import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel, model_validator
from qdrant_client import QdrantClient
from retrievault.config import get_settings
from retrievault.rerank.reranker import rerank
from retrievault.synthesize.graph import build_graph

from fastapi.middleware.cors import CORSMiddleware

STARTUP_TIME = time.time()

def _compute_code_hash() -> str:
    """Recursively computes a short hash of all Python files in the retrievault codebase."""
    hasher = hashlib.sha256()
    root_dir = Path(__file__).parent
    
    # Sort files to ensure deterministic hashing order
    py_files = sorted(root_dir.glob("**/*.py"))
    
    for file_path in py_files:
        try:
            # Normalize line endings to prevent OS encoding mismatches
            content = file_path.read_text(encoding="utf-8").replace("\r\n", "\n")
            hasher.update(content.encode("utf-8"))
        except OSError:
            pass
            
    return hasher.hexdigest()[:8]

BUILD_HASH = _compute_code_hash()

app = FastAPI(title="retrievault API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    settings = get_settings()
    
    # Try pinging Qdrant
    qdrant_ok = False
    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key_or_none)
        # Check if we can get collections as a ping
        client.get_collections()
        qdrant_ok = True
    except Exception:
        pass

    # Read the ingest manifest if present (written by `python -m retrievault.ingest`).
    chunk_count = None
    manifest_path = Path(__file__).parent.parent / "manifest.json"
    if manifest_path.exists():
        try:
            chunk_count = json.loads(manifest_path.read_text()).get("chunk_count")
        except (ValueError, OSError):
            chunk_count = None

    corpus_manifest = {
        "repo": settings.corpus_repo,
        "commit_tag": settings.corpus_tag,
        "chunk_count": chunk_count,
    }

    return {
        "status": "ok" if qdrant_ok else "degraded",
        "qdrant": qdrant_ok,
        "model": settings.retrievault_synthesis_model,
        "corpus": corpus_manifest,
        "build_hash": BUILD_HASH,
        "startup_time": STARTUP_TIME
    }

class QueryRequest(BaseModel):
    question: str | None = None
    query: str | None = None
    top_k: int | None = None

    @model_validator(mode="after")
    def normalize_question(self):
        if self.question is None and self.query is not None:
            self.question = self.query
        if not self.question or not self.question.strip():
            raise ValueError("question is required")
        return self

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]

_searcher = None
_graph = None

def get_searcher():
    global _searcher
    if _searcher is None:
        from retrievault.retrieve.hybrid_search import HybridSearcher

        _searcher = HybridSearcher()
    return _searcher

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    t_start = time.perf_counter()
    question = req.question.strip()
    
    # 1. Retrieve
    t0 = time.perf_counter()
    searcher = get_searcher()
    retrieved = searcher.search(question)
    chunks = [{"point_id": c.point_id, "file_path": c.file_path, "symbol_name": c.symbol_name, 
               "symbol_type": c.symbol_type, "start_line": c.start_line, "end_line": c.end_line, 
               "code": c.code, "github_url": c.github_url} for c in retrieved]
    t_retrieve = time.perf_counter() - t0
    
    # 2. Rerank
    t0 = time.perf_counter()
    reranked = rerank(question, chunks, top_k=req.top_k)
    t_rerank = time.perf_counter() - t0
    
    # 3. Synthesize
    t0 = time.perf_counter()
    graph = get_graph()
    state = {"question": question, "chunks": reranked}
    result = await graph.ainvoke(state)
    t_synth = time.perf_counter() - t0
    
    t_total = time.perf_counter() - t_start
    
    in_tokens = result.get("input_tokens", 0)
    out_tokens = result.get("output_tokens", 0)
    cache_create = result.get("cache_creation_input_tokens", 0)
    cache_read = result.get("cache_read_input_tokens", 0)
    cost = (in_tokens / 1_000_000 * 3.0) + (out_tokens / 1_000_000 * 15.0)
    
    metadata = {
        "latency_ms": {
            "retrieve": round(t_retrieve * 1000, 2),
            "rerank": round(t_rerank * 1000, 2),
            "synthesize": round(t_synth * 1000, 2),
            "total": round(t_total * 1000, 2)
        },
        "tokens": {
            "input": in_tokens,
            "output": out_tokens,
            "cache_creation": cache_create,
            "cache_read": cache_read,
        },
        "model": get_settings().retrievault_synthesis_model,
        "est_cost_usd": cost,
        "retrieved_chunk_ids": [c["point_id"] for c in reranked],
        "retrieved_chunks": [
            {
                "point_id": c["point_id"],
                "file_path": c["file_path"],
                "content": c["code"] # Map 'code' to 'content' for Ragas
            }
            for c in reranked
        ]
    }
    
    return QueryResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        metadata=metadata
    )

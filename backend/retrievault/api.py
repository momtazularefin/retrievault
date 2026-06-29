import json
import time
from pathlib import Path
from typing import List, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel
from qdrant_client import QdrantClient
from retrievault.config import get_settings
from retrievault.retrieve.hybrid_search import HybridSearcher
from retrievault.rerank.reranker import rerank
from retrievault.synthesize.graph import build_graph

app = FastAPI(title="retrievault API")

@app.get("/health")
def health_check():
    settings = get_settings()
    
    # Try pinging Qdrant
    qdrant_ok = False
    try:
        client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
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
        "corpus": corpus_manifest
    }

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]

_searcher = None
_graph = None

def get_searcher():
    global _searcher
    if _searcher is None:
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
    
    # 1. Retrieve
    t0 = time.perf_counter()
    searcher = get_searcher()
    retrieved = searcher.search(req.query)
    chunks = [{"point_id": c.point_id, "file_path": c.file_path, "symbol_name": c.symbol_name, 
               "symbol_type": c.symbol_type, "start_line": c.start_line, "end_line": c.end_line, 
               "code": c.code, "github_url": c.github_url} for c in retrieved]
    t_retrieve = time.perf_counter() - t0
    
    # 2. Rerank
    t0 = time.perf_counter()
    reranked = rerank(req.query, chunks)
    t_rerank = time.perf_counter() - t0
    
    # 3. Synthesize
    t0 = time.perf_counter()
    graph = get_graph()
    state = {"question": req.query, "chunks": reranked}
    result = await graph.ainvoke(state)
    t_synth = time.perf_counter() - t0
    
    t_total = time.perf_counter() - t_start
    
    in_tokens = result.get("input_tokens", 0)
    out_tokens = result.get("output_tokens", 0)
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
            "output": out_tokens
        },
        "est_cost_usd": cost,
        "retrieved_chunk_ids": [c["point_id"] for c in reranked]
    }
    
    return QueryResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations", []),
        metadata=metadata
    )

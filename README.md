# retrievault

Production-grade RAG service for the FastAPI codebase: native hybrid search (Qdrant BM25 + BGE) with reranking and agentic, citation-validated answers.

> Status: work in progress. **Done locally — M0 scaffold, M1 backend/health, M2 ingestion, M3 hybrid retrieval, M4 reranking.** Next — M5 synthesis + citations, then `/query`, frontend, eval, deploy.

## Architecture
- **Corpus**: `fastapi/fastapi` @ 0.136.3
- **Vector Store**: Qdrant (Hybrid: Dense BGE-base + Sparse BM25 via fastembed)
- **Reranker**: BAAI/bge-reranker-base
- **Synthesis**: Claude (`claude-sonnet-4-6`) via LangGraph
- **Frontend**: Next.js 15

More details coming soon.

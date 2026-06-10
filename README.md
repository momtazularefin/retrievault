# retrievault

Production-grade RAG service for the FastAPI codebase. Features native hybrid search (Qdrant BM25/BGE) and agentic citation validation.

## Architecture
- **Corpus**: `fastapi/fastapi` @ 0.136.3
- **Vector Store**: Qdrant (Hybrid: Dense BGE-base + Sparse BM25 via fastembed)
- **Reranker**: BAAI/bge-reranker-base
- **Synthesis**: Claude (`claude-sonnet-4-6`) via LangGraph
- **Frontend**: Next.js 15

More details coming soon.

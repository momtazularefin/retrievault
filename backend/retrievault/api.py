from fastapi import FastAPI
from qdrant_client import QdrantClient
from retrievault.config import get_settings

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

    # The manifest defaults (to be updated by indexer later)
    # Right now we just return what's in settings.
    corpus_manifest = {
        "repo": settings.corpus_repo,
        "commit_tag": settings.corpus_tag,
        "chunk_count": 0  # Placeholder, will be read from actual Qdrant collection payload later if needed, or indexer manifest
    }

    return {
        "status": "ok" if qdrant_ok else "degraded",
        "qdrant": qdrant_ok,
        "model": settings.retrievault_synthesis_model,
        "corpus": corpus_manifest
    }

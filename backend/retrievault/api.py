import json
from pathlib import Path

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

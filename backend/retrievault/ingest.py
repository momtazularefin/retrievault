"""Offline indexing: download the pinned corpus, chunk it, embed it, and load Qdrant.

Run with ``python -m retrievault.ingest``. The collection is rebuilt from scratch, so queries
see an empty or partial index until the run finishes; the manifest point is written last and
/health reports ``index_complete`` only once it matches the number of chunks.
"""

import hashlib
import os
import tarfile
import tempfile
import time
import urllib.request

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from retrievault.chunker import CHUNKER_VERSION, DEFAULT_MAX_CHARS, Chunk, chunk_file
from retrievault.collection import MANIFEST_POINT_ID
from retrievault.config import get_settings
from retrievault.encoders import load_dense_encoder, load_sparse_encoder

BATCH_SIZE = 64
EXCLUDED_DIRECTORIES = {"tests", "docs"}


def download_corpus(repo: str, tag: str, directory: str) -> tuple[str, str, str]:
    """Download and extract the tag archive; return (package path, archive root, archive sha256)."""
    url = f"https://github.com/{repo}/archive/refs/tags/{tag}.tar.gz"
    archive_path = os.path.join(directory, "corpus.tar.gz")
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, archive_path)
    with open(archive_path, "rb") as archive:
        digest = hashlib.sha256(archive.read()).hexdigest()

    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=directory, filter="data")

    package_name = repo.split("/")[-1]
    for entry in sorted(os.listdir(directory)):
        package_path = os.path.join(directory, entry, package_name)
        if os.path.isdir(package_path):
            return package_path, os.path.join(directory, entry), digest
    raise RuntimeError(f"Could not find the {package_name!r} package inside the archive")


def collect_chunks(package_path: str, archive_root: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for root, dirs, files in os.walk(package_path):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED_DIRECTORIES)
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            full_path = os.path.join(root, name)
            relative = os.path.relpath(full_path, archive_root).replace("\\", "/")
            with open(full_path, encoding="utf-8") as source:
                chunks.extend(chunk_file(full_path, relative, source.read()))
    return chunks


def generate_point_id(chunk: Chunk, tag: str) -> str:
    """Stable id for a chunk span at a tag, so re-running an ingest upserts rather than duplicates."""
    return hashlib.md5(f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{tag}".encode()).hexdigest()


def chunk_payload(chunk: Chunk, repo: str, tag: str) -> dict:
    return {
        **chunk.to_dict(),
        "repo": repo,
        "commit_tag": tag,
        "github_url": (
            f"https://github.com/{repo}/blob/{tag}/{chunk.file_path}"
            f"#L{chunk.start_line}-L{chunk.end_line}"
        ),
    }


def ingest() -> dict:
    settings = get_settings()
    collection = settings.qdrant_collection

    print("Loading fastembed models...")
    dense_model = load_dense_encoder(settings)
    sparse_model = load_sparse_encoder(settings)

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key_or_none)
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config={"dense": VectorParams(size=768, distance=Distance.COSINE)},
        # Qdrant/bm25 vectors carry term-frequency weights only; Qdrant applies IDF at query
        # time, and only when the sparse vector is configured with the IDF modifier.
        sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
    )

    with tempfile.TemporaryDirectory() as directory:
        package_path, archive_root, archive_sha256 = download_corpus(
            settings.corpus_repo, settings.corpus_tag, directory
        )
        chunks = collect_chunks(package_path, archive_root)
    print(f"Chunked {settings.corpus_repo}@{settings.corpus_tag} into {len(chunks)} chunks")

    batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    for batch_index in range(batches):
        batch = chunks[batch_index * BATCH_SIZE : (batch_index + 1) * BATCH_SIZE]
        texts = [chunk.embedding_text() for chunk in batch]
        dense_vectors = list(dense_model.embed(texts, batch_size=BATCH_SIZE))
        sparse_vectors = list(sparse_model.embed(texts, batch_size=BATCH_SIZE))
        points = [
            PointStruct(
                id=generate_point_id(chunk, settings.corpus_tag),
                vector={
                    "dense": dense.tolist(),
                    "bm25": SparseVector(
                        indices=sparse.indices.tolist(), values=sparse.values.tolist()
                    ),
                },
                payload=chunk_payload(chunk, settings.corpus_repo, settings.corpus_tag),
            )
            for chunk, dense, sparse in zip(batch, dense_vectors, sparse_vectors, strict=True)
        ]
        client.upsert(collection_name=collection, points=points, wait=True)
        print(f"Upserted batch {batch_index + 1}/{batches}")

    manifest = {
        "kind": "manifest",
        "repo": settings.corpus_repo,
        "commit_tag": settings.corpus_tag,
        "archive_sha256": archive_sha256,
        "chunk_count": len(chunks),
        "chunker_version": CHUNKER_VERSION,
        "max_chars": DEFAULT_MAX_CHARS,
        "embed_model": settings.embed_model,
        "sparse_model": settings.sparse_model,
        # Recorded because execution providers can differ in the last decimals of a vector.
        "acceleration": settings.acceleration,
        "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # Written last: its presence marks a completed ingest.
    client.upsert(
        collection_name=collection,
        points=[PointStruct(id=MANIFEST_POINT_ID, vector={}, payload=manifest)],
        wait=True,
    )
    print(f"Ingestion complete: {manifest}")
    return manifest


if __name__ == "__main__":
    ingest()

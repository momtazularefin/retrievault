import os
import tempfile
import urllib.request
import tarfile
import hashlib
from typing import List
from pathlib import Path

from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, PointStruct, Modifier

from retrievault.collection import COLLECTION_NAME
from retrievault.config import get_settings
from retrievault.chunker import chunk_file, Chunk

def download_and_extract_corpus(repo: str, tag: str, extract_dir: str):
    # e.g., fastapi/fastapi -> https://github.com/fastapi/fastapi/archive/refs/tags/0.136.3.tar.gz
    url = f"https://github.com/{repo}/archive/refs/tags/{tag}.tar.gz"
    tar_path = os.path.join(extract_dir, "corpus.tar.gz")
    
    print(f"Downloading corpus from {url}...")
    urllib.request.urlretrieve(url, tar_path)
    
    print("Extracting corpus...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
        
    # The extracted folder is usually something like fastapi-0.136.3
    repo_name = repo.split('/')[-1]
    # Find the top-level directory extracted
    extracted_dirs = [d for d in os.listdir(extract_dir) if os.path.isdir(os.path.join(extract_dir, d))]
    
    # Return the path to the actual python package inside (e.g. fastapi-0.136.3/fastapi)
    for d in extracted_dirs:
        pkg_path = os.path.join(extract_dir, d, repo_name)
        if os.path.exists(pkg_path):
            return pkg_path, d
    raise RuntimeError("Could not locate python package inside extracted corpus")

def generate_point_id(chunk: Chunk, tag: str) -> str:
    # A stable hash for idempotent re-index
    stable_str = f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{tag}"
    return hashlib.md5(stable_str.encode()).hexdigest()

def ingest():
    settings = get_settings()
    
    print("Loading fastembed models...")
    dense_model = TextEmbedding(model_name=settings.embed_model)
    sparse_model = SparseTextEmbedding(model_name=settings.sparse_model)
    
    print("Connecting to Qdrant...")
    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    
    collection_name = COLLECTION_NAME
    
    # Recreate collection for simplicity in v1
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=768, distance=Distance.COSINE)
        },
        sparse_vectors_config={
            # IDF modifier is REQUIRED for Qdrant/bm25 sparse vectors: Qdrant applies
            # the IDF term at query time. Without it, sparse scoring is raw TF, not BM25.
            "bm25": SparseVectorParams(modifier=Modifier.IDF)
        }
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        pkg_path, base_folder = download_and_extract_corpus(settings.corpus_repo, settings.corpus_tag, tmpdir)
        
        all_chunks: List[Chunk] = []
        
        print(f"Walking package path: {pkg_path}")
        for root, dirs, files in os.walk(pkg_path):
            # Exclude tests and docs
            if 'tests' in dirs:
                dirs.remove('tests')
            if 'docs' in dirs:
                dirs.remove('docs')
                
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    # Rel path should be something like fastapi/routing.py
                    repo_rel_path = os.path.relpath(full_path, os.path.join(tmpdir, base_folder))
                    repo_rel_path = repo_rel_path.replace("\\", "/") # normalize
                    
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    file_chunks = chunk_file(full_path, repo_rel_path, content)
                    all_chunks.extend(file_chunks)
                    
        print(f"Extracted {len(all_chunks)} chunks from {settings.corpus_repo}@{settings.corpus_tag}")
        
        batch_size = 64
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            texts = [c.code for c in batch]
            
            dense_vectors = list(dense_model.embed(texts))
            sparse_vectors = list(sparse_model.embed(texts))
            
            points = []
            for j, chunk in enumerate(batch):
                payload = chunk.to_dict()
                payload["repo"] = settings.corpus_repo
                payload["commit_tag"] = settings.corpus_tag
                payload["github_url"] = f"https://github.com/{settings.corpus_repo}/blob/{settings.corpus_tag}/{chunk.file_path}#L{chunk.start_line}-L{chunk.end_line}"
                
                point_id = generate_point_id(chunk, settings.corpus_tag)
                
                # fastembed sparse returns SparseEmbedding object with indices and values
                sparse_embedding = sparse_vectors[j]
                
                points.append(
                    PointStruct(
                        id=point_id,
                        vector={
                            "dense": dense_vectors[j].tolist(),
                            "bm25": {
                                "indices": sparse_embedding.indices.tolist(),
                                "values": sparse_embedding.values.tolist()
                            }
                        },
                        payload=payload
                    )
                )
                
            client.upsert(
                collection_name=collection_name,
                points=points
            )
            print(f"Upserted batch {i//batch_size + 1}/{(len(all_chunks)+batch_size-1)//batch_size}")
            
        print("Ingestion complete!")
        
        # Write manifest
        manifest_path = Path(__file__).parent.parent / "manifest.json"
        import json
        manifest = {
            "repo": settings.corpus_repo,
            "commit_tag": settings.corpus_tag,
            "chunk_count": len(all_chunks)
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Wrote manifest to {manifest_path}")

if __name__ == "__main__":
    ingest()

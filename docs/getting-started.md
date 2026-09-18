# RetrieVault - Getting Started & Developer Operations

Setting up a clean machine, running the stack, and re-running the evaluation.

---

## 1. Prerequisites

- **Git**: [Git for Windows](https://git-scm.com/download/win), or `brew install git` /
  your package manager elsewhere.
- **Docker Desktop**: [docker.com](https://www.docker.com/products/docker-desktop/). On Windows,
  enable the WSL2 backend (*Settings → General → Use the WSL 2 based engine*).
- **Node.js 24** for the frontend, most easily through a version manager
  ([nvm-windows](https://github.com/coreybutler/nvm-windows/releases), or `nvm` elsewhere):
  ```bash
  nvm install 24
  nvm use 24
  ```
- **[uv](https://docs.astral.sh/uv/)** for the backend. It manages the virtual environment and
  downloads the right Python itself, so Python does not need to be installed globally:
  ```powershell
  powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows
  ```
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh              # macOS / Linux
  ```

---

## 2. Checkout and configuration

```bash
git clone https://github.com/momtazularefin/retrievault.git
cd retrievault
cp .env.example .env
```

Put an `ANTHROPIC_API_KEY` in `.env`. Every other setting has a working default; see
[configuration.md](configuration.md).

---

## 3. Running locally

### Step 1: Qdrant

```bash
docker compose up -d qdrant
```

Dashboard: [http://localhost:6333/dashboard](http://localhost:6333/dashboard).

### Step 2: Backend

```bash
cd backend
uv sync --all-extras
uv run uvicorn retrievault.api:app --reload --port 8000
```

`uv sync` installs exactly what `uv.lock` pins. Swagger UI:
[http://localhost:8000/docs](http://localhost:8000/docs).

`/health` will report `"status": "degraded"` and `"index_complete": false` until the corpus is
indexed, which is the next step.

### Step 3: Index the corpus

```bash
cd backend
uv run python -m retrievault.ingest
```

This downloads the pinned FastAPI release archive, chunks it, embeds every chunk, and loads
Qdrant — about 840 chunks, a few minutes on a CPU. The manifest point is written last, so
`/health` flips to `"status": "ok"` only when the index is complete.

The first run also downloads the ONNX models (about 1.2 GB, mostly the reranker). They are
cached under the system temp directory unless `FASTEMBED_CACHE_PATH` points somewhere stable.

Re-run the ingest after changing `CORPUS_TAG`, the chunker, or the embedding model. The
collection is rebuilt in place, so queries see a partial index while it runs.

### Step 4: Frontend

```bash
cd ../frontend
npm install
npm run dev
```

[http://localhost:3000](http://localhost:3000). For a backend that is not on localhost, set
`NEXT_PUBLIC_API_URL` in `frontend/.env.local`.

### Everything in containers

`docker compose up` builds and runs the backend container alongside Qdrant. The ingest still has
to be run once against that Qdrant, either from the host or with
`docker compose exec backend python -m retrievault.ingest`.

---

## 4. Checks

```bash
cd backend
uv run ruff check .
uv run pytest
```

The integration tests need Qdrant running; they skip themselves when it is not reachable. CI
runs the same two commands with a Qdrant service container, plus `npm run lint` and
`npm run build` for the frontend.

Run both before pushing: CI installs from `uv.lock` (`uv sync --frozen`), so a green local run
with the same lockfile is a real signal.

---

## 5. Evaluation

```bash
cd backend

# Retrieval only: gold-file and gold-symbol hit rates. No LLM, no cost, about a minute.
uv run python -m retrievault.eval --retrieval-only

# Full run: 50 queries through the API, then the Ragas judge. Costs roughly $0.60.
uv run python -m retrievault.eval
```

Reports are written to `eval/reports/` as Markdown and JSON, including the index manifest,
dataset hash, models, and package versions used. Metric definitions and the current results are
in [evaluation.md](evaluation.md).

Useful flags for retrieval experiments, each writing its own report:

```bash
uv run python -m retrievault.eval --retrieval-only --no-rerank --name no-rerank
uv run python -m retrievault.eval --retrieval-only --fusion-depth 30 --name deep-pool
uv run python -m retrievault.eval --retrieval-only --dataset eval/paraphrase-probe.jsonl --name probe
```

---

## 6. Changing code

- **Backend, host**: `uvicorn --reload` picks up saved files. The models stay loaded in the
  process, so the first query after a restart pays the load time again.
- **Backend, container**: rebuild the image (`docker compose build backend`); the image no longer
  mounts your working tree, so a restart alone will not pick up an edit.
- **Frontend**: Fast Refresh applies edits without a reload.
- **Dependencies**: edit `pyproject.toml`, then `uv lock` and `uv sync --all-extras`. Commit
  `uv.lock` with the change — CI installs from it, and that is what keeps a new release of a
  tool from breaking the build.
- **Chunking or embedding changes** invalidate the index. Re-run the ingest, bump
  `CHUNKER_VERSION` in `retrievault/chunker.py` for anything that moves chunk boundaries, and
  re-run the evaluation, since published numbers belong to a specific index.

---

## 7. Shutting down

Stop the frontend and backend with `CTRL+C` in their terminals, then:

```bash
docker compose stop qdrant     # keep the index
docker compose down            # remove containers, keep the volume
docker compose down -v         # also delete the index; a re-ingest is then required
```

Prefer `docker compose stop` over killing the container: Qdrant flushes its write-ahead log and
index segments on SIGTERM. It does recover from its WAL after a hard kill, but a clean stop
avoids the recovery pass entirely.

### Clearing local caches

```bash
# Python and test caches
find . -type d -name "__pycache__" -exec rm -rf {} +    # macOS / Linux
rm -rf .pytest_cache backend/.pytest_cache backend/.ruff_cache

# Next.js build cache
rm -rf frontend/.next
```

On Windows, the PowerShell equivalents are `Get-ChildItem -Recurse -Filter __pycache__ |
Remove-Item -Recurse -Force` and `Remove-Item -Recurse -Force frontend\.next`.

To stop a stray dev server, close its terminal or stop that specific process by port
(`npx kill-port 3000`). Do not kill every `node` process on the machine — that takes editors and
language servers with it.

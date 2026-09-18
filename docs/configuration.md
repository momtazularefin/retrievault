# RetrieVault Configuration Guide

Every setting is an environment variable, read by Pydantic Settings in `retrievault/config.py`.
Local runs read the repository-root `.env`; containers receive the variables directly. Copy
`.env.example` to `.env` to start.

---

## 1. Synthesis

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | *(empty)* | Required to answer queries, and to judge when `EVAL_JUDGE_PROVIDER=claude`. |
| `RETRIEVAULT_SYNTHESIS_MODEL` | `claude-sonnet-4-6` | Anthropic model id used for synthesis. |
| `SYNTHESIS_MAX_TOKENS` | `2048` | Output cap. An answer that hits it is reported with `grounding.status = "truncated"` rather than passed off as complete. |
| `SYNTHESIS_INPUT_USD_PER_MTOK` | `3.0` | Input price used for the cost estimate. Update it with the model. |
| `SYNTHESIS_OUTPUT_USD_PER_MTOK` | `15.0` | Output price used for the cost estimate. |

The cost in `metadata.est_cost_usd` is an estimate from token counts and these prices, including
cache writes at 1.25x and cache reads at 0.1x. It is not a billing figure.

---

## 2. Qdrant

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint. |
| `QDRANT_API_KEY` | *(empty)* | Leave empty for the local Compose service; set for a secured remote instance. |
| `QDRANT_COLLECTION` | `retrievault_fastapi` | Collection to index into and query. Point a second collection at a different tag or chunker to compare them. |

---

## 3. Corpus

| Variable | Default | Description |
|---|---|---|
| `CORPUS_REPO` | `fastapi/fastapi` | GitHub repository to index. |
| `CORPUS_TAG` | `0.136.3` | Release tag. Used for the archive download, citation URLs, and the manifest. Changing it requires a re-ingest and a re-curated evaluation set. |

---

## 4. Retrieval

| Variable | Default | Description |
|---|---|---|
| `PREFETCH_LIMIT` | `50` | Candidates fetched per vector (dense and sparse) before fusion. |
| `TOP_N_FUSION` | `12` | Candidates kept after fusion and handed to the reranker. Was 30; a deeper pool scored no better and cost 3.6x the time. |
| `TOP_K_RERANK` | `6` | Chunks sent to the model. |
| `RERANK_ENABLED` | `true` | `false` sends the fused top-k straight to the model. On the evaluation set that scores the same and removes seconds of latency — see [evaluation.md](evaluation.md). |

---

## 5. Local models and hardware

| Variable | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `BAAI/bge-base-en-v1.5` | Dense embedder (fastembed serves an int8-quantised ONNX export). |
| `SPARSE_MODEL` | `Qdrant/bm25` | Sparse encoder. Pure Python and hashing, no ONNX session. |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder. `Xenova/ms-marco-MiniLM-L-6-v2` is a much smaller alternative. |
| `FASTEMBED_CACHE_PATH` | *(system temp)* | Where the ONNX models are cached. The default lives in the temp directory, which is cleared periodically; set a stable path to avoid re-downloading about 1.2 GB. |
| `ACCELERATION` | `none` | `none` (CPU), `gpu` (CUDA, ROCm, or DirectML), `npu` (Vitis AI). |

`ACCELERATION` is strict in both directions. A requested provider that ONNX Runtime does not
have raises at startup, and the session is re-checked after creation because ONNX Runtime falls
back to the CPU by itself and only prints a warning. DirectML is a GPU path, so it is not
accepted for `npu`: on an AMD Ryzen AI laptop it runs on the Radeon iGPU, and reporting that as
NPU inference would be false.

The published numbers were measured with `ACCELERATION=none`. Provider choice changes the last
decimals of a vector, so an index built on one provider is not bit-identical to another; the
manifest records which was used.

---

## 6. API

| Variable | Default | Description |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:3000` | Comma-separated browser origins allowed to call the API. Credentials are not allowed, so a wildcard is never needed. |

Requests are bounded at the schema: `question` at most 2,000 characters, `top_k` between 1 and
20. The legacy `{"query": ...}` field is still accepted.

---

## 7. Evaluation

| Variable | Default | Description |
|---|---|---|
| `EVAL_GOOD_COUNT` | `45` | Answerable questions to run. The curated set has 45. |
| `EVAL_REFUSAL_COUNT` | `5` | Refusal probes to run. The curated set has 5. |
| `EVAL_JUDGE_PROVIDER` | `claude` | `claude` or `openai`. There is no automatic fallback; a missing key for the selected provider is an error. |
| `EVAL_JUDGE_MODEL` | `claude-haiku-4-5-20251001` | Judge model. The evaluation plan fixes this one for reproducibility, and it is deliberately not the synthesis model. |
| `EVAL_MAX_WORKERS` | `8` | Parallel judge calls. Queries themselves always run one at a time, so each reported latency is a single-request latency. |
| `OPENAI_API_KEY` | *(empty)* | Only for `EVAL_JUDGE_PROVIDER=openai`. |

There is no response cache and no judge cache: a cached answer keyed on settings alone silently
survives a code change, which is exactly when a result must not be reused.

---

## 8. Frontend

| Variable | Default | Description |
|---|---|---|
| `NEXT_PUBLIC_API_URL` | `http://localhost:8000` | Browser-visible API base URL. Set it in `frontend/.env.local`. |

The UI reads the model name, corpus tag, chunk count, and build hash from `/health` instead of
hard-coding them, so it cannot drift from the running backend.

# Retrievault Configuration Guide

This document describes all environment variables used to configure the Retrievault API, hybrid search engine, and evaluation suite. All variables are loaded via Pydantic Settings in `retrievault/config.py`.

---

## 1. LLM & API Access

| Environment Variable | Type | Default | Description |
|----------------------|------|---------|-------------|
| `ANTHROPIC_API_KEY` | `str` | *None* | Required API key for Anthropic Claude. |
| `OPENAI_API_KEY` | `str` | `""` | Required only when `EVAL_JUDGE_PROVIDER=openai` for the Ragas evaluation judge. |
| `RETRIEVAULT_SYNTHESIS_MODEL` | `str` | `claude-sonnet-4-6` | Synthesis model name used by the LangGraph query pipeline. |

---

## 2. Qdrant Vector Database

| Environment Variable | Type | Default | Description |
|----------------------|------|---------|-------------|
| `QDRANT_URL` | `str` | `http://localhost:6333` | Host URL for the self-hosted Qdrant instance. |
| `QDRANT_API_KEY` | `str` | `""` | Optional API key for secured remote Qdrant. Leave empty for the local Docker Compose service. |

---

## 3. Retrieval Parameters

| Environment Variable | Type | Default | Description |
|----------------------|------|---------|-------------|
| `PREFETCH_LIMIT` | `int` | `50` | Maximum candidates to fetch from dense/sparse databases initially. |
| `TOP_N_FUSION` | `int` | `30` | Number of unified candidates merged during Reciprocal Rank Fusion. |
| `TOP_K_RERANK` | `int` | `6` | Number of final chunks passed to Claude after Cross-Encoder reranking. |
| `RERANK_MODEL_DIR` | `str` | `models/bge-reranker-onnx` | Local ONNX reranker export/cache directory, resolved relative to `backend/` when not absolute. The directory is generated and ignored by git. |
| `ACCELERATION` | `str` | `none` | Hardware acceleration for fastembed and the ONNX reranker. `none` = CPU only, `gpu` = GPU (strict), `npu` = NPU (strict). No silent fallback. |

---

## 4. Evaluation Suite Configuration

These settings control the behavior of the evaluation pipeline in `retrievault/eval.py`.

| Environment Variable | Type | Recommended Range | Description |
|----------------------|------|-------------------|-------------|
| `EVAL_GOOD_COUNT` | `int` | `3` to `45` | Number of factual questions from `dataset.jsonl` to evaluate. (45 for a full run). |
| `EVAL_REFUSAL_COUNT` | `int` | `2` to `5` | Number of trick/refusal questions from `dataset.jsonl` to evaluate. (5 for a full run). |
| `EVAL_JUDGE_PROVIDER` | `str` | `claude` or `openai` | Provider used for Ragas judging. The model must match the provider. |
| `EVAL_JUDGE_MODEL` | `str` | *Model name* | LLM used to grade Ragas results (default: `claude-haiku-4-5-20251001`; latest smoke report used `gpt-4o-mini`). |
| `EVAL_USE_RESPONSE_CACHE` | `bool` | `True` or `False` | Enables Layer 1 caching of backend response outputs. (Should be `False` in Dev/CI). |
| `EVAL_USE_JUDGE_CACHE` | `bool` | `True` or `False` | Enables Layer 2 LangChain SQLite caching of Ragas judge calls. (Should be `False` in Dev/CI). |
| `EVAL_MAX_WORKERS` | `int` | `16` to `50` | Concurrency limit for parallel Ragas grading tasks. (Default `30`). |
| `EVAL_CONCURRENT_QUERIES` | `int` | `2` to `10` | Concurrency limit for parallel queries to the FastAPI RAG backend. (Default `5`). |

---

## 5. Frontend Configuration

| Environment Variable | Type | Default | Description |
|----------------------|------|---------|-------------|
| `NEXT_PUBLIC_API_URL` | `str` | `http://localhost:8000` | Browser-visible API base URL used by the Next.js chat UI. Set this in `frontend/.env.local` when the API is not running on localhost. |

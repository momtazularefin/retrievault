# Retrievault Evaluation Suite Reference

This document explains the evaluation suite built for Retrievault in `retrievault/eval.py`. It outlines the dataset, quality metrics, grounding heuristics, caching strategies, and parallelized execution design.

Current status: the latest generated smoke report (`backend/eval/reports/report.md`, 2026-07-03) used 5 queries and `gpt-4o-mini` as judge. Citation validity and refusal correctness passed, but Ragas quality metrics and latency gates failed, so M8/AC4/AC5 are not complete yet.

---

## 1. Quality Metrics (Ragas)

I use the Ragas framework to programmatically assess core response quality. The default LLM judge is `claude-haiku-4-5-20251001`; the runner can also use an OpenAI judge such as `gpt-4o-mini` when `EVAL_JUDGE_PROVIDER=openai`.

* **Faithfulness (Target: >= 0.85)**: Verifies that all claims made in the answer can be directly inferred from the retrieved source chunks. It catches hallucinations and external knowledge leaks.
* **Context Precision (Target: >= 0.70)**: Evaluates whether the retriever correctly ranked the relevant code chunks at the top of the context list.
* **Answer Relevancy (Target: >= 0.80)**: Measures how directly the generated answer addresses the user's initial question, penalizing verbose or redundant replies.

---

## 2. Grounding Heuristics

To verify code safety and citation hygiene, the evaluation pipeline runs two deterministic checks:

* **Citation Validity (Target: == 1.00)**: Asserts that for every factual/how-to query, the generated answer cites at least one of the human-flagged "gold files" from the evaluation dataset.
* **Refusal Correctness (Target: >= 0.90)**: Asserts that for out-of-scope or unanswerable queries (e.g. weather in Tokyo), the model cleanly refuses to answer (containing keywords like `cannot` or `not available`) and includes **exactly 0 citations**.

---

## 3. Parallelized Async Architecture

To prevent long execution wait times, the evaluation script executes all tasks concurrently:
* **Concurrent Generation**: Backend requests are throttled using `asyncio.Semaphore` with the `EVAL_CONCURRENT_QUERIES` setting (default: 5) to parallelize Claude Sonnet synthesis calls without overloading the CPU during reranking.
* **Concurrent Grading**: Ragas requests are parallelized using a custom `RunConfig` with `max_workers=EVAL_MAX_WORKERS` (default: 30) to submit grading prompts to the configured judge model concurrently.

### Latency Amortization under Concurrency
Running queries concurrently causes them to share CPU/GPU/NPU resources for the ONNX Cross-Encoder reranker, inflating the raw wall-clock duration of individual requests. To represent the single-user equivalent performance under batch load, Retrievault calculates **Amortized Latency** by dividing the raw elapsed query time by the active concurrency factor (`EVAL_CONCURRENT_QUERIES`).

---

## 4. Multi-Layer Caching (Optional)

I support optional caching layers which can be toggled via environment settings:

* **Layer 1: Backend Cache (`EVAL_USE_RESPONSE_CACHE`)**: Saves raw API responses to `eval/.cache_responses.json`. 
* **Layer 2: LLM Cache (`EVAL_USE_JUDGE_CACHE`)**: Intercepts LangChain calls and saves Ragas grading prompts to a local SQLite database (`eval/.langchain_cache.db`).

> [!IMPORTANT]
> Both caches are **disabled by default** during local development and CI testing to guarantee that code modifications, retriever weights, and system prompts are evaluated fresh. Caching should only be enabled to debug report layout formatting or test script logic.

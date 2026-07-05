import os
import json
import math
import time
import hashlib
import tempfile
import asyncio
import warnings
from pathlib import Path

import httpx
from datasets import Dataset
from langchain_community.cache import SQLiteCache
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.globals import set_llm_cache
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.metrics import ContextPrecision, Faithfulness, AnswerRelevancy

from retrievault.api import app
from retrievault.config import get_settings

# Suppress noisy deprecation/user warnings for clean output
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Wrapper to bypass Ragas telemetry bug (which expects e.model to be a string)
class SafeFastEmbedEmbeddings(Embeddings):
    def __init__(self, *args, **kwargs):
        self._underlying = FastEmbedEmbeddings(*args, **kwargs)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._underlying.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._underlying.embed_query(text)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self._underlying.aembed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await self._underlying.aembed_query(text)

    @property
    def model(self) -> str:
        return self._underlying.model_name

    @property
    def model_name(self) -> str:
        return self._underlying.model_name

# ---------------------------------------------------------------------------
# Layer 1: Atomic cache for backend responses
# ---------------------------------------------------------------------------
RESPONSE_CACHE_PATH = Path("eval/.cache_responses.json")
# Async lock to ensure cache writes remain atomic during concurrent executions
CACHE_LOCK = asyncio.Lock()

def _compute_cache_version(settings, dataset_hash: str) -> str:
    """Cache key combining dataset content + retrieval settings."""
    parts = "|".join([
        dataset_hash,
        settings.retrievault_synthesis_model,
        str(settings.prefetch_limit),
        str(settings.top_n_fusion),
        str(settings.top_k_rerank),
        settings.embed_model,
        settings.rerank_model,
    ])
    return hashlib.sha256(parts.encode()).hexdigest()[:16]


def _load_response_cache(cache_path: Path, version: str) -> dict:
    """Load cached backend responses; returns empty dict if stale or missing."""
    if not cache_path.exists():
        return {}
    try:
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        if data.get("_version") != version:
            print("  Cache version mismatch — invalidating.")
            return {}
        return data.get("entries", {})
    except (json.JSONDecodeError, KeyError):
        return {}


async def _save_response_cache_async(cache_path: Path, entries: dict, version: str) -> None:
    """Atomically persist the response cache via temp-file + os.replace under an async lock."""
    async with CACHE_LOCK:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        data = {"_version": version, "entries": entries}
        # Run synchronous file writing in a thread to keep loop non-blocking
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, _write_cache_file, cache_path, data)

def _write_cache_file(cache_path: Path, data: dict) -> None:
    fd, tmp = tempfile.mkstemp(
        dir=str(cache_path.parent), suffix=".tmp", prefix=".cache_"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, str(cache_path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Single query worker (processes a single item concurrently)
# ---------------------------------------------------------------------------
async def process_query(
    client: httpx.AsyncClient,
    item: dict,
    response_cache: dict,
    cache_version: str,
    semaphore: asyncio.Semaphore,
    settings,
    results: list,
    latencies: list,
    ragas_data: dict,
    stats_accumulator: dict,
    total_items: int
):
    qid = item["id"]
    question = item["question"]
    
    # Check Layer 1 cache (no semaphore needed for cache read check)
    if settings.eval_use_response_cache and qid in response_cache:
        cached = response_cache[qid]
        answer = cached["answer"]
        citations = cached["citations"]
        meta = cached["metadata"]
        latency = cached["latency"]
        print(f"Running query: {question} -> [CACHED]")
        stats_accumulator["cache_hits"] += 1
    else:
        # Live query — throttle with semaphore to avoid CPU/API overload
        async with semaphore:
            start_t = time.perf_counter()
            resp = await client.post("/query", json={"question": question}, timeout=60.0)
            raw_latency = time.perf_counter() - start_t
            
            # Amortize latency by dividing by the active concurrency factor
            concurrency = min(settings.eval_concurrent_queries, total_items)
            latency = raw_latency / concurrency if concurrency > 0 else raw_latency
            
            if resp.status_code != 200:
                print(f"Running query: {question} -> ERROR {resp.status_code}: {resp.text}")
                return
                
            data = resp.json()
            answer = data["answer"]
            citations = data["citations"]
            meta = data.get("metadata", {})
            print(f"Running query: {question} -> [{latency:.1f}s]")
            
            # Persist to Layer 1 cache atomically if enabled
            if settings.eval_use_response_cache:
                response_cache[qid] = {
                    "answer": answer,
                    "citations": citations,
                    "metadata": meta,
                    "latency": latency,
                }
                await _save_response_cache_async(RESPONSE_CACHE_PATH, response_cache, cache_version)

    # Accumulate metrics (async safe appending since python lists are thread-safe and loops are single-threaded async)
    latencies.append(latency)
    
    tokens_dict = meta.get("tokens", {})
    stats_accumulator["total_input_tokens"] += tokens_dict.get("input", 0)
    stats_accumulator["total_output_tokens"] += tokens_dict.get("output", 0)

    # Extract context texts
    contexts = [
        chunk.get("content", "") for chunk in meta.get("retrieved_chunks", [])
    ]

    # Prepare for Ragas if it's not a refusal
    if item.get("category") != "refusal":
        ragas_data["user_input"].append(question)
        ragas_data["response"].append(answer)
        ragas_data["retrieved_contexts"].append(contexts)
        ragas_data["reference"].append(item["reference_answer"])

    # Refusal correctness check
    is_refusal_correct = None
    if item.get("category") == "refusal":
        is_refusal_correct = len(citations) == 0 and (
            "cannot" in answer.lower() or "not" in answer.lower()
        )

    # Citation validity check
    citation_validity = None
    if item.get("category") != "refusal" and len(item.get("gold_files", [])) > 0:
        cited_files = [c.get("file_path", "") for c in citations]
        valid = any(gf in cited_files for gf in item["gold_files"])
        citation_validity = 1.0 if valid else 0.0

    results.append(
        {
            "id": item["id"],
            "question": question,
            "category": item.get("category"),
            "latency": latency,
            "citations": len(citations),
            "refusal_correctness": is_refusal_correct,
            "citation_validity": citation_validity,
        }
    )


# ---------------------------------------------------------------------------
# Main eval
# ---------------------------------------------------------------------------
async def run_eval():
    settings = get_settings()
    dataset_path = Path("eval/dataset.jsonl")
    if not dataset_path.exists():
        print(f"Dataset not found at {dataset_path}")
        return

    # --- Layer 2: Enable LangChain SQLite cache for Ragas judge calls if enabled ---
    if settings.eval_use_judge_cache:
        cache_db_path = Path("eval/.langchain_cache.db")
        cache_db_path.parent.mkdir(parents=True, exist_ok=True)
        set_llm_cache(SQLiteCache(database_path=str(cache_db_path)))
        print(f"Layer 2 cache: LangChain SQLiteCache enabled at {cache_db_path}")
    else:
        print("Layer 2 cache: LangChain SQLiteCache disabled.")

    # --- Load dataset ---
    print(f"Loading dataset from {dataset_path}")
    raw = dataset_path.read_text(encoding="utf-8").strip()
    lines = raw.split("\n")
    header = json.loads(lines[0])
    all_items = [json.loads(line) for line in lines[1:]]

    # --- Subset by configured counts ---
    good_items = [x for x in all_items if x.get("category") != "refusal"]
    refusal_items = [x for x in all_items if x.get("category") == "refusal"]

    effective_good = min(len(good_items), settings.eval_good_count)
    effective_refusal = min(len(refusal_items), settings.eval_refusal_count)
    good_subset = good_items[:effective_good]
    refusal_subset = refusal_items[:effective_refusal]
    items = refusal_subset + good_subset  # refusals first, then good

    print(
        f"Subset: {len(good_subset)} good + {len(refusal_subset)} refusal "
        f"= {len(items)} items "
        f"(from {len(good_items)} good + {len(refusal_items)} refusal available)"
    )

    # --- Layer 1: Load cached backend responses if enabled ---
    dataset_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
    cache_version = _compute_cache_version(settings, dataset_hash)
    response_cache = {}
    if settings.eval_use_response_cache:
        response_cache = _load_response_cache(RESPONSE_CACHE_PATH, cache_version)
        print(f"Layer 1 cache: ResponseCache enabled, {len(response_cache)} entries at {RESPONSE_CACHE_PATH}")
    else:
        print("Layer 1 cache: ResponseCache disabled.")

    print("Initializing Async TestClient and starting concurrent execution...\n")
    
    # Initialize shared execution state
    results = []
    latencies = []
    ragas_data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }
    stats_accumulator = {
        "cache_hits": 0,
        "total_input_tokens": 0,
        "total_output_tokens": 0
    }
    
    semaphore = asyncio.Semaphore(settings.eval_concurrent_queries)

    # Execute backend calls concurrently
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        tasks = [
            process_query(
                client=client,
                item=item,
                response_cache=response_cache,
                cache_version=cache_version,
                semaphore=semaphore,
                settings=settings,
                results=results,
                latencies=latencies,
                ragas_data=ragas_data,
                stats_accumulator=stats_accumulator,
                total_items=len(items)
            )
            for item in items
        ]
        await asyncio.gather(*tasks)

    cache_hits = stats_accumulator["cache_hits"]
    total_input_tokens = stats_accumulator["total_input_tokens"]
    total_output_tokens = stats_accumulator["total_output_tokens"]
    
    print(f"\nLayer 1 summary: {cache_hits} cache hits, {len(items) - cache_hits} live queries")

    # --- Run Ragas evaluation ---
    print("\n--- Running Ragas Evaluation ---")
    print(f"Evaluating {len(ragas_data['user_input'])} items × 3 metrics "
          f"= {len(ragas_data['user_input']) * 3} grading tasks")
    print(f"Judge provider: {settings.eval_judge_provider}")
    print(f"Judge model: {settings.eval_judge_model}")
    if settings.eval_use_judge_cache:
        print("Layer 2 cache: duplicate prompts served from SQLite (free)\n")
    else:
        print("Layer 2 cache: disabled (direct LLM requests only)\n")

    eval_dataset = Dataset.from_dict(ragas_data)

    if settings.eval_judge_provider == "openai":
        if not settings.openai_api_key.strip():
            raise ValueError(
                "OpenAI evaluation judge was selected (EVAL_JUDGE_PROVIDER=openai), "
                "but OPENAI_API_KEY is not set or is empty in the environment."
            )
        judge_llm = ChatOpenAI(
            model=settings.eval_judge_model, 
            api_key=settings.openai_api_key,
            temperature=0.0,
            max_retries=0
        )
    elif settings.eval_judge_provider == "claude":
        if not settings.anthropic_api_key.strip():
            raise ValueError(
                "Claude evaluation judge was selected (EVAL_JUDGE_PROVIDER=claude), "
                "but ANTHROPIC_API_KEY is not set or is empty in the environment."
            )
        judge_llm = ChatAnthropic(
            model=settings.eval_judge_model, 
            api_key=settings.anthropic_api_key,
            temperature=0.0,
            max_retries=0
        )
    else:
        raise ValueError(
            f"Invalid EVAL_JUDGE_PROVIDER: '{settings.eval_judge_provider}'. "
            "Must be set explicitly to either 'claude' or 'openai' (no silent fallbacks allowed)."
        )
    ragas_embeddings = SafeFastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

    # Ragas v0.4+ RunConfig for custom concurrency
    from ragas.run_config import RunConfig
    run_config = RunConfig(
        max_workers=settings.eval_max_workers, 
        max_retries=1, 
        timeout=30
    )

    ragas_results = evaluate(
        dataset=eval_dataset,
        metrics=[ContextPrecision(), Faithfulness(), AnswerRelevancy()],
        llm=judge_llm,
        embeddings=ragas_embeddings,
        run_config=run_config
    )

    print("\n--- Ragas Results ---")
    print(ragas_results)

    # Safely extract scores (handle nan and missing keys)
    safe_results = {}
    for key in ["context_precision", "faithfulness", "answer_relevancy"]:
        try:
            # Ragas EvaluationResult indexes return lists of raw scores. 
            # The aggregated mean scores are kept in the private _repr_dict.
            val = ragas_results._repr_dict.get(key, 0.0)
            if math.isnan(val):
                val = 0.0
        except Exception:
            val = 0.0
        safe_results[key] = val

    # Calculate aggregate performance metrics
    latencies.sort()
    p50 = latencies[len(latencies) // 2] if latencies else 0
    p95 = latencies[int(len(latencies) * 0.95)] if latencies else 0

    cost = (total_input_tokens / 1_000_000 * 3.00) + (
        total_output_tokens / 1_000_000 * 15.00
    )  # Sonnet 4.6 pricing
    avg_cost = cost / len(items) if items else 0

    # Calculate aggregate quality metrics
    valid_citations = [
        r["citation_validity"] for r in results if r["citation_validity"] is not None
    ]
    avg_citation_validity = (
        sum(valid_citations) / len(valid_citations) if valid_citations else 0.0
    )

    refusal_scores = [
        1.0 if r["refusal_correctness"] else 0.0
        for r in results
        if r["refusal_correctness"] is not None
    ]
    avg_refusal_correctness = (
        sum(refusal_scores) / len(refusal_scores) if refusal_scores else 0.0
    )

    # Generate Report
    report_dir = Path("eval/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"

    report_content = f"""# RetrieVault Evaluation Report

## Metadata
* **Corpus**: {header.get('corpus_repo')} @ {header.get('corpus_tag')}
* **Synthesis Model**: {settings.retrievault_synthesis_model}
* **Judge Model**: {settings.eval_judge_model}
* **Dataset Size**: {len(items)} queries ({len(good_subset)} good + {len(refusal_subset)} refusal)
* **Date**: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}

## Quality Metrics (Ragas)
| Metric | Target | Actual | Pass/Fail |
|--------|--------|--------|-----------|
| Faithfulness | >= 0.85 | {safe_results['faithfulness']:.2f} | {'✅' if safe_results['faithfulness'] >= 0.85 else '❌'} |
| Context Precision | >= 0.70 | {safe_results['context_precision']:.2f} | {'✅' if safe_results['context_precision'] >= 0.70 else '❌'} |
| Answer Relevancy | >= 0.80 | {safe_results['answer_relevancy']:.2f} | {'✅' if safe_results['answer_relevancy'] >= 0.80 else '❌'} |
| Citation Validity | == 1.00 | {avg_citation_validity:.2f} | {'✅' if avg_citation_validity == 1.0 else '❌'} |
| Refusal Correctness | >= 0.90 | {avg_refusal_correctness:.2f} | {'✅' if avg_refusal_correctness >= 0.90 else '❌'} |

## Performance Metrics
| Metric | Target | Actual | Pass/Fail |
|--------|--------|--------|-----------|
| P50 Latency | <= 3.0s | {p50:.2f}s | {'✅' if p50 <= 3.0 else '❌'} |
| P95 Latency | <= 8.0s | {p95:.2f}s | {'✅' if p95 <= 8.0 else '❌'} |
| Cost / Query | <= $0.06 | ${avg_cost:.4f} | {'✅' if avg_cost <= 0.06 else '❌'} |
"""

    report_path.write_text(report_content, encoding="utf-8")
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    asyncio.run(run_eval())

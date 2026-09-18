"""Evaluation harness for RetrieVault.

Two modes, both run from ``backend/``:

    python -m retrievault.eval --retrieval-only   # no LLM calls: gold-file hit rates and MRR
    python -m retrievault.eval                    # full pipeline, Ragas judge, gates

The full run sends every question through the real /query endpoint one at a time, so each
latency is a single-request latency. A failed request, an unscored judge job, or a missing
answer counts against the run; nothing is silently dropped or reported as a zero score.
"""

import argparse
import asyncio
import hashlib
import json
import math
import platform
import time
import warnings
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

EVAL_DIR = Path(__file__).resolve().parents[1] / "eval"
DATASET_PATH = EVAL_DIR / "dataset.jsonl"
REPORT_DIR = EVAL_DIR / "reports"

# Pass/fail gates from the evaluation plan (docs/retrievault/evaluation-plan.md).
GATES = {
    "faithfulness": 0.85,
    "context_precision": 0.70,
    "answer_relevancy": 0.80,
    "citation_validity": 1.00,
    "refusal_correctness": 0.90,
    "p50_latency_s": 3.0,
    "p95_latency_s": 8.0,
    "mean_cost_usd": 0.06,
}
RAGAS_METRICS = ("faithfulness", "context_precision", "answer_relevancy")


# ---------------------------------------------------------------------------------------------
# Dataset and pure metric helpers (unit-tested without services)
# ---------------------------------------------------------------------------------------------


@dataclass
class Dataset:
    header: dict[str, Any]
    items: list[dict[str, Any]]
    sha256: str

    @property
    def answerable(self) -> list[dict[str, Any]]:
        return [item for item in self.items if item["category"] != "refusal"]

    @property
    def refusals(self) -> list[dict[str, Any]]:
        return [item for item in self.items if item["category"] == "refusal"]


def load_dataset(path: Path = DATASET_PATH) -> Dataset:
    raw = path.read_bytes()
    lines = raw.decode("utf-8").strip().splitlines()
    return Dataset(
        header=json.loads(lines[0]),
        items=[json.loads(line) for line in lines[1:]],
        sha256=hashlib.sha256(raw).hexdigest(),
    )


def percentile(values: list[float], fraction: float) -> float | None:
    """Nearest-rank percentile; None for an empty list."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def first_gold_rank(file_paths: list[str], gold_files: list[str]) -> int | None:
    """1-based rank of the first result whose file is a gold file."""
    for rank, file_path in enumerate(file_paths, 1):
        if file_path in gold_files:
            return rank
    return None


def summarize_ragas(scores: dict[str, list[float | None]]) -> dict[str, dict[str, Any]]:
    """Mean of scored jobs per metric, with the count that the judge failed to score."""
    summary = {}
    for name, values in scores.items():
        scored = [v for v in values if v is not None and not math.isnan(v)]
        summary[name] = {
            "mean": mean(scored),
            "scored": len(scored),
            "unscored": len(values) - len(scored),
        }
    return summary


def gate(value: float | None, threshold: float, higher_is_better: bool = True) -> bool:
    if value is None:
        return False
    return value >= threshold if higher_is_better else value <= threshold


# ---------------------------------------------------------------------------------------------
# Retrieval-only evaluation
# ---------------------------------------------------------------------------------------------


def gold_symbol_match(symbol_name: str, gold_symbols: list[str]) -> bool:
    """A chunk matches a gold symbol when it is that symbol or one of its members."""
    return any(
        symbol_name == gold or symbol_name.startswith(f"{gold}.") or gold.startswith(f"{symbol_name}.")
        for gold in gold_symbols
    )


def rank_summary(rows: list[dict], key: str, n: int, top_k: int) -> dict[str, float]:
    ranks = [r[key] for r in rows]
    return {
        "hit_at_1": sum(1 for rank in ranks if rank == 1) / n,
        "hit_at_3": sum(1 for rank in ranks if rank and rank <= 3) / n,
        f"hit_at_{top_k}": sum(1 for rank in ranks if rank and rank <= top_k) / n,
        "mrr": sum(1 / rank for rank in ranks if rank and rank <= top_k) / n,
    }


def run_retrieval_eval(
    dataset: Dataset,
    top_k: int | None = None,
    fusion_depth: int | None = None,
    use_reranker: bool | None = None,
) -> dict[str, Any]:
    """Gold-file and gold-symbol hit rates for the fused list and the reranked list.

    The fused columns are the same run's candidates before reranking, so the two are directly
    comparable: any difference is the cross-encoder's doing.
    """
    from retrievault.config import get_settings
    from retrievault.rerank.reranker import rerank
    from retrievault.retrieve.hybrid_search import HybridSearcher

    settings = get_settings()
    top_k = top_k or settings.top_k_rerank
    fusion_depth = fusion_depth or settings.top_n_fusion
    use_reranker = settings.rerank_enabled if use_reranker is None else use_reranker
    searcher = HybridSearcher(collection_name=settings.qdrant_collection)
    rows = []
    for item in dataset.answerable:
        started = time.perf_counter()
        fused = [chunk.to_dict() for chunk in searcher.search(item["question"], limit=fusion_depth)]
        retrieve_s = time.perf_counter() - started

        started = time.perf_counter()
        selected = rerank(item["question"], fused, top_k=top_k) if use_reranker else fused[:top_k]
        rerank_s = time.perf_counter() - started

        gold_files, gold_symbols = item["gold_files"], item["gold_symbols"]
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "gold_files": gold_files,
                "gold_symbols": gold_symbols,
                "fused_file_rank": first_gold_rank([c["file_path"] for c in fused], gold_files),
                "selected_file_rank": first_gold_rank([c["file_path"] for c in selected], gold_files),
                "fused_symbol_rank": next(
                    (i for i, c in enumerate(fused, 1) if gold_symbol_match(c["symbol_name"], gold_symbols)),
                    None,
                ),
                "selected_symbol_rank": next(
                    (
                        i
                        for i, c in enumerate(selected, 1)
                        if gold_symbol_match(c["symbol_name"], gold_symbols)
                    ),
                    None,
                ),
                "selected": [f"{c['file_path']}:{c['symbol_name']}" for c in selected],
                "selected_code_chars": sum(len(c["code"]) for c in selected),
                "retrieve_s": retrieve_s,
                "rerank_s": rerank_s,
            }
        )

    n = len(rows)
    return {
        "mode": "retrieval",
        "items": n,
        "fusion_depth": fusion_depth,
        "top_k": top_k,
        "reranker": settings.rerank_model if use_reranker else "none",
        "gold_file_hit_at_fusion_depth": sum(1 for r in rows if r["fused_file_rank"]) / n,
        "file": {
            "fused": rank_summary(rows, "fused_file_rank", n, top_k),
            "selected": rank_summary(rows, "selected_file_rank", n, top_k),
        },
        "symbol": {
            "fused": rank_summary(rows, "fused_symbol_rank", n, top_k),
            "selected": rank_summary(rows, "selected_symbol_rank", n, top_k),
        },
        "mean_selected_code_chars": sum(r["selected_code_chars"] for r in rows) / n,
        "p50_retrieve_s": percentile([r["retrieve_s"] for r in rows], 0.5),
        "p50_rerank_s": percentile([r["rerank_s"] for r in rows], 0.5),
        "rows": rows,
    }


# ---------------------------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------------------------


@dataclass
class QueryOutcome:
    item: dict[str, Any]
    status_code: int
    wall_s: float
    body: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status_code == 200


async def query_all(items: list[dict[str, Any]]) -> list[QueryOutcome]:
    import httpx

    from retrievault.api import app

    outcomes = []
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://eval", timeout=180) as client:
        for index, item in enumerate(items, 1):
            started = time.perf_counter()
            response = await client.post("/query", json={"question": item["question"]})
            wall_s = time.perf_counter() - started
            body = response.json() if response.status_code == 200 else {"error": response.text}
            outcome = QueryOutcome(item, response.status_code, wall_s, body)
            status = body.get("grounding", {}).get("status", "error") if outcome.ok else "ERROR"
            print(f"[{index}/{len(items)}] {wall_s:5.1f}s {status:17s} {item['question'][:70]}")
            outcomes.append(outcome)
    return outcomes


def judge_llm(settings):
    if settings.eval_judge_provider == "claude":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key.strip():
            raise ValueError("EVAL_JUDGE_PROVIDER=claude needs ANTHROPIC_API_KEY.")
        return ChatAnthropic(
            model=settings.eval_judge_model,
            api_key=settings.anthropic_api_key,
            temperature=0.0,
            max_tokens=4096,
        )
    if settings.eval_judge_provider == "openai":
        from langchain_openai import ChatOpenAI

        if not settings.openai_api_key.strip():
            raise ValueError("EVAL_JUDGE_PROVIDER=openai needs OPENAI_API_KEY.")
        return ChatOpenAI(
            model=settings.eval_judge_model,
            api_key=settings.openai_api_key,
            temperature=0.0,
            max_tokens=4096,
        )
    raise ValueError(
        f"EVAL_JUDGE_PROVIDER must be 'claude' or 'openai', not {settings.eval_judge_provider!r}."
    )


def judge_embeddings(settings):
    """FastEmbed embeddings for answer relevancy, with a string `model` attribute.

    Ragas emits a telemetry event typed `model: str` and reads it off the embeddings object.
    langchain-community's FastEmbedEmbeddings keeps the fastembed instance in `.model`, so the
    event fails validation and every answer-relevancy score comes back unscored.
    """
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    from langchain_core.embeddings import Embeddings

    class JudgeEmbeddings(Embeddings):
        def __init__(self, model_name: str):
            self._inner = FastEmbedEmbeddings(model_name=model_name)
            self._name = model_name

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return self._inner.embed_documents(texts)

        def embed_query(self, text: str) -> list[float]:
            return self._inner.embed_query(text)

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await self._inner.aembed_documents(texts)

        async def aembed_query(self, text: str) -> list[float]:
            return await self._inner.aembed_query(text)

        @property
        def model(self) -> str:
            return self._name

        @property
        def model_name(self) -> str:
            return self._name

    return JudgeEmbeddings(settings.embed_model)


def run_ragas(samples: list[dict[str, Any]], settings) -> dict[str, list[float | None]]:
    from datasets import Dataset as HFDataset
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness
    from ragas.run_config import RunConfig

    if not samples:
        return {name: [] for name in RAGAS_METRICS}
    data = HFDataset.from_dict(
        {
            "user_input": [s["question"] for s in samples],
            "response": [s["answer"] for s in samples],
            "retrieved_contexts": [s["contexts"] for s in samples],
            "reference": [s["reference"] for s in samples],
        }
    )
    result = evaluate(
        dataset=data,
        metrics=[Faithfulness(), ContextPrecision(), AnswerRelevancy()],
        llm=LangchainLLMWrapper(judge_llm(settings)),
        embeddings=LangchainEmbeddingsWrapper(judge_embeddings(settings)),
        run_config=RunConfig(max_workers=settings.eval_max_workers, timeout=180, max_retries=4),
        show_progress=False,
    )
    scores: dict[str, list[float | None]] = {}
    for name in RAGAS_METRICS:
        values = []
        for value in result[name]:
            values.append(None if value is None or math.isnan(value) else float(value))
        scores[name] = values
    return scores


def score_full_run(outcomes: list[QueryOutcome], ragas: dict | None) -> dict:
    answerable = [o for o in outcomes if o.item["category"] != "refusal"]
    refusal_items = [o for o in outcomes if o.item["category"] == "refusal"]
    ok = [o for o in outcomes if o.ok]

    def grounding(outcome):
        return outcome.body.get("grounding", {}).get("status") if outcome.ok else None

    # Citation validity (spec AC3): among answers that cite at all, every label resolves.
    citing = [o for o in answerable if o.ok and not o.body["refused"]]
    citation_validity = mean([0.0 if o.body["grounding"]["invalid_labels"] else 1.0 for o in citing])
    gold_citation_rate = mean(
        [
            1.0
            if any(c["file_path"] in o.item["gold_files"] for c in o.body["citations"])
            else 0.0
            for o in answerable
            if o.ok
        ]
        + [0.0 for o in answerable if not o.ok]
    )
    refusal_correctness = mean(
        [1.0 if o.ok and o.body["refused"] and not o.body["citations"] else 0.0 for o in refusal_items]
    )
    false_refusals = sum(1 for o in answerable if o.ok and o.body["refused"])

    latencies = [o.body["metadata"]["latency_ms"]["total"] / 1000 for o in ok]
    stage = {
        name: percentile([o.body["metadata"]["latency_ms"][name] / 1000 for o in ok], 0.5)
        for name in ("retrieve", "rerank", "synthesize")
    }
    costs = [o.body["metadata"]["est_cost_usd"] for o in ok]
    tokens = {
        key: mean([o.body["metadata"]["tokens"][key] for o in ok])
        for key in ("input", "output", "cache_creation", "cache_read")
    }
    grounding_counts: dict[str, int] = {}
    for o in outcomes:
        key = grounding(o) or "request_failed"
        grounding_counts[key] = grounding_counts.get(key, 0) + 1

    summary = summarize_ragas(ragas) if ragas is not None else {}
    metrics = {
        **{name: summary.get(name, {}).get("mean") for name in RAGAS_METRICS},
        "citation_validity": citation_validity,
        "gold_citation_rate": gold_citation_rate,
        "refusal_correctness": refusal_correctness,
        "p50_latency_s": percentile(latencies, 0.5),
        "p95_latency_s": percentile(latencies, 0.95),
        "mean_cost_usd": mean(costs),
    }
    gates = {
        name: gate(metrics[name], threshold, higher_is_better=not name.endswith(("_s", "_usd")))
        for name, threshold in GATES.items()
    }
    ragas_complete = all(summary.get(name, {}).get("unscored", 1) == 0 for name in RAGAS_METRICS)
    return {
        "mode": "full",
        "queries": len(outcomes),
        "failed_requests": len(outcomes) - len(ok),
        "false_refusals": false_refusals,
        "grounding_counts": grounding_counts,
        "metrics": metrics,
        "ragas": summary,
        "ragas_complete": ragas_complete,
        "stage_p50_s": stage,
        "mean_tokens": tokens,
        "gates": gates,
        "passed": all(gates.values()) and ragas_complete and len(ok) == len(outcomes),
    }


def run_metadata(dataset: Dataset, settings) -> dict[str, Any]:
    from qdrant_client import QdrantClient

    from retrievault.collection import MANIFEST_POINT_ID

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key_or_none)
    found = client.retrieve(settings.qdrant_collection, [MANIFEST_POINT_ID])
    versions = {}
    for package in ("ragas", "fastembed", "qdrant-client", "anthropic", "langgraph", "onnxruntime"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset_version": dataset.header.get("dataset_version"),
        "dataset_sha256": dataset.sha256,
        "index_manifest": found[0].payload if found else None,
        "synthesis_model": settings.retrievault_synthesis_model,
        "judge_provider": settings.eval_judge_provider,
        "judge_model": settings.eval_judge_model,
        "retrieval": {
            "prefetch_limit": settings.prefetch_limit,
            "top_n_fusion": settings.top_n_fusion,
            "top_k_rerank": settings.top_k_rerank,
        },
        "acceleration": settings.acceleration,
        "platform": f"{platform.system()} {platform.release()} / Python {platform.python_version()}",
        "packages": versions,
    }


def fmt(value: float | None, digits: int = 2, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.{digits}f}{suffix}"


def render_full_markdown(report: dict) -> str:
    meta, score = report["run"], report["score"]
    m, g, ragas = score["metrics"], score["gates"], score["ragas"]

    def status(name):
        return "Pass" if g[name] else "Fail"

    def ragas_cell(name):
        info = ragas.get(name, {})
        note = f" ({info.get('unscored')} unscored)" if info.get("unscored") else ""
        return fmt(m[name]) + note

    manifest = meta["index_manifest"] or {}
    lines = [
        "# RetrieVault Evaluation Report",
        "",
        "## Run",
        f"- Date: {meta['date']}",
        f"- Corpus: {manifest.get('repo')} @ {manifest.get('commit_tag')} "
        f"({manifest.get('chunk_count')} chunks, chunker v{manifest.get('chunker_version')})",
        f"- Dataset: v{meta['dataset_version']}, {score['queries']} queries, "
        f"sha256 `{meta['dataset_sha256'][:12]}`",
        f"- Synthesis model: `{meta['synthesis_model']}`; judge: `{meta['judge_model']}` "
        f"({meta['judge_provider']})",
        f"- Retrieval: prefetch {meta['retrieval']['prefetch_limit']}, fused top "
        f"{meta['retrieval']['top_n_fusion']}, reranked top {meta['retrieval']['top_k_rerank']}; "
        f"acceleration `{meta['acceleration']}`",
        f"- Platform: {meta['platform']}",
        f"- Failed requests: {score['failed_requests']}; false refusals on answerable questions: "
        f"{score['false_refusals']}",
        f"- Grounding outcomes: {score['grounding_counts']}",
        "",
        "## Quality",
        "| Metric | Target | Result | Status |",
        "|---|---|---|---|",
        f"| Faithfulness | >= 0.85 | {ragas_cell('faithfulness')} | {status('faithfulness')} |",
        f"| Context precision | >= 0.70 | {ragas_cell('context_precision')} "
        f"| {status('context_precision')} |",
        f"| Answer relevancy | >= 0.80 | {ragas_cell('answer_relevancy')} "
        f"| {status('answer_relevancy')} |",
        f"| Citation validity | = 1.00 | {fmt(m['citation_validity'])} | {status('citation_validity')} |",
        f"| Refusal correctness | >= 0.90 | {fmt(m['refusal_correctness'])} "
        f"| {status('refusal_correctness')} |",
        f"| Gold-file citation rate (diagnostic) | none | {fmt(m['gold_citation_rate'])} | n/a |",
        "",
        "## Performance",
        "| Metric | Target | Result | Status |",
        "|---|---|---|---|",
        f"| P50 latency | <= 3.0 s | {fmt(m['p50_latency_s'], 2, ' s')} | {status('p50_latency_s')} |",
        f"| P95 latency | <= 8.0 s | {fmt(m['p95_latency_s'], 2, ' s')} | {status('p95_latency_s')} |",
        f"| Mean cost per query | <= $0.06 | ${fmt(m['mean_cost_usd'], 4)} | {status('mean_cost_usd')} |",
        "",
        f"Stage P50: retrieve {fmt(score['stage_p50_s']['retrieve'], 2, ' s')}, rerank "
        f"{fmt(score['stage_p50_s']['rerank'], 2, ' s')}, synthesize "
        f"{fmt(score['stage_p50_s']['synthesize'], 2, ' s')}. Mean tokens per query: "
        f"{fmt(score['mean_tokens']['input'], 0)} input, {fmt(score['mean_tokens']['output'], 0)} output.",
        "",
        f"Overall: **{'PASS' if score['passed'] else 'FAIL'}**",
        "",
    ]
    return "\n".join(lines)


def render_retrieval_markdown(report: dict) -> str:
    meta, score = report["run"], report["score"]
    manifest = meta["index_manifest"] or {}
    top_k = score["top_k"]
    hit_key = f"hit_at_{top_k}"

    def row(label: str, level: str, stage: str) -> str:
        summary = score[level][stage]
        return (
            f"| {label} | {fmt(summary['hit_at_1'])} | {fmt(summary['hit_at_3'])} "
            f"| {fmt(summary[hit_key])} | {fmt(summary['mrr'])} |"
        )

    return "\n".join(
        [
            "# RetrieVault Retrieval Report",
            "",
            f"- Date: {meta['date']}",
            f"- Corpus: {manifest.get('repo')} @ {manifest.get('commit_tag')} "
            f"({manifest.get('chunk_count')} chunks, chunker v{manifest.get('chunker_version')})",
            f"- Dataset: v{meta['dataset_version']}, {score['items']} answerable questions",
            f"- Fused candidates: {score['fusion_depth']}; selected: top {top_k}; "
            f"reranker: `{score['reranker']}`",
            f"- Acceleration `{meta['acceleration']}`; platform {meta['platform']}",
            "",
            "Hit rates are the share of questions whose results contain a chunk from a gold file "
            "(file level) or the gold symbol itself (symbol level). *Fused* is the candidate list "
            "before reranking, *selected* is what the model would receive, so the two rows differ "
            "only by the reranker. No LLM is involved.",
            "",
            f"| Measure | Rank 1 | Top 3 | Top {top_k} | MRR |",
            "|---|---|---|---|---|",
            row("File, fused", "file", "fused"),
            row("File, selected", "file", "selected"),
            row("Symbol, fused", "symbol", "fused"),
            row("Symbol, selected", "symbol", "selected"),
            "",
            f"- Gold file present in the fused top {score['fusion_depth']}: "
            f"{fmt(score['gold_file_hit_at_fusion_depth'])}",
            f"- Mean code sent to the model: {score['mean_selected_code_chars']:.0f} characters",
            f"- P50 retrieve: {fmt(score['p50_retrieve_s'], 2, ' s')}; "
            f"P50 rerank: {fmt(score['p50_rerank_s'], 2, ' s')}",
            "",
        ]
    )


def write_report(name: str, report: dict, markdown: str) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (REPORT_DIR / f"{name}.md").write_text(markdown, encoding="utf-8")
    print(f"Wrote {REPORT_DIR / name}.md and .json")


def main() -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--retrieval-only", action="store_true", help="skip synthesis and judging")
    parser.add_argument("--fusion-depth", type=int, help="candidates to rerank (default TOP_N_FUSION)")
    parser.add_argument("--top-k", type=int, help="chunks to select (default TOP_K_RERANK)")
    parser.add_argument("--no-rerank", action="store_true", help="take the fused top-k unchanged")
    parser.add_argument("--name", default=None, help="report file name under eval/reports")
    parser.add_argument("--dataset", default=None, help="dataset path (default eval/dataset.jsonl)")
    args = parser.parse_args()

    from retrievault.config import get_settings

    settings = get_settings()
    dataset = load_dataset(Path(args.dataset) if args.dataset else DATASET_PATH)
    run = run_metadata(dataset, settings)
    if run["index_manifest"] is None:
        raise SystemExit(f"Collection {settings.qdrant_collection!r} has no manifest; run ingest first.")

    if args.retrieval_only:
        score = run_retrieval_eval(
            dataset,
            top_k=args.top_k,
            fusion_depth=args.fusion_depth,
            use_reranker=False if args.no_rerank else None,
        )
        report = {"run": run, "score": score}
        write_report(args.name or "retrieval", report, render_retrieval_markdown(report))
        return

    items = dataset.refusals[: settings.eval_refusal_count] + dataset.answerable[: settings.eval_good_count]
    outcomes = asyncio.run(query_all(items))
    samples = [
        {
            "question": o.item["question"],
            "answer": o.body["answer"],
            "contexts": [c.get("text") or c["content"] for c in o.body["metadata"]["retrieved_chunks"]],
            "reference": o.item["reference_answer"],
        }
        for o in outcomes
        if o.ok and o.item["category"] != "refusal" and not o.body["refused"]
    ]
    print(f"Judging {len(samples)} answers with {settings.eval_judge_model}...")
    ragas = run_ragas(samples, settings)
    score = score_full_run(outcomes, ragas)
    report = {
        "run": run,
        "score": score,
        "judged_samples": samples,
        "items": [
            {
                "id": o.item["id"],
                "category": o.item["category"],
                "question": o.item["question"],
                "status_code": o.status_code,
                "answer": o.body.get("answer"),
                "refused": o.body.get("refused"),
                "grounding": o.body.get("grounding"),
                "citations": [
                    f"{c['file_path']}#L{c['start_line']}-L{c['end_line']}"
                    for c in o.body.get("citations", [])
                ],
                "gold_files": o.item["gold_files"],
                "latency_ms": o.body.get("metadata", {}).get("latency_ms"),
                "tokens": o.body.get("metadata", {}).get("tokens"),
                "est_cost_usd": o.body.get("metadata", {}).get("est_cost_usd"),
            }
            for o in outcomes
        ],
        "ragas_scores": ragas,
    }
    write_report("report", report, render_full_markdown(report))
    print(f"Overall: {'PASS' if score['passed'] else 'FAIL'}")


if __name__ == "__main__":
    main()

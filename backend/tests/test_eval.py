import math

import pytest

from retrievault.eval import (
    DATASET_PATH,
    GATES,
    QueryOutcome,
    first_gold_rank,
    load_dataset,
    percentile,
    score_full_run,
    summarize_ragas,
)


def test_percentile_is_nearest_rank():
    values = [5.0, 1.0, 3.0, 2.0, 4.0]
    assert percentile(values, 0.5) == 3.0
    assert percentile(values, 0.95) == 5.0
    assert percentile(list(range(1, 101)), 0.95) == 95
    assert percentile([], 0.5) is None


def test_first_gold_rank():
    assert first_gold_rank(["a.py", "b.py", "a.py"], ["b.py"]) == 2
    assert first_gold_rank(["a.py"], ["z.py"]) is None


def test_unscored_judge_jobs_are_counted_not_turned_into_zero():
    summary = summarize_ragas({"faithfulness": [1.0, None, math.nan, 0.5]})

    assert summary["faithfulness"] == {"mean": 0.75, "scored": 2, "unscored": 2}
    assert summarize_ragas({"faithfulness": [None]})["faithfulness"]["mean"] is None


def test_shipped_dataset_is_curated_and_consistent():
    dataset = load_dataset(DATASET_PATH)

    assert dataset.header["corpus_tag"] == "0.136.3"
    assert dataset.header["dataset_version"] == "1.1"
    assert len(dataset.answerable) == 45 and len(dataset.refusals) == 5
    assert len({item["id"] for item in dataset.items}) == 50
    for item in dataset.answerable:
        assert item["category"] in {"factual", "location", "how-it-works"}
        assert item["gold_files"] and all(f.startswith("fastapi/") for f in item["gold_files"])
        assert item["reference_answer"].strip()


def outcome(category, *, refused=False, citations=(), invalid=(), gold=("fastapi/a.py",), ok=True):
    item = {"id": "x", "question": "q", "category": category, "gold_files": list(gold)}
    if not ok:
        return QueryOutcome(item, 500, 1.0, {"error": "boom"})
    body = {
        "answer": "a",
        "refused": refused,
        "citations": [{"file_path": f} for f in citations],
        "grounding": {"status": "refused" if refused else "grounded", "invalid_labels": list(invalid)},
        "metadata": {
            "latency_ms": {"retrieve": 100, "rerank": 200, "synthesize": 1700, "total": 2000},
            "tokens": {"input": 3000, "output": 200, "cache_creation": 0, "cache_read": 0},
            "est_cost_usd": 0.012,
        },
    }
    return QueryOutcome(item, 200, 2.0, body)


def ragas(value):
    return {"faithfulness": [value], "context_precision": [value], "answer_relevancy": [value]}


def test_full_run_passes_only_when_every_gate_and_request_passes():
    outcomes = [
        outcome("factual", citations=["fastapi/a.py"]),
        outcome("refusal", refused=True, gold=()),
    ]

    score = score_full_run(outcomes, ragas(0.9))

    assert score["passed"] is True
    assert score["metrics"]["citation_validity"] == 1.0
    assert score["metrics"]["refusal_correctness"] == 1.0
    assert score["metrics"]["mean_cost_usd"] == pytest.approx(0.012)
    assert set(score["gates"]) == set(GATES)


def test_failed_request_and_invalid_labels_fail_the_run():
    outcomes = [
        outcome("factual", citations=["fastapi/a.py"], invalid=["[S9]"]),
        outcome("factual", ok=False),
        outcome("refusal", refused=False, citations=["fastapi/a.py"], gold=()),
    ]

    score = score_full_run(outcomes, ragas(0.9))

    assert score["failed_requests"] == 1
    assert score["metrics"]["citation_validity"] == 0.0
    assert score["metrics"]["refusal_correctness"] == 0.0
    # The failed request counts against the gold-file citation rate instead of vanishing.
    assert score["metrics"]["gold_citation_rate"] == 0.5
    assert score["passed"] is False


def test_unscored_ragas_blocks_a_pass():
    outcomes = [outcome("factual", citations=["fastapi/a.py"]), outcome("refusal", refused=True, gold=())]

    score = score_full_run(outcomes, ragas(None))

    assert score["ragas_complete"] is False
    assert score["metrics"]["faithfulness"] is None
    assert score["passed"] is False

# Retrievault Evaluation Report

## Metadata
* **Corpus**: fastapi/fastapi @ 0.136.3
* **Synthesis Model**: claude-sonnet-4-6
* **Judge Model**: gpt-4o-mini
* **Dataset Size**: 5 queries (3 good + 2 refusal)
* **Date**: 2026-07-03T20:58:16Z

## Quality Metrics (Ragas)
| Metric | Target | Actual | Pass/Fail |
|--------|--------|--------|-----------|
| Faithfulness | >= 0.85 | 0.00 | ❌ |
| Context Precision | >= 0.70 | 0.00 | ❌ |
| Answer Relevancy | >= 0.80 | 0.00 | ❌ |
| Citation Validity | == 1.00 | 1.00 | ✅ |
| Refusal Correctness | >= 0.90 | 1.00 | ✅ |

## Performance Metrics
| Metric | Target | Actual | Pass/Fail |
|--------|--------|--------|-----------|
| P50 Latency | <= 3.0s | 7.70s | ❌ |
| P95 Latency | <= 8.0s | 12.03s | ❌ |
| Cost / Query | <= $0.06 | $0.0037 | ✅ |

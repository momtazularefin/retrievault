# RetrieVault Evaluation Report

## Run
- Date: 2026-09-17T20:45:12Z
- Corpus: fastapi/fastapi @ 0.136.3 (838 chunks, chunker v2)
- Dataset: v1.1, 50 queries, sha256 `dc7a263f6f39`
- Synthesis model: `claude-sonnet-4-6`; judge: `claude-haiku-4-5-20251001` (claude)
- Retrieval: prefetch 50, fused top 12, reranked top 6; acceleration `none`
- Platform: Windows 11 / Python 3.12.13
- Failed requests: 0; false refusals on answerable questions: 2
- Grounding outcomes: {'refused': 7, 'grounded': 43}

## Quality
| Metric | Target | Result | Status |
|---|---|---|---|
| Faithfulness | >= 0.85 | 0.91 | Pass |
| Context precision | >= 0.70 | 0.70 | Pass |
| Answer relevancy | >= 0.80 | 0.84 | Pass |
| Citation validity | = 1.00 | 1.00 | Pass |
| Refusal correctness | >= 0.90 | 1.00 | Pass |
| Gold-file citation rate (diagnostic) | none | 0.93 | n/a |

## Performance
| Metric | Target | Result | Status |
|---|---|---|---|
| P50 latency | <= 3.0 s | 8.70 s | Fail |
| P95 latency | <= 8.0 s | 14.14 s | Fail |
| Mean cost per query | <= $0.06 | $0.0090 | Pass |

Stage P50: retrieve 0.04 s, rerank 3.10 s, synthesize 5.69 s. Mean tokens per query: 1833 input, 232 output.

Overall: **FAIL**

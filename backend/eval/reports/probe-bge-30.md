# RetrieVault Retrieval Report

- Date: 2026-09-17T20:15:05Z
- Corpus: fastapi/fastapi @ 0.136.3 (838 chunks, chunker v2)
- Dataset: vparaphrase-probe-1.0, 15 answerable questions
- Fused candidates: 30; selected: top 6; reranker: `BAAI/bge-reranker-base`
- Acceleration `gpu`; platform Windows 11 / Python 3.12.13

Hit rates are the share of questions whose results contain a chunk from a gold file (file level) or the gold symbol itself (symbol level). *Fused* is the candidate list before reranking, *selected* is what the model would receive, so the two rows differ only by the reranker. No LLM is involved.

| Measure | Rank 1 | Top 3 | Top 6 | MRR |
|---|---|---|---|---|
| File, fused | 0.40 | 0.73 | 0.80 | 0.57 |
| File, selected | 0.47 | 0.60 | 0.60 | 0.53 |
| Symbol, fused | 0.33 | 0.40 | 0.53 | 0.39 |
| Symbol, selected | 0.40 | 0.40 | 0.53 | 0.43 |

- Gold file present in the fused top 30: 0.93
- Mean code sent to the model: 6208 characters
- P50 retrieve: 0.11 s; P50 rerank: 8.83 s

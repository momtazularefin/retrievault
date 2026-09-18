# RetrieVault Retrieval Report

- Date: 2026-09-18T16:20:01Z
- Corpus: fastapi/fastapi @ 0.136.3 (838 chunks, chunker v2)
- Dataset: v1.1, 45 answerable questions
- Fused candidates: 12; selected: top 6; reranker: `BAAI/bge-reranker-base`
- Acceleration `none`; platform Windows 11 / Python 3.12.13

Hit rates are the share of questions whose results contain a chunk from a gold file (file level) or the gold symbol itself (symbol level). *Fused* is the candidate list before reranking, *selected* is what the model would receive, so the two rows differ only by the reranker. No LLM is involved.

| Measure | Rank 1 | Top 3 | Top 6 | MRR |
|---|---|---|---|---|
| File, fused | 0.82 | 0.91 | 1.00 | 0.88 |
| File, selected | 0.80 | 0.91 | 0.96 | 0.86 |
| Symbol, fused | 0.80 | 0.91 | 0.98 | 0.87 |
| Symbol, selected | 0.80 | 0.89 | 0.93 | 0.85 |

- Gold file present in the fused top 12: 1.00
- Mean code sent to the model: 5107 characters
- P50 retrieve: 0.05 s; P50 rerank: 3.72 s

# RetrieVault Retrieval Report

- Date: 2026-09-17T20:12:33Z
- Corpus: fastapi/fastapi @ 0.136.3 (838 chunks, chunker v2)
- Dataset: v1.1, 45 answerable questions
- Fused candidates: 8; selected: top 6; reranker: `BAAI/bge-reranker-base`
- Acceleration `gpu`; platform Windows 11 / Python 3.12.13

Hit rates are the share of questions whose results contain a chunk from a gold file (file level) or the gold symbol itself (symbol level). *Fused* is the candidate list before reranking, *selected* is what the model would receive, so the two rows differ only by the reranker. No LLM is involved.

| Measure | Rank 1 | Top 3 | Top 6 | MRR |
|---|---|---|---|---|
| File, fused | 0.84 | 0.91 | 0.98 | 0.89 |
| File, selected | 0.80 | 0.91 | 1.00 | 0.87 |
| Symbol, fused | 0.84 | 0.91 | 0.98 | 0.89 |
| Symbol, selected | 0.80 | 0.91 | 0.98 | 0.86 |

- Gold file present in the fused top 8: 1.00
- Mean code sent to the model: 4976 characters
- P50 retrieve: 0.10 s; P50 rerank: 1.04 s

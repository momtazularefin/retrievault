# RetrieVault Retrieval Report

- Date: 2026-09-17T20:17:33Z
- Corpus: fastapi/fastapi @ 0.136.3 (838 chunks, chunker v2)
- Dataset: vparaphrase-probe-1.0, 15 answerable questions
- Fused candidates: 12; selected: top 6; reranker: `BAAI/bge-reranker-base`
- Acceleration `gpu`; platform Windows 11 / Python 3.12.13

Hit rates are the share of questions whose results contain a chunk from a gold file (file level) or the gold symbol itself (symbol level). *Fused* is the candidate list before reranking, *selected* is what the model would receive, so the two rows differ only by the reranker. No LLM is involved.

| Measure | Rank 1 | Top 3 | Top 6 | MRR |
|---|---|---|---|---|
| File, fused | 0.40 | 0.73 | 0.80 | 0.57 |
| File, selected | 0.47 | 0.60 | 0.73 | 0.56 |
| Symbol, fused | 0.33 | 0.47 | 0.53 | 0.40 |
| Symbol, selected | 0.33 | 0.47 | 0.53 | 0.41 |

- Gold file present in the fused top 12: 0.80
- Mean code sent to the model: 6359 characters
- P50 retrieve: 0.08 s; P50 rerank: 1.59 s

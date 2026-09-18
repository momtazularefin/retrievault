# RetrieVault Evaluation

What is measured, how, and what the current numbers are — including the gates that fail.

```bash
cd backend
uv run python -m retrievault.eval --retrieval-only   # no LLM: hit rates and MRR, about a minute
uv run python -m retrievault.eval                    # 50 queries + judge, about $0.60
```

Both write Markdown and JSON to `eval/reports/`, carrying the index manifest, dataset hash, model
ids, retrieval settings, acceleration mode, platform, and package versions. A report names the
index it measured.

---

## 1. Current results

| Metric | Target | Measured | |
|---|---|---|---|
| Faithfulness (Ragas) | >= 0.85 | **0.91** | Pass |
| Context precision (Ragas) | >= 0.70 | **0.70** | Pass |
| Answer relevancy (Ragas) | >= 0.80 | **0.84** | Pass |
| Citation validity | = 1.00 | **1.00** | Pass |
| Refusal correctness | >= 0.90 | **1.00** | Pass |
| Cost per query | <= $0.06 | **$0.0090** | Pass |
| P50 latency | <= 3.0 s | **8.70 s** | **Fail** |
| P95 latency | <= 8.0 s | **14.14 s** | **Fail** |

50 queries (45 answerable, 5 refusal probes) against the running service, one at a time, on
laptop CPU. Zero failed requests, zero unscored judge jobs, and zero answers needing a corrective
retry. Two answerable questions were refused where retrieval missed — a refusal, not a
fabrication. Dataset v1.1, judge `claude-haiku-4-5-20251001`, index of 838 chunks at chunker v2.

**The latency gates fail, and the breakdown says why.** Stage P50: retrieve 0.04 s, rerank 3.10 s,
synthesize 5.69 s. Generation alone is roughly twice the 3-second P50 target, which was set during
planning before anything had been measured. The honest options are streaming (moving the
user-visible number to time-to-first-token), turning off the reranker (measured as equal quality on
this corpus, saving about 3 s), a GPU (rerank drops to 1.7 s), or a smaller synthesis model. None
was applied here, because this run is the baseline they would be measured against.

Context precision landing exactly on its threshold is worth treating as a coin flip rather than a
pass: repeated runs of the same configuration varied by about 0.03 on the judged metrics.

---

## 2. What each metric means

### Quality, judged by an LLM

The judge is `claude-haiku-4-5-20251001`, deliberately not the synthesis model, fixed by the
evaluation plan for reproducibility.

- **Faithfulness** — the answer is decomposed into atomic claims, and each is checked against the
  retrieved contexts. Reference-free. This is the hallucination measure.
- **Context precision** — for each retrieved context, is it useful for arriving at the reference
  answer, weighted by rank. Reference-dependent, so it is only as good as the curated answers.
- **Answer relevancy** — questions are generated from the answer and compared with the real
  question by embedding similarity. Penalises evasive and padded answers.

Scored over answerable, non-refused answers, with the count reported.

> **An unscored job is not a zero.** If the judge times out, is rate-limited, or returns something
> unparseable, that item is reported as *unscored* and any unscored job blocks a pass. The
> previous harness mapped both NaN and caught exceptions to 0.0, which is how an earlier report
> came to publish "Faithfulness 0.00 — Fail" for a system that was answering correctly.

The judge receives exactly the text the model received, header and enclosing signature included —
not the bare code. Judging faithfulness against less context than the generator had penalises
correct answers: "`Schema` is defined in `fastapi/openapi/models.py`" is supported by the chunk's
location header, and scored 0.00 until the judge was given it.

### Grounding, checked mechanically

- **Citation validity** — in every citing answer, each `[S#]` resolves to a chunk that was
  actually supplied. Gated at 1.00. This is a deterministic check, not a judgement.
- **Refusal correctness** — each unanswerable probe is refused with the mandated opening sentence
  and zero citations.
- **Gold-file citation rate** — a diagnostic, not a gate: how often an answer cites one of the
  question's gold files.

What citation validity proves is narrow and worth stating: the model cited a source that was in
front of it. It does not prove the chunk supports the sentence — that is what faithfulness
estimates, and an estimate is what it stays.

### Performance

Queries run **one at a time**, so every latency is a single-request latency. Percentiles are
nearest-rank over successful requests. Cost is computed from all four Anthropic token counters
(uncached input, output, cache writes at 1.25x, cache reads at 0.1x).

A failed request is counted, reported, and blocks a pass; it is never dropped from a denominator.

### Retrieval, with no LLM at all

`--retrieval-only` reports gold-**file** and gold-**symbol** hit rates at rank 1, top 3, and top k,
plus MRR, for both the fused candidate list and the final selected list of the same run. The
difference between those two rows is precisely the reranker's contribution. It costs nothing to
run, which is why it is the harness used for experiments.

---

## 3. Does the reranker earn its latency?

The cross-encoder is by far the most expensive stage, and it had never been compared against not
having it. Measured on the 45 curated questions, top 6 selected, on this laptop's integrated GPU:

| Configuration | file@1 | file@6 | file MRR | symbol@1 | symbol@6 | symbol MRR | rerank P50 |
|---|---|---|---|---|---|---|---|
| **Fusion only, no reranker** | **0.822** | 0.978 | **0.879** | 0.800 | 0.978 | **0.867** | — |
| bge-reranker-base, 30 candidates | 0.800 | 0.933 | 0.850 | 0.800 | 0.911 | 0.836 | 6.21 s |
| bge-reranker-base, 12 candidates | 0.800 | 0.956 | 0.861 | 0.800 | 0.933 | 0.850 | 1.70 s |
| bge-reranker-base, 8 candidates | 0.800 | 1.000 | 0.866 | 0.800 | 0.978 | 0.864 | 1.04 s |
| ms-marco-MiniLM-L-6-v2, 30 candidates | 0.778 | 0.978 | 0.856 | 0.667 | 0.978 | 0.783 | 2.10 s |

Read the first two rows together: **plain reciprocal rank fusion scores the same or better than
the reranked list**, and reranking 30 candidates costs 6.2 seconds. Note also that the reranker
does *worse* with more candidates — more room to promote something wrong.

What changed as a result:

- `TOP_N_FUSION` dropped from 30 to 12. Same or better quality, 3.6x less rerank time.
- `RERANK_ENABLED` exists so the measured alternative is one environment variable away.
- The reranker stays on by default: it is a specified stage of the system, and the evidence says
  "not demonstrated on this corpus", not "cross-encoders do not work".

Candidate depth 8 scored nominally best and was **not** adopted. Choosing the best point of a
sweep on the same 45 questions used to report results is overfitting, and a 45-item set cannot
resolve a 0.02 difference.

### The counter-test

Those questions mostly name the symbol they ask about ("What does `analyze_param` return?"), which
flatters lexical retrieval. `eval/paraphrase-probe.jsonl` asks 15 of the same things in a
developer's words, without naming the symbol:

| Configuration | file@1 | file@6 | file MRR | symbol@1 | symbol MRR |
|---|---|---|---|---|---|
| Fusion only, 30 candidates | 0.400 | **0.800** | **0.572** | 0.333 | 0.400 |
| bge-reranker-base, 30 candidates | **0.467** | 0.600 | 0.533 | **0.400** | **0.433** |
| bge-reranker-base, 12 candidates | **0.467** | 0.733 | 0.561 | 0.333 | 0.406 |

Mixed, on 15 items: the reranker is better at rank 1 and worse at top-6 recall. That is consistent
with what a cross-encoder is for — sharpening the top of a list — and not enough evidence to
overturn the main measurement. Both are published rather than the convenient one.

---

## 4. What the chunker rewrite changed

Same questions, same query pipeline, different index:

| Index | file@1 | file@6 | file MRR |
|---|---|---|---|
| Chunker v1 index as deployed (333 chunks, 23% over the 512-token window) | 0.622 | 0.844 | 0.709 |
| Chunker v2 (838 chunks, 0.6% over) | **0.800** | **0.956** | **0.861** |

The v1 chunker also silently omitted decorators, class attributes, and module-level `if`/`try`
blocks, and its last revision would have given 86 method chunks the line span of their entire
class.

---

## 5. Known limits of this evaluation

- **45 answerable questions is directional, not a benchmark.** Treat differences under about 0.05
  as noise.
- **The questions came from a generator, then were hand-corrected against the source.** Their
  phrasing still names symbols, which favours lexical retrieval (hence the paraphrase probe).
- **Judge variance is real.** Two runs of the same configuration differed by about 0.03 on
  faithfulness and context precision. One judge, fixed temperature, full-set means, per-item scores
  kept in the JSON report.
- **Context precision is reference-dependent**, so it inherits any imprecision in the curated
  reference answers.
- **Latency is a laptop number.** Measured with `ACCELERATION=none` on a Ryzen 7 8845HS; the report
  records the platform. There is no deployment, so there is no production latency to quote.

# Phase 04 Report — CRAG + Rerank + Contextual Retrieval

**Pipeline:** LangGraph StateGraph (CRAG) + BAAI/bge-reranker-base + FinanceBench_v2 (contextual prefix)
**vs Phase03b:** GPT-4o-mini + hybrid retrieval + metadata filter (FinanceBench original)
**Eval set:** 100 queries from FinanceBench

---

## What Changed

| Component | Phase03b | Phase04 |
|---|---|---|
| Pipeline | Simple agent (tool loop) | CRAG (grade → relax_filter → generate) |
| Reranker | None | BAAI/bge-reranker-base, top-10 |
| Collection | FinanceBench (27,462 chunks, 79 docs) | FinanceBench_v2 (84 docs, contextual prefix) |
| Contextual prefix | No | `Company: X | Document: Y | Year: Z` |

---

## Tier 1 — Retrieval

| Metric | Phase03b | Phase04 | Delta |
|--------|----------|---------|-------|
| Recall@6 | 0.940 | 0.950 | +0.010 |
| Precision@6 | 0.406 | 0.427 | +0.021 |
| MRR | 0.630 | 0.578 | -0.051 |

Retrieval misses: 6 → 5

### Resolved misses
- ACTIVISIONBLIZZARD_2019_10K
- KRAFTHEINZ_2019_10K

### New misses
- JOHNSON_JOHNSON_2022Q4_EARNINGS

---

## Tier 2 — Generation (judge v2)

| Metric | Phase03b | Phase04 | Delta |
|--------|----------|---------|-------|
| A\|GT Correctness | 3.23 | 3.02 | -0.21 |

Correct (>=4): Phase03b=51/100  Phase04=47/100

### Calibration (30 human labels)

| | Phase03b | Phase04 |
|-|----------|---------|
| TPR | 0.88 | 0.82 |
| TNR | 0.92 | 0.92 |

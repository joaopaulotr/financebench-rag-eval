# Baseline Report — Phase 02

**Date:** 2026-05-19
**Pipeline:** GPT-4o-mini + text-embedding-3-small + Qdrant (k=6, no filtering)
**Eval set:** 100 queries from FinanceBench

---

## Tier 1 — Retrieval Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Recall@6 | **0.830** | Correct doc retrieved in top-6 |
| Precision@6 | **0.422** | Avg fraction of retrieved chunks from correct doc |
| MRR | **0.646** | Mean Reciprocal Rank of first correct chunk |

83/100 queries retrieved the correct document.

### Retrieval Misses by Document (17 total)

| Document | Misses |
|----------|--------|
| JOHNSON_JOHNSON_2022_10K | 3 |
| JOHNSON_JOHNSON_2023_8K_dated-2023-08-30 | 3 |
| ADOBE_2022_10K | 2 |
| JOHNSON_JOHNSON_2022Q4_EARNINGS | 2 |
| ADOBE_2015_10K | 1 |
| ADOBE_2016_10K | 1 |
| ADOBE_2017_10K | 1 |
| AMCOR_2023_10K | 1 |
| AMD_2015_10K | 1 |
| JOHNSON_JOHNSON_2023Q2_EARNINGS | 1 |

**Pattern:** Johnson & Johnson (multiple doc types) + Adobe account for most retrieval failures.

---

## Tier 2 — Generation (LLM-as-Judge, scale 1–5)

| Metric | Avg Score | Description |
|--------|-----------|-------------|
| Context Relevance C\|Q | **3.04** | Retrieved context relevant to question |
| Answer Faithfulness A\|C | **3.30** | Answer grounded in context |
| Answer Relevance A\|Q | **3.96** | Answer addresses the question |
| Answer Correctness A\|GT | **3.19** | Answer matches ground truth |

Judge classified 53/100 answers as correct (A|GT >= 4).

---

## Judge Calibration (30 human labels)

| Metric | Value |
|--------|-------|
| Human accuracy | **0.27** (8/30 correct) |
| TPR (sensitivity) | **1.00** — judge catches correct answers |
| TNR (specificity) | **0.86** — judge catches wrong answers |

> **Key finding:** A|Q judge (no ground truth) had TNR=0.55 — approved numerically wrong answers with score 5/5.
> Fix: added A|GT judge with ground truth in prompt → TNR improved to 0.86.
> Lesson: without calibration against human labels, eval is theater.

---

## Primary Failure Hypotheses

1. **Retrieval cross-empresa** — semantic similarity across SEC filings causes wrong company chunks to be retrieved. JnJ + Adobe = 14/17 retrieval misses.

2. **Numerical error in generation** — model retrieves correct doc but calculates wrong value. Judge blind to this (fluency bias).

3. **Agent multi-retrieval** — complex multi-year queries trigger 5–6 tool calls, diluting precision and introducing cross-company noise.

---

## Proposed Fixes (Phase 3)

| Problem | Fix | Priority |
|---------|-----|----------|
| Retrieval cross-empresa | Metadata filtering (LLM extracts company from query) | HIGH |
| Numerical errors | Hybrid retrieval dense + BM25 | MEDIUM |
| Agent noise | Cap tool calls + sub-query routing | MEDIUM |

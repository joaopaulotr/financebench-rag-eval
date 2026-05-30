# Phase 03 Report — Iteracao com Fixes

**Date:** 2026-05-21
**Pipeline:** GPT-4o-mini + text-embedding-3-small + Qdrant hybrid (dense + BM25) + metadata filtering
**Eval set:** 100 queries from FinanceBench

---

## Tier 1 -- Retrieval Comparison

| Metric | Phase02 (baseline) | Phase03 (fixes) | Delta |
|--------|-------------------|-----------------|-------|
| Recall@6 | **0.830** | **0.840** | +0.010 |
| Precision@6 | **0.422** | **0.405** | -0.017 |
| MRR | **0.646** | **0.594** | -0.052 |

Retrieval misses: 17 → 16

### Resolved misses
- AMCOR_2023_10K

### New misses
None

---

## Tier 2 -- Generation Comparison

| Metric | Phase02 | Phase03 | Delta |
|--------|---------|---------|-------|
| C\|Q Context Relevance | **3.04** | **3.32** | +0.28 |
| A\|C Faithfulness | **3.30** | **4.08** | +0.78 |
| A\|Q Answer Relevance | **3.96** | **4.29** | +0.33 |
| A\|GT Correctness | **3.19** | **3.39** | +0.20 |

Judge classified Phase02=53/100 vs Phase03=58/100 answers as correct (A|GT >= 4).

---

## Fixes Implemented

| Fix | Approach | Target Failure Mode |
|-----|----------|-------------------|
| Metadata filtering | LLM extracts company name, Qdrant pre-filters candidates | wrong_doc (16/100) |
| Hybrid retrieval | dense (text-embedding-3-small) + BM25 (FastEmbed), fused via RRF | numerical_error (14/100) |

## Fixes NOT Implemented (and why)

| Fix | Reason skipped |
|-----|---------------|
| Semantic chunking | Requires full re-ingestion, impact uncertain vs hybrid |
| Reranker (Cohere) | Adds latency + cost, lower priority than retrieval fixes |
| Query rewriting / HyDE | Adds LLM call per query, cross-company noise likely persists |

---

## Key Findings

1. Metadata filtering partially effective — resolved 1 retrieval miss(es). Adobe docs remain unretrievable via semantic search regardless of filtering, pointing to an embedding representation issue.

2. MRR dropped (0.646 to 0.594) — BM25 hybrid changes ranking, sometimes pushing the correct chunk lower. Dense-only had better rank precision for this dataset.

3. wrong_doc diagnosis was incomplete — JnJ misses are cross-document within the same company (2022_10K vs 2023_8K), not cross-company. Filtering by company name does not resolve these.

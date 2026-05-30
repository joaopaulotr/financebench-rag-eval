# Phase 03b Report — Full Corpus (+5 Missing PDFs)

**Date:** 2026-05-26
**Change from Phase03:** Ingested 4 missing Johnson & Johnson PDFs + 1 MGM PDF
**Pipeline:** GPT-4o-mini + text-embedding-3-small + Qdrant hybrid (dense + BM25) + metadata filtering
**Eval set:** 100 queries from FinanceBench

---

## What Changed

Previously missing from Qdrant:
- JOHNSON_JOHNSON_2022Q4_EARNINGS
- JOHNSON_JOHNSON_2022_10K
- JOHNSON_JOHNSON_2023Q2_EARNINGS
- JOHNSON_JOHNSON_2023_8K_dated-2023-08-30
- MGMRESORTS_2022Q4_EARNINGS

These were never ingested → retrieval impossible regardless of filtering strategy.

---

## Tier 1 -- Retrieval Comparison

| Metric | Phase03 (79 PDFs) | Phase03b (84 PDFs) | Delta |
|--------|------------------|-------------------|-------|
| Recall@6 | **0.840** | **0.940** | +0.100 |
| Precision@6 | **0.405** | **0.406** | +0.001 |
| MRR | **0.594** | **0.630** | +0.036 |

Retrieval misses: 16 → 6

### Resolved misses
- ADOBE_2015_10K
- ADOBE_2016_10K
- ADOBE_2017_10K
- JOHNSON_JOHNSON_2022Q4_EARNINGS
- JOHNSON_JOHNSON_2023Q2_EARNINGS
- JOHNSON_JOHNSON_2023_8K_dated-2023-08-30

### New misses
- ACTIVISIONBLIZZARD_2019_10K

---

## Tier 2 -- Generation Comparison

| Metric | Phase03 | Phase03b | Delta |
|--------|---------|----------|-------|
| C\|Q Context Relevance | **3.32** | **3.87** | +0.55 |
| A\|C Faithfulness | **4.08** | **4.49** | +0.41 |
| A\|Q Answer Relevance | **4.29** | **4.43** | +0.14 |
| A\|GT Correctness | **3.39** | **3.61** | +0.22 |

Judge classified Phase03=58/100 vs Phase03b=63/100 answers as correct (A|GT >= 4).

### Judge Calibration (30 human labels)

| | Phase03 | Phase03b |
|-|---------|----------|
| TPR | 0.88 | 1.00 |
| TNR | 0.68 | 0.59 |

---

## Key Findings

1. Ingesting the 5 missing PDFs resolved 6 retrieval miss(es): `ADOBE_2015_10K, ADOBE_2016_10K, ADOBE_2017_10K, JOHNSON_JOHNSON_2022Q4_EARNINGS, JOHNSON_JOHNSON_2023Q2_EARNINGS, JOHNSON_JOHNSON_2023_8K_dated-2023-08-30`

2. Metadata filtering (company name extraction) can now actually work for JnJ queries — before, chunks didn't exist in Qdrant at all.

3. Remaining JnJ misses after fix are likely cross-document within same company (2022_10K vs 2023_8K, etc.) — company-level filtering doesn't resolve year/doc-type confusion.

# FinanceBench RAG Eval

A production-grade Retrieval-Augmented Generation system evaluated against the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark (Patronus AI). Built with rigorous multi-tier eval, end-to-end observability, and full public documentation of the iteration process — including what failed and why.

![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)
![LangChain](https://img.shields.io/badge/orchestration-LangGraph-orange?style=flat-square)
![Qdrant](https://img.shields.io/badge/vector_db-Qdrant-red?style=flat-square)
![LangSmith](https://img.shields.io/badge/observability-LangSmith-yellow?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Why financial domain

- Public dataset with ground truth — objective, measurable progress
- Numerical precision is critical — forces serious eval, no vibe checks
- Benchmarks are published — results are directly comparable to state of the art
- High enterprise signal for international recruiters

---

## Architecture

```
Query
  ↓
Metadata Filter (LLM extracts company + year → Qdrant pre-filter)
  ↓
Hybrid Retrieval (text-embedding-3-small dense + BM25 sparse, top-k=6, RRF fusion)
  ↓
LangGraph Agent (GPT-4o-mini, agentic loop, recursion_limit=10)
  ↓
Response + Source Citations
```

All LLM calls traced end-to-end via LangSmith.

---

## Stack

| Component       | Tool                              | Reason                                          |
|-----------------|-----------------------------------|-------------------------------------------------|
| LLM             | GPT-4o-mini                       | Fast, cheap, sufficient for iteration           |
| Embeddings      | text-embedding-3-small            | Simple, cheap, strong baseline                  |
| Sparse          | BM25 (FastEmbedSparse)            | Exact number matching, hybrid fusion via RRF    |
| Vector DB       | Qdrant (local via Docker)         | Open source, production-ready, hybrid search    |
| Orchestration   | LangGraph (`create_agent`)        | Agentic loop with tool use                      |
| Observability   | LangSmith                         | Traces, cost tracking                           |
| Eval            | Custom multi-tier                 | Full control, calibrated against human labels   |

---

## Eval Framework

### Tier 1 — Retrieval

| Metric       | Description                                          |
|--------------|------------------------------------------------------|
| Recall@6     | Fraction of queries where correct doc is in top-6   |
| Precision@6  | Fraction of retrieved chunks from the correct doc    |
| MRR          | Mean Reciprocal Rank of first correct chunk          |

### Tier 2 — Generation (LLM-as-Judge, scale 1–5)

| Metric               | Notation | Description                                                  |
|----------------------|----------|--------------------------------------------------------------|
| Context Relevance    | C\|Q     | Retrieved chunks address the query                           |
| Answer Faithfulness  | A\|C     | Answer grounded in retrieved context (hallucination check)   |
| Answer Relevance     | A\|Q     | Answer addresses the original question                       |
| Answer Correctness   | A\|GT    | Answer matches the ground truth expected answer              |

**Judge calibration:** every judge validated against 30 human-labeled examples. TPR and TNR reported. Without calibration, eval is theater.

**Judge v2 (stricter):** A|GT prompt enforces numerical extraction + tolerance rules. Numbers >15% off ground truth score ≤ 2 regardless of reasoning quality. TNR improved from 0.75 → 0.94.

---

## Results

### Tier 1 — Retrieval

| Phase | Setup | Recall@6 | Precision@6 | MRR | Misses |
|-------|-------|----------|-------------|-----|--------|
| Phase 02 | Dense only, no filter | 0.830 | 0.422 | 0.646 | 17 |
| Phase 03 | Hybrid + company filter | 0.840 | 0.405 | 0.594 | 16 |
| Phase 03b | Hybrid + company+year filter + full corpus | **0.940** | **0.406** | **0.630** | **6** |

### Tier 2 — Generation

| Phase | C\|Q | A\|C | A\|Q | A\|GT | Correct (judge) | Correct (human-calibrated) |
|-------|------|------|------|-------|-----------------|---------------------------|
| Phase 02 | — | 3.30 | 4.29 | 3.19 | 53/100 | ~47/100 |
| Phase 03 | — | 4.08 | — | 3.39 | 58/100 | — |
| Phase 03b | 3.87 | **4.49** | **4.43** | **3.61** | 63/100 (v1) / 51/100 (v2) | **~47/100** |

### Judge Calibration (30 human labels)

| Judge | TPR | TNR | False Positives | Notes |
|-------|-----|-----|-----------------|-------|
| v1 (Phase 02) | 1.00 | 0.86 | 3/30 | Used in Phase 02 baseline |
| v1 (Phase 03b) | 0.93 | 0.75 | 4/30 | Fluency bias: approves wrong numbers |
| **v2 (Phase 03b)** | 0.71 | **0.94** | **1/30** | Strict numerical tolerance enforced |

**Key finding:** Judge v1 inflated "correct" count by +16/100 (63 nominal vs 47 human-calibrated). v2 reduces inflation to +4 — but trades off some TPR.

---

## Key Findings

1. **Corpus completeness is the biggest lever.** 5 missing PDFs (never ingested) caused 10 retrieval misses. Fixing this alone added +10pp Recall@6. Retrieval can't work without the document.

2. **Hybrid retrieval (dense + BM25) improved faithfulness, not recall.** A|C jumped +0.78 (3.30 → 4.08) because BM25 finds exact number matches. MRR dropped slightly — dense-only had better rank precision for some queries.

3. **Company+year metadata filtering resolved cross-company confusion.** Filtering by `ADOBE_2022` instead of just `ADOBE` prevents older filings from outranking the correct year.

4. **LLM-as-Judge has strong fluency bias without calibration.** Judge v1 gave 5/5 to an answer with quick ratio 1.76 when ground truth was 1.57. Prompt engineering (v2) with explicit numerical extraction cuts false positives from 4 → 1 in 30 samples.

5. **Real accuracy (~47%) is lower than nominal judge score (63%).** Always calibrate against human labels. The gap is the fluency bias: well-explained wrong numbers look correct to a judge without strict numerical rules.

---

## Failure Mode Distribution (Phase 03b)

| Mode | Count | Root Cause |
|------|-------|-----------|
| Retrieval miss | 6 | Doc not in top-6 despite being in corpus |
| Numerical error | ~15 | Model retrieves right doc, calculates wrong value |
| Incomplete | ~12 | Partial answer, misses key sub-question |
| Hallucination | ~8 | Answer not grounded in retrieved context |
| Agent timeout | ~5 | Recursion limit hit before answer produced |
| Correct | ~47 | — |

---

## Roadmap

- [x] **Phase 1** — Functional baseline (weeks 1–3)
  - [x] Dataset: FinanceBench, 84 PDFs, 27,462 chunks in Qdrant
  - [x] Fixed chunking (512 tokens, overlap 50)
  - [x] Dense retrieval (text-embedding-3-small) + GPT-4o-mini
  - [x] Initial eval: 20 queries, cross-company retrieval identified as primary failure
- [x] **Phase 2** — Eval infrastructure (weeks 4–6)
  - [x] 100-query eval set with ground truth
  - [x] Tier 1: Recall@6, Precision@6, MRR
  - [x] Tier 2: LLM-as-Judge (C|Q, A|C, A|Q, A|GT)
  - [x] Judge calibrated against 30 human labels (TPR/TNR reported)
  - [x] Baseline: Recall@6=0.830, 53/100 correct (judge), ~47/100 (human)
- [x] **Phase 3** — Error analysis + iteration (weeks 7–9)
  - [x] Failure mode classification (wrong_doc, retrieval_miss, hallucination, etc.)
  - [x] Fix 1: Hybrid retrieval — dense + BM25 (FastEmbedSparse), RRF fusion
  - [x] Fix 2: Metadata filtering — LLM extracts company+year → Qdrant pre-filter
  - [x] Corpus audit: 5 missing PDFs found and ingested (EDGAR HTM fallback)
  - [x] Phase 03b: Recall@6=0.940, 6 retrieval misses remaining
  - [x] Judge v2: strict numerical tolerance, TNR 0.75 → 0.94
  - [x] Human recalibration: real accuracy ~47/100 vs 63/100 nominal
- [ ] **Phase 4** — Polish + final writeup (weeks 10–12)
  - [ ] Post 2 published on dev.to
  - [ ] Remaining 6 retrieval misses: reranker or doc-type filter
  - [ ] Final comparative eval table
  - [ ] Post 3 published on dev.to
  - [ ] Live demo deployed

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker
- `OPENAI_API_KEY`

### Install

```bash
git clone https://github.com/<your-username>/financebench-rag-eval
cd financebench-rag-eval
uv sync
```

### Environment

```bash
cp .env.example .env
# fill OPENAI_API_KEY
```

### Run Qdrant

```bash
docker compose up -d
```

### Ingest documents

```bash
uv run python ingestion.py
```

### Run a query

```bash
uv run python backend/core.py
```

### Run eval (Phase 03b)

```bash
uv run python eval/Phase03/runBaseline_b.py     # retrieval + generation (100 queries)
uv run python eval/Phase03/runJudge_b_v2.py     # strict numerical judge scoring
uv run python eval/Phase03/report_b.py          # comparative report
uv run python eval/Phase03/calibrate_v2.py      # v1 vs v2 judge calibration
```

---

## Eval Output Files

```
eval/
  Phase02/
    results_100.csv                      # Phase 02 baseline results
    results_with_judge.csv               # Phase 02 judge scores
    human_labels_30.csv                  # 30 human labels for calibration
  Phase03/
    results_phase3b.csv                  # Phase 03b results (full corpus + fixes)
    results_phase3b_with_judge_v2.csv    # Strict numerical judge scores
    label_phase3b.csv                    # 30 human labels (Phase 03b)
    error_analysis_phase3.csv            # Failure mode classification
docs/
  phase03b_report.md                     # Phase 03 vs 03b comparative report
  judge_v2_calibration.md               # v1 vs v2 judge comparison
  judge_false_positives.md              # Cases where judge was wrong
  missing_pdfs.md                        # Corpus audit (5 missing PDFs)
```

---

## Dataset

[FinanceBench](https://github.com/patronus-ai/financebench) by Patronus AI. 10-K, 10-Q, and earnings filings from 84 public companies, with expert-annotated Q&A pairs and document-level ground truth.

---

## References

- [FinanceBench paper](https://arxiv.org/abs/2311.11944) — Patronus AI
- [6 RAG Evals](https://jxnl.co) — Jason Liu
- [LLM Evals FAQ](https://hamel.dev/blog/posts/evals-faq/) — Hamel Husain
- [LangGraph docs](https://langchain-ai.github.io/langgraph/)
- [Qdrant hybrid search](https://qdrant.tech/documentation/concepts/hybrid-queries/)

---

## License

MIT

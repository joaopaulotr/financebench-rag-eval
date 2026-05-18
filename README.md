# FinanceBench RAG Eval

A production-grade Retrieval-Augmented Generation system evaluated against the [FinanceBench](https://github.com/patronus-ai/financebench) benchmark (Patronus AI). Built with rigorous multi-tier eval, end-to-end observability, and full public documentation of the iteration process — including what failed and why.

![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)
![LangChain](https://img.shields.io/badge/orchestration-LangChain-orange?style=flat-square)
![Qdrant](https://img.shields.io/badge/vector_db-Qdrant-red?style=flat-square)
![Phoenix](https://img.shields.io/badge/observability-Arize_Phoenix-purple?style=flat-square)
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
Retrieval (text-embedding-3-small + Qdrant, top-k=6)
  ↓
Generation (GPT-4o-mini)
  ↓
Response + Source Citations
```

All steps traced end-to-end via Arize Phoenix *(Phase 2)*.

---

## Stack

| Component       | Tool                        | Reason                                          |
|-----------------|-----------------------------|-------------------------------------------------|
| LLM             | GPT-4o-mini                 | Fast, cheap, good enough for iteration          |
| Embeddings      | text-embedding-3-small      | Simple, cheap, good enough                      |
| Vector DB       | Qdrant (local via Docker)   | Open source, production-ready                   |
| Orchestration   | LangChain → LangGraph       | LangGraph wired in Phase 4                      |
| Observability   | Arize Phoenix               | Open source, industry standard                  |
| Eval            | Custom + Phoenix            | Full control, more learning                     |
| Demo deploy     | Modal / HF Spaces           | Free tier, no infra overhead                    |

---

## Eval Framework

### Tier 1 — Retrieval

| Metric       | Description                                          |
|--------------|------------------------------------------------------|
| Recall@k     | Fraction of relevant docs retrieved in top-k         |
| Precision@k  | Fraction of retrieved docs that are relevant         |
| MRR          | Mean Reciprocal Rank of first relevant result        |

### Tier 2 — Generation (LLM-as-Judge)

| Metric               | Notation | Description                                                  |
|----------------------|----------|--------------------------------------------------------------|
| Context Relevance    | C\|Q     | Retrieved chunks actually address the query                  |
| Answer Faithfulness  | A\|C     | Answer is grounded in the retrieved context                  |
| Answer Relevance     | A\|Q     | Answer actually addresses the original question              |

Judge calibration: every LLM judge validated against 30+ human-labeled examples. TPR and TNR reported. Without calibration, eval is theater.

---

## Results

| Phase      | Accuracy (manual) | Recall@k | Faithfulness | Answer Relevance | Cost/query |
|------------|--------------------|----------|--------------|------------------|------------|
| Baseline   | 2/20 (10%)         | —        | —            | —                | —          |
| +Fix 1     |                    |          |              |                  |            |
| +Fix 2     |                    |          |              |                  |            |
| Final      |                    |          |              |                  |            |

### Baseline failure analysis

Primary failure mode: **cross-company retrieval**. With 25,094 chunks from 84 companies in a single collection, semantic search retrieves chunks from the wrong company. Financial language across SEC filings is too similar — a query about 3M's revenue pulls chunks from Coca-Cola and Nike.

Confirmed that metadata filtering by source document resolves retrieval accuracy. Automatic metadata filtering (LLM extracts company → filters Qdrant) is the planned fix for Phase 3.

---

## Roadmap

- [x] Phase 1 — Functional baseline (weeks 1–3)
  - [x] Dataset selection and ingestion (FinanceBench, 84 PDFs)
  - [x] Fixed chunking (512 tokens, overlap 50) + Qdrant (25,094 chunks)
  - [x] PDF loading with PyMuPDFLoader (robust to malformed files)
  - [x] Embeddings with text-embedding-3-small
  - [x] Retrieval top-k=6 + generation with GPT-4o-mini
  - [x] Baseline eval: 20 queries, 2/20 correct (10%)
  - [x] Failure mode identified: cross-company retrieval
  - [x] Public repo with this README
- [ ] Phase 2 — Eval infrastructure (weeks 4–6)
  - [ ] 100 example eval set with ground truth
  - [ ] Tier 1 and Tier 2 metrics implemented
  - [ ] LLM-as-judge calibrated against human labels
  - [ ] Arize Phoenix traces wired end-to-end
  - [ ] Post 1 published on dev.to
- [ ] Phase 3 — Error analysis + iteration (weeks 7–9)
  - [ ] Failure mode categorization (retrieval miss, chunk boundary, hallucination, etc.)
  - [ ] Top 2 failure modes fixed and measured
  - [ ] Post 2 published on dev.to
- [ ] Phase 4 — Polish + final writeup (weeks 10–12)
  - [ ] LangGraph multi-step refactor
  - [ ] Embedding cache, rate limiting, structured logging
  - [ ] Final comparative eval table
  - [ ] Live demo deployed
  - [ ] Post 3 published on dev.to

---

## Getting Started

### Prerequisites

- Python 3.11+
- Docker
- API key: `OPENAI_API_KEY`

### Install

```bash
git clone https://github.com/<your-username>/financebench-rag-eval
cd financebench-rag-eval
uv sync
```

### Environment

```bash
cp .env.example .env
# fill in OPENAI_API_KEY
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
uv run python main.py
```

### Run baseline eval

```bash
uv run python eval/run_baseline.py
```

---

## Dataset

[FinanceBench](https://github.com/patronus-ai/financebench) by Patronus AI. 10-K and 10-Q filings from public companies, with 150 expert-annotated Q&A pairs and document-level relevance labels.

---

## References

- [FinanceBench paper](https://arxiv.org/abs/2311.11944) — Patronus AI
- [6 RAG Evals](https://jxnl.co) — Jason Liu
- [LLM Evals FAQ](https://hamel.dev/blog/posts/evals-faq/) — Hamel Husain
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [LangGraph docs](https://langchain-ai.github.io/langgraph/)

---

## License

MIT
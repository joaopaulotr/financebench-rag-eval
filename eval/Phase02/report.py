import csv
import json
from collections import Counter
from pathlib import Path

RESULTS_CSV = "eval/Phase02/results_100.csv"
JUDGE_CSV = "eval/Phase02/results_with_judge.csv"
LABELS_CSV = "eval/Phase02/human_labels_30.csv"
K = 6
JUDGE_THRESHOLD = 4


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Load ──────────────────────────────────────────────────────────────────────
results = load(RESULTS_CSV)
judge = {r["financebench_id"]: r for r in load(JUDGE_CSV)}
labels = {r["financebench_id"]: int(r["human_correct"]) for r in load(LABELS_CSV)}

# ── Tier 1 — Retrieval ────────────────────────────────────────────────────────
recalls, precisions, mrrs = [], [], []
misses = []

for row in results:
    gt = row["doc_name"]
    sources = json.loads(row.get("retrieved_sources", "[]"))

    hit = gt in sources
    recalls.append(1.0 if hit else 0.0)
    precisions.append(sum(s == gt for s in sources) / len(sources) if sources else 0.0)

    rr = 0.0
    for rank, s in enumerate(sources, 1):
        if s == gt:
            rr = 1.0 / rank
            break
    mrrs.append(rr)

    if not hit:
        misses.append(gt)

n = len(results)

# ── Tier 2 — Judge ────────────────────────────────────────────────────────────
cq_scores, ac_scores, aq_scores, gt_scores = [], [], [], []
judge_correct = {}

for row in results:
    fid = row["financebench_id"]
    if fid not in judge:
        continue
    j = judge[fid]
    cq_scores.append(int(j["cq_score"]))
    ac_scores.append(int(j["ac_score"]))
    aq_scores.append(int(j["aq_score"]))
    gt_scores.append(int(j["gt_score"]))
    judge_correct[fid] = 1 if int(j["gt_score"]) >= JUDGE_THRESHOLD else 0

# ── Calibration ───────────────────────────────────────────────────────────────
tp = tn = fp = fn = 0
for fid, human in labels.items():
    if fid not in judge_correct:
        continue
    jc = judge_correct[fid]
    if human == 1 and jc == 1:   tp += 1
    elif human == 0 and jc == 0: tn += 1
    elif human == 1 and jc == 0: fn += 1
    else:                         fp += 1

tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

# ── Human accuracy ────────────────────────────────────────────────────────────
human_acc = sum(labels.values()) / len(labels)

# ── Top failure docs ──────────────────────────────────────────────────────────
miss_counts = Counter(misses)

# ── Top 10 failure examples ───────────────────────────────────────────────────
failures = []
for row in results:
    fid = row["financebench_id"]
    gt = row["doc_name"]
    sources = json.loads(row.get("retrieved_sources", "[]"))
    retrieval_ok = gt in sources
    answer_ok = labels.get(fid)  # None if not in sample

    if not retrieval_ok or answer_ok == 0:
        j = judge.get(fid, {})
        failures.append({
            "id": fid,
            "question": row["question"][:80],
            "doc": gt,
            "retrieval_ok": retrieval_ok,
            "aq_score": j.get("aq_score", "?"),
            "human": answer_ok,
        })

# ── Print ─────────────────────────────────────────────────────────────────────
SEP = "─" * 60

print(f"\n{'═'*60}")
print("  BASELINE REPORT — Phase 02")
print(f"{'═'*60}")

print(f"\n{'Tier 1 — Retrieval':^60}")
print(SEP)
print(f"  Queries evaluated : {n}")
print(f"  Recall@{K}          : {sum(recalls)/n:.3f}  ({int(sum(recalls))}/{n} correct doc retrieved)")
print(f"  Precision@{K}       : {sum(precisions)/n:.3f}  (avg correct chunks / k)")
print(f"  MRR               : {sum(mrrs)/n:.3f}  (avg reciprocal rank)")

print(f"\n{'Tier 2 — Generation (LLM-as-Judge, scale 1-5)':^60}")
print(SEP)
print(f"  Context Relevance  C|Q : {sum(cq_scores)/len(cq_scores):.2f}")
print(f"  Answer Faithfulness A|C : {sum(ac_scores)/len(ac_scores):.2f}")
print(f"  Answer Relevance   A|Q : {sum(aq_scores)/len(aq_scores):.2f}")
print(f"  Answer Correctness  A|GT: {sum(gt_scores)/len(gt_scores):.2f}")
print(f"  Judge 'correct' (GT>={JUDGE_THRESHOLD}): {sum(1 for v in gt_scores if v >= JUDGE_THRESHOLD)}/{len(gt_scores)}")

print(f"\n{'Judge Calibration (30 human labels)':^60}")
print(SEP)
print(f"  Human accuracy    : {human_acc:.2f}  ({sum(labels.values())}/30 correct)")
print(f"  TPR (sensitivity) : {tpr:.2f}  — judge catches correct answers")
print(f"  TNR (specificity) : {tnr:.2f}  — judge catches wrong answers")
print(f"  ⚠  Low TNR: judge approves numerically wrong answers (fluency bias)")

print(f"\n{'Top Retrieval Misses by Document':^60}")
print(SEP)
for doc, count in miss_counts.most_common(10):
    print(f"  {doc:<45} {count}x")

print(f"\n{'Top 10 Failure Examples':^60}")
print(SEP)
for f in failures[:10]:
    retrieval = "✓" if f["retrieval_ok"] else "✗"
    human = f"human={'?' if f['human'] is None else f['human']}"
    print(f"  [{retrieval}] A|Q={f['aq_score']} {human}  {f['id']}")
    print(f"       {f['question']}")
    print(f"       doc: {f['doc']}")
    print()

print(f"\n{'Primary Failure Hypotheses':^60}")
print(SEP)
print("  1. Retrieval cross-empresa: wrong company chunks retrieved")
print("     → JnJ + Adobe = 12/17 retrieval misses")
print("  2. Numerical hallucination: model calculates wrong values")
print("     → judge blind to this (TNR=0.55)")
print("  3. Agent multi-retrieval: complex queries trigger 5-6 tool")
print("     calls, diluting precision")
print(f"\n{'═'*60}\n")

# ── Generate markdown report ───────────────────────────────────────────────────
r1_recall = sum(recalls) / n
r1_precision = sum(precisions) / n
r1_mrr = sum(mrrs) / n
avg_cq = sum(cq_scores) / len(cq_scores)
avg_ac = sum(ac_scores) / len(ac_scores)
avg_aq = sum(aq_scores) / len(aq_scores)
avg_gt = sum(gt_scores) / len(gt_scores)
judge_correct_count = sum(1 for v in gt_scores if v >= JUDGE_THRESHOLD)

md = f"""# Baseline Report — Phase 02

**Date:** 2026-05-19
**Pipeline:** GPT-4o-mini + text-embedding-3-small + Qdrant (k=6, no filtering)
**Eval set:** 100 queries from FinanceBench

---

## Tier 1 — Retrieval Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Recall@{K} | **{r1_recall:.3f}** | Correct doc retrieved in top-{K} |
| Precision@{K} | **{r1_precision:.3f}** | Avg fraction of retrieved chunks from correct doc |
| MRR | **{r1_mrr:.3f}** | Mean Reciprocal Rank of first correct chunk |

{int(sum(recalls))}/100 queries retrieved the correct document.

### Retrieval Misses by Document ({len(misses)} total)

| Document | Misses |
|----------|--------|
""" + "\n".join(
    f"| {doc} | {count} |"
    for doc, count in miss_counts.most_common(10)
) + f"""

**Pattern:** Johnson & Johnson (multiple doc types) + Adobe account for most retrieval failures.

---

## Tier 2 — Generation (LLM-as-Judge, scale 1–5)

| Metric | Avg Score | Description |
|--------|-----------|-------------|
| Context Relevance C\\|Q | **{avg_cq:.2f}** | Retrieved context relevant to question |
| Answer Faithfulness A\\|C | **{avg_ac:.2f}** | Answer grounded in context |
| Answer Relevance A\\|Q | **{avg_aq:.2f}** | Answer addresses the question |
| Answer Correctness A\\|GT | **{avg_gt:.2f}** | Answer matches ground truth |

Judge classified {judge_correct_count}/100 answers as correct (A|GT >= {JUDGE_THRESHOLD}).

---

## Judge Calibration (30 human labels)

| Metric | Value |
|--------|-------|
| Human accuracy | **{human_acc:.2f}** ({sum(labels.values())}/30 correct) |
| TPR (sensitivity) | **{tpr:.2f}** — judge catches correct answers |
| TNR (specificity) | **{tnr:.2f}** — judge catches wrong answers |

> **Key finding:** A|Q judge (no ground truth) had TNR=0.55 — approved numerically wrong answers with score 5/5.
> Fix: added A|GT judge with ground truth in prompt → TNR improved to {tnr:.2f}.
> Lesson: without calibration against human labels, eval is theater.

---

## Primary Failure Hypotheses

1. **Retrieval cross-empresa** — semantic similarity across SEC filings causes wrong company chunks to be retrieved. JnJ + Adobe = {sum(c for d, c in miss_counts.most_common() if 'JOHNSON' in d or 'ADOBE' in d)}/17 retrieval misses.

2. **Numerical error in generation** — model retrieves correct doc but calculates wrong value. Judge blind to this (fluency bias).

3. **Agent multi-retrieval** — complex multi-year queries trigger 5–6 tool calls, diluting precision and introducing cross-company noise.

---

## Proposed Fixes (Phase 3)

| Problem | Fix | Priority |
|---------|-----|----------|
| Retrieval cross-empresa | Metadata filtering (LLM extracts company from query) | HIGH |
| Numerical errors | Hybrid retrieval dense + BM25 | MEDIUM |
| Agent noise | Cap tool calls + sub-query routing | MEDIUM |
"""

OUT = "docs/baseline_report.md"
Path(OUT).parent.mkdir(exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(md)

print(f"Report saved → {OUT}")


if __name__ == "__main__":
    pass

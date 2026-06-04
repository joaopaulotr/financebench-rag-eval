"""Phase 04 report — compares Phase03b vs Phase04 (CRAG + rerank + contextual retrieval)."""
import csv
import json
from pathlib import Path

K = 6
JUDGE_THRESHOLD = 4

P3B_RESULTS = "eval/Phase03/results_phase3b.csv"
P3B_JUDGE   = "eval/Phase03/results_phase3b_with_judge_v2.csv"
P4_RESULTS  = "eval/Phase04/results_phase4.csv"
P4_JUDGE    = "eval/Phase04/results_phase4_with_judge.csv"
LABELS_CSV  = "eval/Phase02/human_labels_30.csv"
OUT_MD      = "docs/phase04_report.md"


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def tier1(results):
    recalls, precisions, mrrs, misses = [], [], [], []
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
    return {
        "n": n,
        "recall": sum(recalls) / n,
        "precision": sum(precisions) / n,
        "mrr": sum(mrrs) / n,
        "misses": misses,
    }


def tier2(results, judge_rows, gt_field="gt_score"):
    judge = {r["financebench_id"]: r for r in judge_rows}
    gt_s, correct = [], {}
    for row in results:
        fid = row["financebench_id"]
        if fid not in judge or not judge[fid].get(gt_field):
            continue
        score = int(judge[fid][gt_field])
        gt_s.append(score)
        correct[fid] = 1 if score >= JUDGE_THRESHOLD else 0
    if not gt_s:
        return None
    return {
        "gt": sum(gt_s) / len(gt_s),
        "correct_count": sum(1 for v in gt_s if v >= JUDGE_THRESHOLD),
        "n": len(gt_s),
        "judge_correct": correct,
    }


def calibrate(judge_correct, labels):
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
    return tpr, tnr


r3b    = load(P3B_RESULTS)
r4     = load(P4_RESULTS)
labels = {r["financebench_id"]: int(r["human_correct"]) for r in load(LABELS_CSV)}

j3b = load(P3B_JUDGE) if Path(P3B_JUDGE).exists() else []
j4  = load(P4_JUDGE)  if Path(P4_JUDGE).exists()  else []

t1_3b = tier1(r3b)
t1_4  = tier1(r4)
t2_3b = tier2(r3b, j3b, gt_field="gt_score_v2") if j3b else None
t2_4  = tier2(r4,  j4,  gt_field="gt_score")    if j4  else None

resolved = sorted(set(t1_3b["misses"]) - set(t1_4["misses"]))
new_miss = sorted(set(t1_4["misses"])  - set(t1_3b["misses"]))

SEP = "-" * 64
print(f"\n{'='*64}")
print("  PHASE 04 -- CRAG + Rerank + Contextual Retrieval v2")
print(f"{'='*64}")

print(f"\n{'Tier 1 -- Retrieval':^64}")
print(SEP)
print(f"  {'Metric':<20} {'Phase03b':>10} {'Phase04':>10} {'Delta':>10}")
print(f"  {'-'*50}")
for label, v3b, v4 in [
    (f"Recall@{K}",    t1_3b["recall"],    t1_4["recall"]),
    (f"Precision@{K}", t1_3b["precision"], t1_4["precision"]),
    ("MRR",            t1_3b["mrr"],       t1_4["mrr"]),
]:
    d = v4 - v3b
    sign = "+" if d >= 0 else ""
    print(f"  {label:<20} {v3b:>10.3f} {v4:>10.3f} {sign}{d:>9.3f}")

print(f"\n  Misses: Phase03b={len(t1_3b['misses'])}  Phase04={len(t1_4['misses'])}")
if resolved:
    print(f"  Resolved: {resolved}")
if new_miss:
    print(f"  New misses: {new_miss}")

if t2_3b and t2_4:
    print(f"\n{'Tier 2 -- Generation (judge v2)':^64}")
    print(SEP)
    print(f"  {'Metric':<25} {'Phase03b':>10} {'Phase04':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    for label, v3b, v4 in [("A|GT Correctness", t2_3b["gt"], t2_4["gt"])]:
        d = v4 - v3b
        sign = "+" if d >= 0 else ""
        print(f"  {label:<25} {v3b:>10.2f} {v4:>10.2f} {sign}{d:>9.2f}")

    print(f"\n  Correct (GT>={JUDGE_THRESHOLD}):")
    print(f"    Phase03b: {t2_3b['correct_count']}/{t2_3b['n']}")
    print(f"    Phase04:  {t2_4['correct_count']}/{t2_4['n']}")

    tpr3b, tnr3b = calibrate(t2_3b["judge_correct"], labels)
    tpr4,  tnr4  = calibrate(t2_4["judge_correct"],  labels)
    print(f"\n  Calibration (30 human labels):")
    print(f"    TPR: Phase03b={tpr3b:.2f}  Phase04={tpr4:.2f}")
    print(f"    TNR: Phase03b={tnr3b:.2f}  Phase04={tnr4:.2f}")

print(f"\n{'='*64}\n")

# Markdown
md = f"""# Phase 04 Report — CRAG + Rerank + Contextual Retrieval

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
| Recall@{K} | {t1_3b['recall']:.3f} | {t1_4['recall']:.3f} | {t1_4['recall']-t1_3b['recall']:+.3f} |
| Precision@{K} | {t1_3b['precision']:.3f} | {t1_4['precision']:.3f} | {t1_4['precision']-t1_3b['precision']:+.3f} |
| MRR | {t1_3b['mrr']:.3f} | {t1_4['mrr']:.3f} | {t1_4['mrr']-t1_3b['mrr']:+.3f} |

Retrieval misses: {len(t1_3b['misses'])} → {len(t1_4['misses'])}

### Resolved misses
{chr(10).join(f'- {d}' for d in resolved) if resolved else '- None'}

### New misses
{chr(10).join(f'- {d}' for d in new_miss) if new_miss else '- None'}

---

## Tier 2 — Generation (judge v2)
"""

if t2_3b and t2_4:
    md += f"""
| Metric | Phase03b | Phase04 | Delta |
|--------|----------|---------|-------|
| A\\|GT Correctness | {t2_3b['gt']:.2f} | {t2_4['gt']:.2f} | {t2_4['gt']-t2_3b['gt']:+.2f} |

Correct (>=4): Phase03b={t2_3b['correct_count']}/{t2_3b['n']}  Phase04={t2_4['correct_count']}/{t2_4['n']}

### Calibration (30 human labels)

| | Phase03b | Phase04 |
|-|----------|---------|
| TPR | {tpr3b:.2f} | {tpr4:.2f} |
| TNR | {tnr3b:.2f} | {tnr4:.2f} |
"""
else:
    md += "\n*Judge results pending.*\n"

Path(OUT_MD).parent.mkdir(exist_ok=True)
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md)

print(f"Report saved -> {OUT_MD}")

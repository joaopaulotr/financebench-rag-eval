import csv
import json
from collections import Counter
from pathlib import Path

K = 6
JUDGE_THRESHOLD = 4

RESULTS_P3  = "eval/Phase03/results_phase3.csv"
JUDGE_P3    = "eval/Phase03/results_phase3_with_judge.csv"
RESULTS_P3B = "eval/Phase03/results_phase3b.csv"
JUDGE_P3B   = "eval/Phase03/results_phase3b_with_judge.csv"
LABELS_CSV  = "eval/Phase02/human_labels_30.csv"
OUT_MD      = "docs/phase03b_report.md"


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


def tier2(results, judge_rows):
    judge = {r["financebench_id"]: r for r in judge_rows}
    cq, ac, aq, gt_s = [], [], [], []
    correct = {}
    for row in results:
        fid = row["financebench_id"]
        if fid not in judge:
            continue
        j = judge[fid]
        cq.append(int(j["cq_score"]))
        ac.append(int(j["ac_score"]))
        aq.append(int(j["aq_score"]))
        gt_s.append(int(j["gt_score"]))
        correct[fid] = 1 if int(j["gt_score"]) >= JUDGE_THRESHOLD else 0
    if not gt_s:
        return None
    return {
        "cq": sum(cq) / len(cq),
        "ac": sum(ac) / len(ac),
        "aq": sum(aq) / len(aq),
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


r3  = load(RESULTS_P3)
r3b = load(RESULTS_P3B)
labels = {r["financebench_id"]: int(r["human_correct"]) for r in load(LABELS_CSV)}

j3_rows  = load(JUDGE_P3)  if Path(JUDGE_P3).exists()  else []
j3b_rows = load(JUDGE_P3B) if Path(JUDGE_P3B).exists() else []

t1_p3  = tier1(r3)
t1_p3b = tier1(r3b)
t2_p3  = tier2(r3,  j3_rows)  if j3_rows  else None
t2_p3b = tier2(r3b, j3b_rows) if j3b_rows else None

miss3  = Counter(t1_p3["misses"])
miss3b = Counter(t1_p3b["misses"])
resolved = sorted(set(miss3) - set(miss3b))
new_miss = sorted(set(miss3b) - set(miss3))

SEP = "-" * 64

print(f"\n{'='*64}")
print("  PHASE 03b -- Comparative Report (full corpus)")
print(f"{'='*64}")

print(f"\n{'Tier 1 -- Retrieval':^64}")
print(SEP)
print(f"  {'Metric':<20} {'Phase03':>10} {'Phase03b':>10} {'Delta':>10}")
print(f"  {'-'*50}")
for label, v3, v3b in [
    (f"Recall@{K}",    t1_p3["recall"],    t1_p3b["recall"]),
    (f"Precision@{K}", t1_p3["precision"], t1_p3b["precision"]),
    ("MRR",            t1_p3["mrr"],       t1_p3b["mrr"]),
]:
    d = v3b - v3
    sign = "+" if d >= 0 else ""
    print(f"  {label:<20} {v3:>10.3f} {v3b:>10.3f} {sign}{d:>9.3f}")

print(f"\n  Misses: Phase03={len(t1_p3['misses'])}  Phase03b={len(t1_p3b['misses'])}")
if resolved:
    print(f"  Resolved: {resolved}")
if new_miss:
    print(f"  New misses: {new_miss}")

if t2_p3 and t2_p3b:
    print(f"\n{'Tier 2 -- Generation (LLM-as-Judge 1-5)':^64}")
    print(SEP)
    print(f"  {'Metric':<25} {'Phase03':>10} {'Phase03b':>10} {'Delta':>10}")
    print(f"  {'-'*55}")
    for label, v3, v3b in [
        ("C|Q Context Relevance", t2_p3["cq"], t2_p3b["cq"]),
        ("A|C Faithfulness",      t2_p3["ac"], t2_p3b["ac"]),
        ("A|Q Answer Relevance",  t2_p3["aq"], t2_p3b["aq"]),
        ("A|GT Correctness",      t2_p3["gt"], t2_p3b["gt"]),
    ]:
        d = v3b - v3
        sign = "+" if d >= 0 else ""
        print(f"  {label:<25} {v3:>10.2f} {v3b:>10.2f} {sign}{d:>9.2f}")

    print(f"\n  Judge 'correct' (GT>={JUDGE_THRESHOLD}):")
    print(f"    Phase03:  {t2_p3['correct_count']}/{t2_p3['n']}")
    print(f"    Phase03b: {t2_p3b['correct_count']}/{t2_p3b['n']}")

    tpr3,  tnr3  = calibrate(t2_p3["judge_correct"],  labels)
    tpr3b, tnr3b = calibrate(t2_p3b["judge_correct"], labels)
    print(f"\n{'Judge Calibration (30 human labels)':^64}")
    print(SEP)
    print(f"  TPR: Phase03={tpr3:.2f}  Phase03b={tpr3b:.2f}")
    print(f"  TNR: Phase03={tnr3:.2f}  Phase03b={tnr3b:.2f}")

print(f"\n{'='*64}\n")

# Markdown
md = f"""# Phase 03b Report — Full Corpus (+5 Missing PDFs)

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
| Recall@{K} | **{t1_p3['recall']:.3f}** | **{t1_p3b['recall']:.3f}** | {t1_p3b['recall']-t1_p3['recall']:+.3f} |
| Precision@{K} | **{t1_p3['precision']:.3f}** | **{t1_p3b['precision']:.3f}** | {t1_p3b['precision']-t1_p3['precision']:+.3f} |
| MRR | **{t1_p3['mrr']:.3f}** | **{t1_p3b['mrr']:.3f}** | {t1_p3b['mrr']-t1_p3['mrr']:+.3f} |

Retrieval misses: {len(t1_p3['misses'])} → {len(t1_p3b['misses'])}

### Resolved misses
{chr(10).join(f'- {d}' for d in resolved) if resolved else 'None'}

### New misses
{chr(10).join(f'- {d}' for d in new_miss) if new_miss else 'None'}

---

## Tier 2 -- Generation Comparison
"""

if t2_p3 and t2_p3b:
    md += f"""
| Metric | Phase03 | Phase03b | Delta |
|--------|---------|----------|-------|
| C\\|Q Context Relevance | **{t2_p3['cq']:.2f}** | **{t2_p3b['cq']:.2f}** | {t2_p3b['cq']-t2_p3['cq']:+.2f} |
| A\\|C Faithfulness | **{t2_p3['ac']:.2f}** | **{t2_p3b['ac']:.2f}** | {t2_p3b['ac']-t2_p3['ac']:+.2f} |
| A\\|Q Answer Relevance | **{t2_p3['aq']:.2f}** | **{t2_p3b['aq']:.2f}** | {t2_p3b['aq']-t2_p3['aq']:+.2f} |
| A\\|GT Correctness | **{t2_p3['gt']:.2f}** | **{t2_p3b['gt']:.2f}** | {t2_p3b['gt']-t2_p3['gt']:+.2f} |

Judge classified Phase03={t2_p3['correct_count']}/{t2_p3['n']} vs Phase03b={t2_p3b['correct_count']}/{t2_p3b['n']} answers as correct (A|GT >= {JUDGE_THRESHOLD}).

### Judge Calibration (30 human labels)

| | Phase03 | Phase03b |
|-|---------|----------|
| TPR | {tpr3:.2f} | {tpr3b:.2f} |
| TNR | {tnr3:.2f} | {tnr3b:.2f} |
"""
else:
    md += "\n*Judge results pending.*\n"

md += f"""
---

## Key Findings

1. Ingesting the 5 missing PDFs resolved {len(resolved)} retrieval miss(es): `{", ".join(resolved) if resolved else "none yet"}`

2. Metadata filtering (company name extraction) can now actually work for JnJ queries — before, chunks didn't exist in Qdrant at all.

3. Remaining JnJ misses after fix are likely cross-document within same company (2022_10K vs 2023_8K, etc.) — company-level filtering doesn't resolve year/doc-type confusion.
"""

Path(OUT_MD).parent.mkdir(exist_ok=True)
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md)

print(f"Report saved -> {OUT_MD}")

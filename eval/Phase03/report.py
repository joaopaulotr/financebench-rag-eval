import csv
import json
from collections import Counter
from pathlib import Path

K = 6
JUDGE_THRESHOLD = 4

RESULTS_P2   = "eval/Phase02/results_100.csv"
JUDGE_P2     = "eval/Phase02/results_with_judge.csv"
RESULTS_P3   = "eval/Phase03/results_phase3.csv"
JUDGE_P3     = "eval/Phase03/results_phase3_with_judge.csv"
LABELS_CSV   = "eval/Phase02/human_labels_30.csv"
OUT_MD       = "docs/phase03_report.md"


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
    cq, ac, aq, gt_s, correct = [], [], [], [], {}
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
        if human == 1 and jc == 1:    tp += 1
        elif human == 0 and jc == 0:  tn += 1
        elif human == 1 and jc == 0:  fn += 1
        else:                          fp += 1
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return tpr, tnr


# ── Load ──────────────────────────────────────────────────────────────────────
r2 = load(RESULTS_P2)
r3 = load(RESULTS_P3)
labels = {r["financebench_id"]: int(r["human_correct"]) for r in load(LABELS_CSV)}

j2_rows = load(JUDGE_P2)
j3_rows = load(JUDGE_P3) if Path(JUDGE_P3).exists() else []

t1_p2 = tier1(r2)
t1_p3 = tier1(r3)
t2_p2 = tier2(r2, j2_rows)
t2_p3 = tier2(r3, j3_rows) if j3_rows else None

miss2 = Counter(t1_p2["misses"])
miss3 = Counter(t1_p3["misses"])
resolved = sorted(set(miss2) - set(miss3))
new_miss = sorted(set(miss3) - set(miss2))

SEP = "-" * 64

# ── Print ─────────────────────────────────────────────────────────────────────
print(f"\n{'='*64}")
print("  PHASE 03 -- Comparative Report")
print(f"{'='*64}")

print(f"\n{'Tier 1 -- Retrieval':^64}")
print(SEP)
print(f"  {'Metric':<20} {'Phase02':>10} {'Phase03':>10} {'Delta':>10}")
print(f"  {'-'*50}")
for label, v2, v3 in [
    (f"Recall@{K}",    t1_p2["recall"],    t1_p3["recall"]),
    (f"Precision@{K}", t1_p2["precision"], t1_p3["precision"]),
    ("MRR",            t1_p2["mrr"],       t1_p3["mrr"]),
]:
    d = v3 - v2
    sign = "+" if d >= 0 else ""
    print(f"  {label:<20} {v2:>10.3f} {v3:>10.3f} {sign}{d:>9.3f}")

print(f"\n  Misses: Phase02={len(t1_p2['misses'])}  Phase03={len(t1_p3['misses'])}")
if resolved:
    print(f"  Resolved: {resolved}")
if new_miss:
    print(f"  New misses: {new_miss}")

if t2_p2:
    print(f"\n{'Tier 2 -- Generation (LLM-as-Judge 1-5)':^64}")
    print(SEP)
    print(f"  {'Metric':<25} {'Phase02':>10}", end="")
    print(f" {'Phase03':>10}" if t2_p3 else "")
    print(f"  {'-'*50}")
    rows_t2 = [
        ("C|Q Context Relevance", t2_p2["cq"], t2_p3["cq"] if t2_p3 else None),
        ("A|C Faithfulness",      t2_p2["ac"], t2_p3["ac"] if t2_p3 else None),
        ("A|Q Answer Relevance",  t2_p2["aq"], t2_p3["aq"] if t2_p3 else None),
        ("A|GT Correctness",      t2_p2["gt"], t2_p3["gt"] if t2_p3 else None),
    ]
    for label, v2, v3 in rows_t2:
        if v3 is not None:
            d = v3 - v2
            sign = "+" if d >= 0 else ""
            print(f"  {label:<25} {v2:>10.2f} {v3:>10.2f} {sign}{d:>9.2f}")
        else:
            print(f"  {label:<25} {v2:>10.2f}   (judge pending)")

    print(f"\n  Judge 'correct' (GT>={JUDGE_THRESHOLD}):")
    print(f"    Phase02: {t2_p2['correct_count']}/{t2_p2['n']}")
    if t2_p3:
        print(f"    Phase03: {t2_p3['correct_count']}/{t2_p3['n']}")

    if t2_p3:
        tpr2, tnr2 = calibrate(t2_p2["judge_correct"], labels)
        tpr3, tnr3 = calibrate(t2_p3["judge_correct"], labels)
        print(f"\n{'Judge Calibration (30 human labels)':^64}")
        print(SEP)
        print(f"  TPR: Phase02={tpr2:.2f}  Phase03={tpr3:.2f}")
        print(f"  TNR: Phase02={tnr2:.2f}  Phase03={tnr3:.2f}")

print(f"\n{'='*64}\n")

# ── Markdown ──────────────────────────────────────────────────────────────────
t2_p3_gt = t2_p3["gt"] if t2_p3 else "N/A"
t2_p3_correct = f"{t2_p3['correct_count']}/{t2_p3['n']}" if t2_p3 else "N/A"

md = f"""# Phase 03 Report — Iteracao com Fixes

**Date:** 2026-05-21
**Pipeline:** GPT-4o-mini + text-embedding-3-small + Qdrant hybrid (dense + BM25) + metadata filtering
**Eval set:** 100 queries from FinanceBench

---

## Tier 1 -- Retrieval Comparison

| Metric | Phase02 (baseline) | Phase03 (fixes) | Delta |
|--------|-------------------|-----------------|-------|
| Recall@{K} | **{t1_p2['recall']:.3f}** | **{t1_p3['recall']:.3f}** | {t1_p3['recall']-t1_p2['recall']:+.3f} |
| Precision@{K} | **{t1_p2['precision']:.3f}** | **{t1_p3['precision']:.3f}** | {t1_p3['precision']-t1_p2['precision']:+.3f} |
| MRR | **{t1_p2['mrr']:.3f}** | **{t1_p3['mrr']:.3f}** | {t1_p3['mrr']-t1_p2['mrr']:+.3f} |

Retrieval misses: {len(t1_p2['misses'])} → {len(t1_p3['misses'])}

### Resolved misses
{chr(10).join(f'- {d}' for d in resolved) if resolved else 'None'}

### New misses
{chr(10).join(f'- {d}' for d in new_miss) if new_miss else 'None'}

---

## Tier 2 -- Generation Comparison
"""

if t2_p2 and t2_p3:
    md += f"""
| Metric | Phase02 | Phase03 | Delta |
|--------|---------|---------|-------|
| C\\|Q Context Relevance | **{t2_p2['cq']:.2f}** | **{t2_p3['cq']:.2f}** | {t2_p3['cq']-t2_p2['cq']:+.2f} |
| A\\|C Faithfulness | **{t2_p2['ac']:.2f}** | **{t2_p3['ac']:.2f}** | {t2_p3['ac']-t2_p2['ac']:+.2f} |
| A\\|Q Answer Relevance | **{t2_p2['aq']:.2f}** | **{t2_p3['aq']:.2f}** | {t2_p3['aq']-t2_p2['aq']:+.2f} |
| A\\|GT Correctness | **{t2_p2['gt']:.2f}** | **{t2_p3['gt']:.2f}** | {t2_p3['gt']-t2_p2['gt']:+.2f} |

Judge classified Phase02={t2_p2['correct_count']}/{t2_p2['n']} vs Phase03={t2_p3['correct_count']}/{t2_p3['n']} answers as correct (A|GT >= {JUDGE_THRESHOLD}).
"""
else:
    md += "\n*Judge results pending — run eval/Phase03/runJudge.py*\n"

md += f"""
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

1. Metadata filtering partially effective — resolved {len(resolved)} retrieval miss(es). Adobe docs remain unretrievable via semantic search regardless of filtering, pointing to an embedding representation issue.

2. MRR dropped ({t1_p2['mrr']:.3f} to {t1_p3['mrr']:.3f}) — BM25 hybrid changes ranking, sometimes pushing the correct chunk lower. Dense-only had better rank precision for this dataset.

3. wrong_doc diagnosis was incomplete — JnJ misses are cross-document within the same company (2022_10K vs 2023_8K), not cross-company. Filtering by company name does not resolve these.
"""

Path(OUT_MD).parent.mkdir(exist_ok=True)
with open(OUT_MD, "w", encoding="utf-8") as f:
    f.write(md)

print(f"Report saved -> {OUT_MD}")

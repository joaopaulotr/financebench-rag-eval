"""Compare v1 vs v2 judge scores: TNR/TPR, score changes, false positive delta."""
import csv
from pathlib import Path

LABELS    = "eval/Phase03/label_phase3b.csv"
JUDGE_V1  = "eval/Phase03/results_phase3b_with_judge.csv"
JUDGE_V2  = "eval/Phase03/results_phase3b_with_judge_v2.csv"
OUT       = "docs/judge_v2_calibration.md"
THRESHOLD = 4


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


labels = {r["financebench_id"]: int(r["human_correct"])
          for r in load(LABELS) if r["human_correct"] != ""}

v1_rows = {r["financebench_id"]: r for r in load(JUDGE_V1)}
v2_rows = {r["financebench_id"]: r for r in load(JUDGE_V2)}


def calibrate(score_map, labels, threshold=THRESHOLD):
    tp = tn = fp = fn = 0
    for fid, human in labels.items():
        if fid not in score_map:
            continue
        jc = 1 if int(score_map[fid]) >= threshold else 0
        if   human == 1 and jc == 1: tp += 1
        elif human == 0 and jc == 0: tn += 1
        elif human == 1 and jc == 0: fn += 1
        else:                         fp += 1
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return tpr, tnr, tp, tn, fp, fn


v1_scores = {fid: r["gt_score"]    for fid, r in v1_rows.items()}
v2_scores = {fid: r["gt_score_v2"] for fid, r in v2_rows.items()}

tpr1, tnr1, tp1, tn1, fp1, fn1 = calibrate(v1_scores, labels)
tpr2, tnr2, tp2, tn2, fp2, fn2 = calibrate(v2_scores, labels)

nom_v1 = sum(1 for r in v1_rows.values() if int(r["gt_score"])    >= THRESHOLD)
nom_v2 = sum(1 for r in v2_rows.values() if int(r["gt_score_v2"]) >= THRESHOLD)

human_correct = sum(labels.values())
est_v2 = round(sum(labels.values()) / len(labels) * 100)

# Score changes
changes = []
for fid, v1r in v1_rows.items():
    v2r = v2_rows.get(fid)
    if not v2r:
        continue
    s1 = int(v1r["gt_score"])
    s2 = int(v2r["gt_score_v2"])
    if s1 != s2:
        changes.append({
            "fid": fid,
            "v1": s1, "v2": s2, "delta": s2 - s1,
            "question": v1r["question"],
            "expected": v1r.get("expected_answer", ""),
            "model": v1r.get("model_answer", ""),
            "reason_v2": v2r.get("gt_reason_v2", ""),
        })

changes.sort(key=lambda x: x["delta"])

SEP = "-" * 64
print(f"\n{'='*64}")
print("  Judge v2 Calibration Report")
print(f"{'='*64}")
print(f"\n  Nominal correct (n=100, GT>={THRESHOLD}): v1={nom_v1}  v2={nom_v2}  delta={nom_v2-nom_v1:+d}")
print(f"\n  Calibration on 30 human labels:")
print(f"  {'':4} {'TPR':>6} {'TNR':>6} {'tp':>4} {'tn':>4} {'fp':>4} {'fn':>4}")
print(f"  {'v1':4} {tpr1:>6.2f} {tnr1:>6.2f} {tp1:>4} {tn1:>4} {fp1:>4} {fn1:>4}")
print(f"  {'v2':4} {tpr2:>6.2f} {tnr2:>6.2f} {tp2:>4} {tn2:>4} {fp2:>4} {fn2:>4}")
print(f"\n  Score changes: {len(changes)}/100")
for c in changes:
    print(f"    {c['fid']}  {c['v1']} -> {c['v2']} ({c['delta']:+d})  {c['question'][:50]}")
print(f"\n{'='*64}\n")

# Markdown
md = f"""# Judge v2 Calibration — Phase03b

**Change:** Replaced A|GT prompt with strict numerical extraction + tolerance rules.
**Rule:** Numbers differing >15% from ground truth → score ≤ 2, regardless of reasoning quality.

---

## Summary

| Metric | Judge v1 | Judge v2 | Delta |
|--------|---------|---------|-------|
| Nominal correct (n=100) | {nom_v1} | {nom_v2} | {nom_v2-nom_v1:+d} |
| TPR (catches correct answers) | {tpr1:.2f} | {tpr2:.2f} | {tpr2-tpr1:+.2f} |
| TNR (catches wrong answers) | {tnr1:.2f} | {tnr2:.2f} | {tnr2-tnr1:+.2f} |
| False positives (30 sample) | {fp1} | {fp2} | {fp2-fp1:+d} |
| Human-corrected estimate | ~{round(human_correct/len(labels)*100)}/100 | ~{round(human_correct/len(labels)*100)}/100 | — |

**Key insight:** v2 judge TNR improved from {tnr1:.2f} → {tnr2:.2f} — better at catching numerically wrong answers that sound convincing.

---

## Score Changes ({len(changes)} queries)

| ID | v1 | v2 | Delta | Question |
|----|----|----|-------|---------|
"""

for c in changes:
    q = c["question"][:60].replace("|", "\\|")
    md += f"| `{c['fid']}` | {c['v1']} | {c['v2']} | {c['delta']:+d} | {q} |\n"

md += "\n---\n\n## False Positive Analysis\n\n"
md += f"### v1 False Positives ({fp1} cases)\n"
for fid, human in labels.items():
    if human == 1:
        continue
    s1 = int(v1_scores.get(fid, 0))
    if s1 >= THRESHOLD:
        r = v1_rows[fid]
        md += f"\n**`{fid}`** (score {s1}/5)\n"
        md += f"- Q: {r['question'][:80]}\n"
        md += f"- Expected: `{r.get('expected_answer','')[:80]}`\n"
        md += f"- Model: `{r.get('model_answer','')[:120]}`\n"

md += f"\n### v2 False Positives ({fp2} cases)\n"
fps_v2 = []
for fid, human in labels.items():
    if human == 1:
        continue
    s2 = int(v2_scores.get(fid, 0))
    if s2 >= THRESHOLD:
        fps_v2.append(fid)
        r = v2_rows[fid]
        md += f"\n**`{fid}`** (score {s2}/5)\n"
        md += f"- Q: {r['question'][:80]}\n"
        md += f"- Expected: `{r.get('expected_answer','')[:80]}`\n"

if not fps_v2:
    md += "\nNone — all former false positives correctly downgraded.\n"

md += "\n---\n\n## Conclusion\n\n"
md += f"v2 judge is stricter on numerical accuracy. "
md += f"Nominal 'correct' count dropped from {nom_v1} → {nom_v2}/100. "
md += f"TNR improved {tnr1:.2f} → {tnr2:.2f}, meaning fewer wrong-but-fluent answers pass as correct. "
md += f"Human-calibrated real accuracy remains ~{round(human_correct/len(labels)*100)}/100.\n"

Path(OUT).parent.mkdir(exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write(md)
print(f"Report saved -> {OUT}")

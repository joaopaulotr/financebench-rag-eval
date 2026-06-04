"""Phase 04 judge — reuses judge_v2 from Phase03."""

import csv
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Phase03"))

from judge_v2 import judge_answer_correctness_v2

CSV_IN = "eval/Phase04/results_phase4.csv"
CSV_OUT = "eval/Phase04/results_phase4_with_judge.csv"
CHECKPOINT_EVERY = 10

with open(CSV_IN, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

fieldnames = list(rows[0].keys())
for field in ("cq_score", "ac_score", "aq_score", "gt_score", "gt_reason"):
    if field not in fieldnames:
        fieldnames.append(field)

results = []
t_start = time.time()

for i, row in enumerate(rows):
    print(f"[{i+1:03d}/{len(rows)}] {row['question'][:70]}...")
    t0 = time.time()

    gt = judge_answer_correctness_v2(
        row["question"],
        row.get("expected_answer", ""),
        row.get("model_answer", ""),
    )

    results.append(
        {
            **row,
            "cq_score": "",
            "ac_score": "",
            "aq_score": "",
            "gt_score": gt["score"],
            "gt_reason": gt["reasoning"],
        }
    )

    print(f"  gt={gt['score']}  ({time.time()-t0:.1f}s)")

    if (i + 1) % CHECKPOINT_EVERY == 0:
        with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"  [checkpoint {i+1}/{len(rows)}]")

with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

n = len(results)
avg_gt = sum(int(r["gt_score"]) for r in results if r["gt_score"]) / n
correct = sum(1 for r in results if r["gt_score"] and int(r["gt_score"]) >= 4)

print(f"\nDone. {n} rows -> {CSV_OUT} ({time.time()-t_start:.0f}s)")
print(f"Avg GT: {avg_gt:.2f}  Correct (>=4): {correct}/{n}")

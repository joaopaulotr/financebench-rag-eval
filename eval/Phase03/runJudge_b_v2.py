"""Re-judge Phase03b results using the v2 stricter A|GT prompt.
Keeps C|Q, A|C, A|Q from original run — only replaces gt_score/gt_reason."""

import csv
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

from judge_v2 import judge_answer_correctness_v2

CSV_IN = "eval/Phase03/results_phase3b_with_judge.csv"
CSV_OUT = "eval/Phase03/results_phase3b_with_judge_v2.csv"
CHECKPOINT_EVERY = 10

with open(CSV_IN, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

fieldnames = list(rows[0].keys())
for field in ("gt_score_v2", "gt_reason_v2"):
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
            "gt_score_v2": gt["score"],
            "gt_reason_v2": gt["reasoning"],
        }
    )

    old = row.get("gt_score", "?")
    new = gt["score"]
    flag = " <-- CHANGED" if str(old) != str(new) else ""
    print(f"  gt_v1={old}  gt_v2={new}{flag}  ({time.time()-t0:.1f}s)")

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

total = time.time() - t_start
n = len(results)
avg_v1 = sum(int(r["gt_score"]) for r in results) / n
avg_v2 = sum(int(r["gt_score_v2"]) for r in results) / n
changed = sum(1 for r in results if str(r["gt_score"]) != str(r["gt_score_v2"]))

print(f"\nDone. {n} rows -> {CSV_OUT} ({total:.0f}s)")
print(f"Avg GT v1: {avg_v1:.2f}  v2: {avg_v2:.2f}  delta: {avg_v2-avg_v1:+.2f}")
print(f"Changed scores: {changed}/{n}")

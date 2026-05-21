import csv
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "Phase02"))

from judge import (
    judge_answer_correctness,
    judge_answer_faithfulness,
    judge_answer_relevance,
    judge_context_relevance,
)

CSV_IN = "eval/Phase03/results_phase3.csv"
CSV_OUT = "eval/Phase03/results_phase3_with_judge.csv"
CHECKPOINT_EVERY = 10

JUDGE_FIELDS = [
    "cq_score", "cq_reason",
    "ac_score", "ac_reason",
    "aq_score", "aq_reason",
    "gt_score", "gt_reason",
]

with open(CSV_IN, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

fieldnames = list(rows[0].keys())
for field in JUDGE_FIELDS:
    if field not in fieldnames:
        fieldnames.append(field)

results = []
t_start = time.time()

for i, row in enumerate(rows):
    print(f"[{i+1:03d}/{len(rows)}] {row['question'][:70]}...")
    t0 = time.time()

    context = row.get("retrieved_context", "")

    cq = judge_context_relevance(row["question"], context)
    ac = judge_answer_faithfulness(context, row["model_answer"])
    aq = judge_answer_relevance(row["question"], row["model_answer"])
    gt = judge_answer_correctness(row["question"], row["expected_answer"], row["model_answer"])

    results.append({
        **row,
        "cq_score":  cq["score"],
        "cq_reason": cq["reasoning"],
        "ac_score":  ac["score"],
        "ac_reason": ac["reasoning"],
        "aq_score":  aq["score"],
        "aq_reason": aq["reasoning"],
        "gt_score":  gt["score"],
        "gt_reason": gt["reasoning"],
    })

    print(f"  C|Q={cq['score']}  A|C={ac['score']}  A|Q={aq['score']}  GT={gt['score']}  ({time.time()-t0:.1f}s)")

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
avg_cq = sum(int(r["cq_score"]) for r in results) / n
avg_ac = sum(int(r["ac_score"]) for r in results) / n
avg_aq = sum(int(r["aq_score"]) for r in results) / n

print(f"\nDone. {n} rows -> {CSV_OUT} ({total:.0f}s)")
print(f"Avg C|Q: {avg_cq:.2f}  A|C: {avg_ac:.2f}  A|Q: {avg_aq:.2f}  (scale 1-5)")

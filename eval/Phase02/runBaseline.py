import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.core import run_llm

CSV_IN = "eval/Phase02/baseline_100.csv"
CSV_OUT = "eval/Phase02/results_100.csv"
CHECKPOINT_EVERY = 10

with open(CSV_IN, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

fieldnames = list(rows[0].keys())
if "retrieved_sources" not in fieldnames:
    fieldnames.append("retrieved_sources")

results = []
t_start = time.time()

for i, row in enumerate(rows):
    print(f"[{i+1:03d}/{len(rows)}] {row['question'][:70]}...")
    t0 = time.time()

    result = run_llm(row["question"])
    answer = result["answer"]
    sources = [Path(s).stem for s in result["context_docs"]]

    results.append(
        {
            **row,
            "model_answer": answer,
            "retrieved_sources": json.dumps(sources),
        }
    )

    print(f"  sources: {sources}")
    print(f"  answer:  {answer[:100]}")
    print(f"  {time.time() - t0:.1f}s")

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

print(
    f"\nDone. {len(results)} results -> {CSV_OUT} ({time.time() - t_start:.0f}s total)"
)

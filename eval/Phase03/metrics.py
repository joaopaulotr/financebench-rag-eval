import csv
import json
from collections import Counter

RESULTS_CSV_P2 = "eval/Phase02/results_100.csv"
RESULTS_CSV_P3 = "eval/Phase03/results_phase3.csv"
K = 6


def compute_tier1(results_csv: str, label: str, k: int = K) -> dict:
    with open(results_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows = [r for r in rows if r.get("retrieved_sources")]

    recalls, precisions, mrrs = [], [], []

    for row in rows:
        gt = row["doc_name"]
        sources: list[str] = json.loads(row["retrieved_sources"])

        recalls.append(1.0 if gt in sources else 0.0)
        precisions.append(
            sum(s == gt for s in sources) / len(sources) if sources else 0.0
        )

        rr = 0.0
        for rank, s in enumerate(sources, 1):
            if s == gt:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

    n = len(rows)
    metrics = {
        "label": label,
        "n": n,
        f"recall@{k}": round(sum(recalls) / n, 4),
        f"precision@{k}": round(sum(precisions) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
        "misses": [r["doc_name"] for r, hit in zip(rows, recalls) if hit == 0.0],
    }
    return metrics


if __name__ == "__main__":
    p2 = compute_tier1(RESULTS_CSV_P2, "Phase02 — dense only, no filter")
    p3 = compute_tier1(RESULTS_CSV_P3, "Phase03 — hybrid + metadata filter")

    SEP = "-" * 64
    print(f"\n{'='*64}")
    print("  PHASE 03 -- Tier 1 Comparison")
    print(f"{'='*64}")

    print(f"\n{'Metric':<20} {'Phase02':>12} {'Phase03':>12} {'Delta':>10}")
    print(SEP)
    for key in (f"recall@{K}", f"precision@{K}", "mrr"):
        v2 = p2[key]
        v3 = p3[key]
        delta = v3 - v2
        sign = "+" if delta >= 0 else ""
        print(f"  {key:<18} {v2:>12.3f} {v3:>12.3f} {sign}{delta:>9.3f}")

    print(f"\n  Queries evaluated: Phase02={p2['n']}  Phase03={p3['n']}")

    miss2 = Counter(p2["misses"])
    miss3 = Counter(p3["misses"])
    print(
        f"\n  Retrieval misses: Phase02={len(p2['misses'])}  Phase03={len(p3['misses'])}"
    )

    resolved = set(miss2) - set(miss3)
    new_misses = set(miss3) - set(miss2)

    if resolved:
        print(f"\n  Resolved by fix ({len(resolved)} docs):")
        for doc in sorted(resolved):
            print(f"    + {doc}")

    if new_misses:
        print(f"\n  New misses ({len(new_misses)} docs):")
        for doc in sorted(new_misses):
            print(f"    - {doc}")

    print(f"\n{'='*64}\n")

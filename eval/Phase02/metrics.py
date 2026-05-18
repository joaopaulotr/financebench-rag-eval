import csv
import json

RESULTS_CSV = "eval/Phase02/results_100.csv"
K = 6


def compute_tier1(results_csv: str = RESULTS_CSV, k: int = K) -> dict:
    with open(results_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Only score rows that were actually run
    rows = [r for r in rows if r.get("retrieved_sources")]

    recalls, precisions, mrrs = [], [], []

    for row in rows:
        gt = row["doc_name"]
        sources: list[str] = json.loads(row["retrieved_sources"])

        # Recall@k: was the correct doc retrieved at all?
        recalls.append(1.0 if gt in sources else 0.0)

        # Precision@k: fraction of retrieved chunks from correct doc
        precisions.append(sum(s == gt for s in sources) / len(sources) if sources else 0.0)

        # MRR: reciprocal rank of first correct chunk
        rr = 0.0
        for rank, s in enumerate(sources, 1):
            if s == gt:
                rr = 1.0 / rank
                break
        mrrs.append(rr)

    n = len(rows)
    metrics = {
        "n": n,
        f"recall@{k}": round(sum(recalls) / n, 4),
        f"precision@{k}": round(sum(precisions) / n, 4),
        "mrr": round(sum(mrrs) / n, 4),
    }

    print(f"\nTier 1 Retrieval Metrics — {n} queries, k={k}")
    print(f"  Recall@{k}:    {metrics[f'recall@{k}']:.3f}")
    print(f"  Precision@{k}: {metrics[f'precision@{k}']:.3f}")
    print(f"  MRR:           {metrics['mrr']:.3f}")

    # Per-doc breakdown: how often does each doc_name fail?
    failures = [r["doc_name"] for r, hit in zip(rows, recalls) if hit == 0.0]
    if failures:
        from collections import Counter
        print(f"\n  Retrieval misses by doc ({len(failures)} total):")
        for doc, count in Counter(failures).most_common(10):
            print(f"    {doc}: {count}x")

    return metrics


if __name__ == "__main__":
    compute_tier1()

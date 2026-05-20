import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

RESULTS_CSV = "eval/Phase02/results_100.csv"
JUDGE_CSV = "eval/Phase02/results_with_judge.csv"
OUT_CSV = "eval/Phase03/error_analysis.csv"
GT_THRESHOLD = 4  # gt_score >= 4 = correct


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def extract_company(doc_name: str) -> str:
    if doc_name.startswith("JOHNSON_JOHNSON"):
        return "JOHNSON_JOHNSON"
    if doc_name.startswith("CROWDSTRIKE"):
        return "CROWDSTRIKE"
    return doc_name.split("_")[0]


def has_number(text: str) -> bool:
    return bool(re.search(r"\d", text))


def classify(row: dict, j: dict) -> str:
    gt = row["doc_name"]
    sources = json.loads(row.get("retrieved_sources", "[]"))
    retrieval_ok = gt in sources

    gt_score = int(j["gt_score"]) if j.get("gt_score") else 0
    ac_score = int(j["ac_score"]) if j.get("ac_score") else 0

    if not retrieval_ok:
        gt_company = extract_company(gt)
        cross = any(extract_company(s) != gt_company for s in sources)
        return "wrong_doc" if cross else "retrieval_miss"

    if gt_score >= GT_THRESHOLD:
        return "correct"

    if ac_score <= 2:
        return "hallucination"

    expected = row.get("expected_answer", "")
    if has_number(expected) and gt_score <= 2:
        return "numerical_error"

    if gt_score == 3:
        return "incomplete"

    return "generation_error"


def load_judge(path: str) -> dict:
    return {r["financebench_id"]: r for r in load(path)}


results = load(RESULTS_CSV)
judge = load_judge(JUDGE_CSV)

classified = []
for row in results:
    fid = row["financebench_id"]
    j = judge.get(fid, {})
    mode = classify(row, j)
    sources = json.loads(row.get("retrieved_sources", "[]"))
    classified.append({
        "financebench_id": fid,
        "failure_mode": mode,
        "retrieval_ok": row["doc_name"] in sources,
        "gt_score": j.get("gt_score", ""),
        "ac_score": j.get("ac_score", ""),
        "aq_score": j.get("aq_score", ""),
        "cq_score": j.get("cq_score", ""),
        "doc_name": row["doc_name"],
        "question": row["question"],
        "expected_answer": row.get("expected_answer", ""),
        "model_answer": row.get("model_answer", ""),
        "gt_reason": j.get("gt_reason", ""),
    })

mode_counts = Counter(r["failure_mode"] for r in classified)
total = len(classified)
failures = [(m, c) for m, c in mode_counts.items() if m != "correct"]
failures.sort(key=lambda x: -x[1])

SEP = "-" * 64

print(f"\n{'='*64}")
print("  PHASE 03 -- Error Analysis")
print(f"{'='*64}")

print(f"\n{'Failure Mode Distribution':^64}")
print(SEP)
for mode, count in sorted(mode_counts.items(), key=lambda x: -x[1]):
    pct = count / total * 100
    bar = "#" * int(pct / 2)
    tag = "  " if mode == "correct" else ""
    print(f"  {tag}{mode:<22} {count:3d}  ({pct:4.1f}%)  {bar}")

print(f"\n{'Top Failure Modes -- Fix Priority':^64}")
print(SEP)
fix_map = {
    "wrong_doc":        "Metadata filtering (extract company from query -> Qdrant filter)",
    "retrieval_miss":   "Hybrid retrieval dense+BM25 or better chunking",
    "numerical_error":  "Hybrid retrieval (BM25 matches exact numbers)",
    "hallucination":    "Faithfulness prompt + temperature=0 check",
    "incomplete":       "Multi-step retrieval / query decomposition",
    "generation_error": "Prompt engineering",
}
for i, (mode, count) in enumerate(failures[:5], 1):
    priority = "HIGH  " if i <= 2 else "MEDIUM"
    fix = fix_map.get(mode, "TBD")
    print(f"  {i}. [{priority}] {mode} ({count}x)")
    print(f"          -> {fix}")

print(f"\n{'2 Chosen Fixes':^64}")
print(SEP)
top2 = [m for m, _ in failures[:2]]
for i, mode in enumerate(top2, 1):
    print(f"  Fix {i}: {mode} — {fix_map.get(mode, 'TBD')}")

print(f"\n{'Sample Failures (2 per mode)':^64}")
print(SEP)
shown = defaultdict(int)
for r in classified:
    mode = r["failure_mode"]
    if mode == "correct" or shown[mode] >= 2:
        continue
    print(f"  [{mode}] {r['financebench_id']}")
    print(f"    Q:   {r['question'][:72]}")
    print(f"    doc: {r['doc_name']}")
    print(f"    gt={r['gt_score']} ac={r['ac_score']} aq={r['aq_score']}")
    if r["gt_reason"]:
        print(f"    why: {r['gt_reason'][:72]}")
    print()
    shown[mode] += 1

Path(OUT_CSV).parent.mkdir(exist_ok=True)
with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(classified[0].keys()))
    writer.writeheader()
    writer.writerows(classified)

print(f"  Saved -> {OUT_CSV}")
print(f"{'='*64}\n")


if __name__ == "__main__":
    pass

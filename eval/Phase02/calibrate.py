import csv

LABELS_CSV = "eval/Phase02/human_labels_30.csv"
JUDGE_CSV = "eval/Phase02/results_with_judge.csv"
SCORE_THRESHOLD = 4  # gt_score >= 4 → judge says "correct"
SCORE_FIELD = "gt_score"


def load_labels(path: str) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    missing = [r["financebench_id"] for r in rows if r["human_correct"] == ""]
    if missing:
        raise ValueError(f"Missing human_correct for: {missing}")
    return {r["financebench_id"]: int(r["human_correct"]) for r in rows}


def load_judge_scores(path: str, ids: set) -> dict[str, int]:
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["financebench_id"] in ids]
    return {r["financebench_id"]: int(r[SCORE_FIELD]) for r in rows}


def calibrate():
    labels = load_labels(LABELS_CSV)
    scores = load_judge_scores(JUDGE_CSV, set(labels.keys()))

    ids_missing = set(labels) - set(scores)
    if ids_missing:
        raise ValueError(f"IDs in labels but not in judge results: {ids_missing}")

    tp = tn = fp = fn = 0
    errors = []

    for fid, human in labels.items():
        judge_correct = 1 if scores[fid] >= SCORE_THRESHOLD else 0

        if human == 1 and judge_correct == 1:
            tp += 1
        elif human == 0 and judge_correct == 0:
            tn += 1
        elif human == 1 and judge_correct == 0:
            fn += 1
            errors.append((fid, "FN", human, scores[fid]))
        else:
            fp += 1
            errors.append((fid, "FP", human, scores[fid]))

    n = len(labels)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    acc = (tp + tn) / n

    print(f"\nCalibration — {n} examples, threshold {SCORE_FIELD} >= {SCORE_THRESHOLD}")
    print(f"  TPR (sensitivity): {tpr:.2f}  ({tp} TP / {tp+fn} positives)")
    print(f"  TNR (specificity): {tnr:.2f}  ({tn} TN / {tn+fp} negatives)")
    print(f"  Accuracy:          {acc:.2f}")
    print(f"\n  Confusion matrix:")
    print(f"             Judge+  Judge-")
    print(f"  Human+  :  {tp:5d}   {fn:5d}")
    print(f"  Human-  :  {fp:5d}   {tn:5d}")

    if errors:
        print(f"\n  Mismatches ({len(errors)}):")
        for fid, kind, h, s in errors:
            print(f"    {kind}  {fid}  human={h}  aq_score={s}")

    return {"tpr": tpr, "tnr": tnr, "accuracy": acc}


if __name__ == "__main__":
    calibrate()

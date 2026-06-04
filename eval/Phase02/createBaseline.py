from datasets import load_dataset
import csv

ds = load_dataset("PatronusAI/financebench")

with open("eval/baseline_100.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "financebench_id",
            "question",
            "expected_answer",
            "doc_name",
            "question_type",
            "model_answer",
            "correct",
            "notes",
        ]
    )

    for item in list(ds["train"])[:100]:
        writer.writerow(
            [
                item["financebench_id"],
                item["question"],
                item["answer"],
                item["doc_name"],
                item.get("question_type", ""),
                "",
                "",
                "",
            ]
        )

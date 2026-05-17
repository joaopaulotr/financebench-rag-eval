from datasets import load_dataset
ds = load_dataset("PatronusAI/financebench")
print(ds["train"][0])
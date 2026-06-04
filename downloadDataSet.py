import os
import requests
from datasets import load_dataset

ds = load_dataset("PatronusAI/financebench")
os.makedirs("data/pdfs", exist_ok=True)

doc_links = {}
for item in ds["train"]:
    doc_links[item["doc_name"]] = item["doc_link"]

print(f"Baixando {len(doc_links)} PDFs...")

for name, url in doc_links.items():
    path = f"data/pdfs/{name}.pdf"
    if os.path.exists(path):
        print(f"Ja existe: {name}")
        continue
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"OK: {name}")
    except Exception as e:
        print(f"ERRO: {name} - {e}")

print("Concluido.")

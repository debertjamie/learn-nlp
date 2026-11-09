import json
import os
import random
from datasets import load_dataset

N_DOCS = 10_000
MIN_CHARS = 200
MAX_CHARS = 1000
SEED = 42

os.makedirs("data", exist_ok=True)
ds = load_dataset("wikimedia/wikipedia", "20231101.id", split="train", streaming=True)
ds = ds.shuffle(seed=SEED, buffer_size=20_000)

docs = []
for row in ds:
    intro = row["text"].split("\n\n")[0].strip()
    if len(intro) < MIN_CHARS:
        continue
    docs.append({"id": row["id"], "title": row["title"], "text": intro[:MAX_CHARS]})
    if len(docs) >= N_DOCS:
        break

with open("data/docs.jsonl", "w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print(f"Saved {len(docs)} documents to data/docs.jsonl")
random.seed(SEED)
for d in random.sample(docs, 3):
    print("-", d["title"], "|", d["text"][:100], "...")
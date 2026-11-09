import json
import re
import sys
from rank_bm25 import BM25Okapi

def tokenize(text):
    return re.findall(r"\w+", text.lower())

def load_docs(path="data/docs.jsonl"):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def build_bm25(docs):
    return BM25Okapi([tokenize(d["text"]) for d in docs])

def search(bm25, docs, query, k=5):
    scores = bm25.get_scores(tokenize(query))
    top = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:k]
    return [(docs[i], float(scores[i])) for i in top]

if __name__ == "__main__":
    docs = load_docs()
    bm25 = build_bm25(docs)
    query = " ".join(sys.argv[1:]) or "kota pahlawan"
    for doc, score in search(bm25, docs, query):
        print(f"{score:6.2f}   {doc['title']}: {doc['text'][:90]}...")
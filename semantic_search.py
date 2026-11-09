import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from importlib import import_module

bm25_mod = import_module("baseline")
MODEL_NAME = "intfloat/multilingual-e5-small"
EMB_PATH = "data/embeddings.npy"

def doc_text(d):
    return "passage: " + d["title"] + ". " + d["text"]

def load_model():
    return SentenceTransformer(MODEL_NAME)

def get_embeddings(model, docs):
    if os.path.exists(EMB_PATH):
        emb = np.load(EMB_PATH)
        if len(emb) == len(docs):
            return emb
    emb = model.encode(
        [doc_text(d) for d in docs],
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    np.save(EMB_PATH, emb)
    return emb

def semantic_search(model, emb, docs, query, k=5):
    q = model.encode(["query: " + query], normalize_embeddings=True)[0]
    scores = emb @ q
    top = np.argsort(-scores)[:k]
    return [(docs[i], float(scores[i])) for i in top]

def show(title, results):
    print(f"=== {title} ===")
    for doc, score in results:
        print(f"{score:6.3f}   {doc['title']}: {doc['text'][:80]}...")

if __name__ == "__main__":
    args = sys.argv[1:]
    compare = "--compare" in args
    query = " ".join(a for a in args if a != "--compare") or "kota pahlawan"

    docs = bm25_mod.load_docs()
    model = load_model()
    emb = get_embeddings(model, docs)

    show("Semantic", semantic_search(model, emb, docs, query))
    if compare:
        bm25 =  bm25_mod.build_bm25(docs)
        show("BM25", bm25_mod.search(bm25, docs, query))
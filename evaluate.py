import json
from importlib import import_module

import numpy as np

bm = import_module("baseline")
sem = import_module("semantic_search")
hy = import_module("hybrid")

K = 10

HYBRID_CONFIGS = [(1.0, 60), (0.5, 60), (0.5, 10), (0.25, 10)]
MAIN_CONFIG = (0.5, 10)


def ranked_bm25(bm25, q):
    scores = bm25.get_scores(bm.tokenize(q))
    return [int(i) for i in np.argsort(-scores)[: hy.POOL] if scores[i] > 0]


def ranked_sem(model, emb, q):
    v = model.encode(["query: " + q], normalize_embeddings=True)[0]
    return [int(i) for i in np.argsort(-(emb @ v))[: hy.POOL]]


def metrics(results, queries, titles):
    hits5, rr = 0, 0.0
    for ranking, item in zip(results, queries):
        rel = set(item["relevant"])
        ranks = [r for r, i in enumerate(ranking[:K]) if titles[i] in rel]
        if ranks:
            if ranks[0] < 5:
                hits5 += 1
            rr += 1 / (ranks[0] + 1)
    n = len(queries)
    return hits5 / n, rr / n


if __name__ == "__main__":
    docs = bm.load_docs()
    titles = [d["title"] for d in docs]
    queries = json.load(open("data/queries.json", encoding="utf-8"))

    missing = [r for q in queries for r in q["relevant"] if r not in set(titles)]
    if missing:
        raise SystemExit(f"These 'relevant' titles are not in your corpus: {missing}")

    bm25 = bm.build_bm25(docs)
    model = sem.load_model()
    emb = sem.get_embeddings(model, docs)

    r_bm = [ranked_bm25(bm25, q["query"]) for q in queries]
    r_se = [ranked_sem(model, emb, q["query"]) for q in queries]

    def hybrid_runs(w, c):
        return [hy.fuse(a, b, w, c) for a, b in zip(r_bm, r_se)]

    print(f"{len(queries)} queries\n")
    rows = [("BM25", r_bm), ("Semantic", r_se)]
    for w, c in HYBRID_CONFIGS:
        rows.append((f"Hybrid w={w} C={c}", hybrid_runs(w, c)))
    print(f"{'method':20} {'Recall@5':>9} {'MRR@10':>8}")
    for name, res in rows:
        rec, mrr = metrics(res, queries, titles)
        print(f"{name:20} {rec:9.2f} {mrr:8.2f}")

    print("\nBy query type (Recall@5 / MRR@10):")
    runs = {"BM25": r_bm, "Semantic": r_se, "Hybrid": hybrid_runs(*MAIN_CONFIG)}
    print(f"{'type':18} {'n':>3}  " + "  ".join(f"{m:>14}" for m in runs))
    for tag in sorted({q.get("note", "paraphrase") for q in queries}):
        idx = [i for i, q in enumerate(queries) if q.get("note", "paraphrase") == tag]
        sub_q = [queries[i] for i in idx]
        cells = []
        for res in runs.values():
            rec, mrr = metrics([res[i] for i in idx], sub_q, titles)
            cells.append(f"{rec:.2f} / {mrr:.2f}")
        print(f"{tag:18} {len(idx):3}  " + "  ".join(f"{c:>14}" for c in cells))

    print(f"\nQueries where hybrid (w={MAIN_CONFIG[0]}, C={MAIN_CONFIG[1]}) MISSES the top 5:")
    for q, ranking in zip(queries, hybrid_runs(*MAIN_CONFIG)):
        rel = set(q["relevant"])
        if not any(titles[i] in rel for i in ranking[:5]):
            print(f"  {q['query']}  (wanted {q['relevant']})")
            for i in ranking[:3]:
                print(f"      got: {titles[i]}")

    print("\nQueries where semantic and BM25 disagree on top-5 success:")
    for q, a, b in zip(queries, r_bm, r_se):
        rel = set(q["relevant"])
        a_ok = any(titles[i] in rel for i in a[:5])
        b_ok = any(titles[i] in rel for i in b[:5])
        if a_ok != b_ok:
            print(f"  {'semantic' if b_ok else 'BM25':8} wins: {q['query']}")
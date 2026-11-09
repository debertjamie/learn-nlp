# Semantic Search vs BM25 on Indonesian Wikipedia

A small, evaluated comparison of keyword search (BM25) and embedding-based semantic search over 10,000 Indonesian Wikipedia article intros, with a side-by-side demo app. The goal is to learn how the two approaches differ, measure where each one wins, and practise evaluating a retrieval system instead of judging it by eye.

Status: early stage, a learning project. Numbers below come from a small test set.  
See [Limitations](#limitations).

Link: Soon.

## Summary of findings

On 33 test queries, semantic search found the correct article in the top 5 for **31 of 33**, BM25 for **22 of 33**.

- **Cross-language queries are the biggest gap.** English queries against Indonesian articles: BM25 found 0 of 6, semantic search 6 of 6. BM25 matches words, not meaning, and an English query shares almost no words with Indonesian text.
- **Indonesian paraphrases are closer.** BM25 9 of 14, semantic search 12 of 14.
- **Exact matches are a tie.** On 13 queries built around names, identifiers and abbreviations, both methods found every answer in the top 5. These queries turned out too easy to separate the methods.
- **Hybrid search did not beat semantic search alone.** (details below)
- **Search is not question answering.** For "siapa presiden pertama indonesia?" every method returned articles about Indonesian presidents, not the answer. The article on Soekarno was not in the 10,000-article sample, and the top semantic result (Fatmawati) only mentions him in passing. Retrieval finds relevant documents; extracting an answer is a separate step (the motivation for RAG).

## Dataset

- Hugging Face `wikimedia/wikipedia`, config `20231101.id` (Indonesian, November 2023 snapshot).
- Random sample of 10,000 articles, streamed with a fixed shuffle seed (42). Only the first paragraph of each article is kept (minimum 200 characters, truncated to 1,000), so the corpus is intros, not full articles.
- The sample contains many near-identical bot-generated stubs (asteroids, sponge species). I kept them as realistic noise and avoided them when writing general queries.

## Methods

| | |
|---|---|
| **BM25** | `rank_bm25` (Okapi), lowercase `\w+` tokenization over title + intro. No stemming or stopword removal. |
| **Semantic** | `intfloat/multilingual-e5-small` via `sentence-transformers`. Documents embedded as `passage: ...`, queries as `query: ...`, normalized, cosine similarity (dot product). Embeddings cached to disk. |
| **Hybrid** | Weighted reciprocal rank fusion over the top 20 results of each method: each document scores `w / (C + rank)` per list. BM25 weight `w` and constant `C` are configurable; semantic weight is fixed at 1. BM25 results with score 0 are dropped before fusion. |

## Evaluation

- **Test set:** 33 queries in `data/queries.json`, each with one article title marked correct. Types: 14 Indonesian paraphrases, 6 English, 13 exact-match (6 names, 3 identifiers, 3 abbreviations, 1 exact keywords).
- **Metrics:** Recall@5 (is the correct article in the top 5?) and MRR@10 (mean of 1 / rank of the correct article, 0 if outside the top 10).

### Overall

| Method | Recall@5 | MRR@10 |
|---|---:|---:|
| BM25 | 0.67 | 0.64 |
| Semantic | 0.94 | 0.90 |
| Hybrid, BM25 weight 1.0, C=60 | 0.94 | 0.77 |
| Hybrid, BM25 weight 0.5, C=60 | 0.94 | 0.84 |
| Hybrid, BM25 weight 0.5, C=10 | 0.94 | 0.88 |
| Hybrid, BM25 weight 0.25, C=10 | 0.94 | 0.90 |

### By query type (Recall@5 / MRR@10; hybrid = weight 0.5, C=10)

| Type | n | BM25 | Semantic | Hybrid |
|---|---:|---:|---:|---:|
| Paraphrase (Indonesian) | 14 | 0.64 / 0.63 | 0.86 / 0.81 | 0.86 / 0.80 |
| English | 6 | 0.00 / 0.00 | 1.00 / 1.00 | 1.00 / 0.81 |
| Exact name | 6 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| Exact identifier | 3 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |
| Abbreviation | 3 | 1.00 / 0.83 | 1.00 / 0.83 | 1.00 / 1.00 |
| Exact keywords | 1 | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 |

## What I learned

**Why BM25 fails on English queries.** BM25 only scores documents that share tokens with the query. For "capital city of china" it matched the words "China" and "city" in unrelated articles (an airport, a bank) and never reached Beijing. Semantic search ranked Beijing first, because the embedding model maps "capital city" close to "ibu kota" across languages.

**Why BM25 can't connect compound words.** For "ilmu yang memakai komputer untuk menganalisis data biologi" BM25 ranked the general "Informatika" article first. The target, "Bioinformatika", is one word, so it shares no token with "biologi" or "komputer".

**Equal-weight hybrid fusion hurt.** With BM25 at full weight, MRR fell from 0.90 to 0.77 even though Recall@5 stayed the same. BM25 is much weaker here, and its noisy matches also appear in semantic search's top results, so they get credit from both lists and overtake the correct answer. In an early demo configuration (50 candidates per method, C=60) the hybrid dropped Beijing from the top 5 for "capital city of China" entirely. A toy simulation showed why: with a large C, adjacent ranks score almost the same, so a single keyword hit can outvote a confident semantic #1. Lowering the BM25 weight and C reduces the damage, but with this test set hybrid at best equals semantic search. The one place it looked useful is abbreviations (MRR 1.00 vs 0.83), but that is only 3 queries.

**Similarity scores are not comparable across queries.** E5 cosine scores sit in a narrow band (roughly 0.7 to 0.9) even for poor matches, so only the ranking is meaningful. A score threshold such as "above 0.8 is relevant" would not work.

## Failure cases

Both are misses for semantic search and hybrid, and BM25 also misses the first:

- **"pengaruh calon presiden populer terhadap perolehan suara partai di pemilu legislatif"** (wanted: Efek ekor jas, the coat-tail effect). Returned general election articles. My reading: the long query is full of generic election words, so the model matched the topic but not the specific concept.
- **"provinsi di Polandia bagian barat laut dekat Laut Baltik"** (wanted: Voivodat Pomorze Zachodnie). Returned the Baltic Sea, Lithuania and Zeeland. My reading: the query relies on geography that the article's intro may not state, so there was nothing to match. This is arguably a flaw in the query, not the model.

## Limitations

- **Small test set.** 33 queries, with 1 to 14 per type. One query moves Recall@5 by about 3 points, so treat the numbers as indicative, not precise. No significance testing was done.
- **Query bias.** The queries were drafted by looking at the sample, with AI help, and each has a single marked answer. Some may have other valid answers, and I did not check for them systematically. The exact-match queries were easy for both methods, so they cannot show an advantage for either.
- **Hybrid settings were explored on the same queries.** The hybrid rows are exploratory, not tuned on a held-out set.
- **One embedding model, one corpus size.** Results may differ for other models, much larger corpora (rare-token behaviour in particular), or full articles with chunking.
- **Intros only.** Documents are truncated to the first paragraph (1,000 characters).

## Run it

```bash
uv venv
uv pip install -r requirements.txt
uv run 01_prepare_data.py        # saves 10,000 intros to data/docs.jsonl
uv run 02_bm25_baseline.py "kota pahlawan"
uv run 03_semantic_search.py --compare "capital city of China"
uv run 05_evaluate.py            # results tables above
uv run streamlit run app.py      # side-by-side demo
```

The first semantic run downloads the model (about 470 MB) and embeds the corpus, which takes a few minutes on CPU, then caches `data/embeddings.npy`.

## Files

| File | Purpose |
|---|---|
| `prepare.py` | Stream and sample the Wikipedia intros |
| `baseline.py` | BM25 search |
| `semantic_search.py` | Embedding search with cached vectors |
| `sample.py` | Print random articles to help write test queries |
| `evaluate.py` | Recall@5 / MRR@10 for BM25, semantic and hybrid, overall and by query type |
| `hybrid.py` | Weighted reciprocal rank fusion, shared by the app and evaluation |
| `app.py` | Streamlit demo with the three methods side by side |
| `data/queries.json` | Test queries and the article marked correct for each |

## Next steps

- Add a held-out query set to choose hybrid settings fairly.
- Try score-based fusion and a cross-encoder reranker.
- Compare other embedding models and test chunking on full articles.
- Put an LLM on top of this retriever (RAG), so the system returns an answer with sources instead of a list of articles.

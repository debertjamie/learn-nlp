from importlib import import_module

import numpy as np
import streamlit as st

bm = import_module("baseline")
sem = import_module("semantic_search")
hy = import_module("hybrid")

EXAMPLES = [
    "siapa presiden pertama Indonesia?",
    "capital city of China",
    "ilmu yang memakai komputer untuk menganalisis data biologi",
    "Kolkata",
]

st.set_page_config(page_title="Search Indonesian Wikipedia", layout="wide")


@st.cache_resource(show_spinner="Loading model and index...")
def load():
    docs = bm.load_docs()
    model = sem.load_model()
    return docs, bm.build_bm25(docs), model, sem.get_embeddings(model, docs)


docs, bm25, model, emb = load()

st.title("Search Indonesian Wikipedia")
st.caption(
    f"{len(docs):,} article intros. Compare keyword search (BM25), semantic search "
    "(multilingual-e5-small) and a weighted hybrid. Try English queries and paraphrases."
)

w_bm = st.sidebar.slider("BM25 weight in hybrid", 0.0, 1.0, 0.5, 0.05)
rrf_c = st.sidebar.slider(
    "Hybrid rank smoothing (C)", 1, 60, 10,
    help="Small C: top-ranked results count much more. Large C: a document found "
         "by both methods easily beats a confident #1 from one method.",
)
k = st.sidebar.slider("Results per method", 3, 10, 5)
st.sidebar.caption(
    "Semantic scores are cosine similarities. With e5 models they all sit in a narrow "
    "band (about 0.7 to 0.9), so compare the ranking, not the number."
)

if "q" not in st.session_state:
    st.session_state.q = EXAMPLES[1]
for col, ex in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    if col.button(ex, use_container_width=True):
        st.session_state.q = ex
query = st.text_input("Query (Indonesian or English)", key="q")


def render(col, title, ranking, scores=None):
    col.subheader(title)
    if not ranking:
        col.info("No documents share a word with this query.")
        return
    for n, i in enumerate(ranking[:k], 1):
        d = docs[i]
        tag = f"  `{scores[i]:.2f}`" if scores is not None else ""
        col.markdown(f"**{n}. {d['title']}**{tag}")
        col.caption(d["text"][:200] + "...")


if query.strip():
    b_scores = bm25.get_scores(bm.tokenize(query))
    b_rank = [int(i) for i in np.argsort(-b_scores)[: hy.POOL] if b_scores[i] > 0]
    qv = model.encode(["query: " + query], normalize_embeddings=True)[0]
    s_scores = emb @ qv
    s_rank = [int(i) for i in np.argsort(-s_scores)[: hy.POOL]]
    h_rank = hy.fuse(b_rank, s_rank, w_bm, rrf_c)

    c1, c2, c3 = st.columns(3)
    render(c1, "Keyword (BM25)", b_rank, b_scores)
    render(c2, "Semantic", s_rank, s_scores)
    render(c3, "Hybrid", h_rank)
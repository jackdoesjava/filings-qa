import json
import math

import numpy as np
import pytest

from fqa import retrieve
from fqa.retrieve import BM25, METHODS, Index, load_embeddings, rrf, save_embeddings, term_counts, tokenize

TOY = [
    {"doc": "ALPHA", "page": 0, "chunk": 0, "text": "Alpha Corp annual report for fiscal 2020"},
    {"doc": "ALPHA", "page": 3, "chunk": 1, "text": "Purchases of property and equipment were $1,577 million in 2020"},
    {"doc": "ALPHA", "page": 3, "chunk": 2, "text": "Cash flow from operations rose to 2,100 million"},
    {"doc": "ALPHA", "page": 7, "chunk": 3, "text": "Risk factors include competition and regulation"},
    {"doc": "BETA", "page": 0, "chunk": 0, "text": "Beta Inc quarterly report"},
    {"doc": "BETA", "page": 2, "chunk": 1, "text": "Revenue grew 12% on strong demand for equipment"},
    {"doc": "BETA", "page": 5, "chunk": 2, "text": "Purchases of equipment were 300 million"},
]
TOY_TERMS = sorted({t for c in TOY for t in tokenize(c["text"])})


def fake_encode(texts):
    """Unit-length bag of words over the toy vocabulary, standing in for bge-small."""
    out = np.zeros((len(texts), len(TOY_TERMS)), dtype=np.float32)
    for i, text in enumerate(texts):
        for t in tokenize(text):
            if t in TOY_TERMS:
                out[i, TOY_TERMS.index(t)] += 1
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(norms == 0, 1, norms)


def toy_index():
    return Index(TOY, emb=fake_encode([c["text"] for c in TOY]), encode=fake_encode)


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    def fail():
        raise AssertionError("tests must not load the embedding model")

    monkeypatch.setattr(retrieve, "load_model", fail)


# ---- tokeniser ---------------------------------------------------------------

@pytest.mark.parametrize("text, tokens", [
    ("Net Sales (in millions)", ["net", "sales", "millions"]),
    ("Capex was $1,577 million", ["capex", "1577", "million"]),
    ("1,577,000 and (1,234.5)", ["1577000", "1234", "5"]),
    ("pages 2,3,4", ["pages", "2", "3", "4"]),
    ("in 2018, 2019 and 2020", ["2018", "2019", "2020"]),
    ("What is the FY2018 capex?", ["2018", "capex"]),
    ("FY22 vs fy 2021", ["2022", "vs", "2021"]),
    ("3M's Form 10-K", ["3m", "form", "10", "k"]),
    ("the company’s revenue", ["company", "revenue"]),
    ("", []),
])
def test_tokenize(text, tokens):
    assert tokenize(text) == tokens


# ---- BM25 --------------------------------------------------------------------

def test_bm25_hand_computed():
    docs = [["cash", "flow", "cash"], ["cash", "revenue"], ["revenue", "growth", "margin", "growth"]]
    vocab = {}
    bm25 = BM25(term_counts(docs, vocab))
    # N = 3, lengths 3, 2, 4, so avgdl = 3
    # idf(cash) = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln 1.6 = 0.470004
    # idf(flow) = ln(1 + (3 - 1 + 0.5) / (1 + 0.5)) = ln 2.6667 = 0.980829
    # length factor k1 (1 - b + b dl / avgdl) = 1.5, 1.125, 1.875
    # cash in doc 0: 0.470004 * 2 * 2.5 / (2 + 1.5) = 0.671434
    # cash in doc 1: 0.470004 * 1 * 2.5 / (1 + 1.125) = 0.552945
    # flow in doc 0: 0.980829 * 1 * 2.5 / (1 + 1.5) = 0.980829
    # growth in doc 2: 0.980829 * 2 * 2.5 / (2 + 1.875) = 1.265586
    assert bm25.idf[vocab["cash"]] == pytest.approx(0.470004, abs=1e-6)
    assert bm25.idf[vocab["flow"]] == pytest.approx(0.980829, abs=1e-6)
    assert bm25.scores([vocab["cash"]]) == pytest.approx([0.671434, 0.552945, 0.0], abs=1e-6)
    assert bm25.scores([vocab["cash"], vocab["flow"]]) == pytest.approx([1.652263, 0.552945, 0.0], abs=1e-6)
    assert bm25.scores([vocab["growth"]]) == pytest.approx([0.0, 0.0, 1.265586], abs=1e-6)


def test_bm25_idf_stays_positive_for_a_term_in_every_passage():
    vocab = {}
    bm25 = BM25(term_counts([["the", "a"], ["the"], ["the", "b"]], vocab))
    # ln(1 + 0.5 / 3.5); the version without the +1 would be ln(0.5 / 3.5) < 0
    assert bm25.idf[vocab["the"]] == pytest.approx(math.log(1 + 0.5 / 3.5))
    assert (bm25.scores([vocab["the"]]) > 0).all()


def bm25_by_hand(docs, query, k1=1.5, b=0.75):
    n = len(docs)
    avgdl = sum(map(len, docs)) / n
    out = []
    for d in docs:
        s = 0.0
        for t in query:
            df = sum(t in other for other in docs)
            tf = d.count(t)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            s += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avgdl))
        out.append(s)
    return out


@pytest.mark.parametrize("k1, b", [(1.5, 0.75), (1.2, 0.0), (2.0, 1.0)])
def test_bm25_matches_the_formula_on_a_random_corpus(k1, b):
    rng = np.random.default_rng(0)
    words = [f"w{i}" for i in range(12)]
    docs = [[str(w) for w in rng.choice(words, rng.integers(1, 15))] for _ in range(40)]
    vocab = {}
    bm25 = BM25(term_counts(docs, vocab), k1=k1, b=b)
    for query in (["w0"], ["w1", "w5", "w5"], words[:6], ["w3", "not-a-term"]):
        got = bm25.scores(vocab[t] for t in query if t in vocab)
        assert got == pytest.approx(bm25_by_hand(docs, query, k1, b))


def test_bm25_repeated_query_term_counts_twice_and_empty_query_is_zero():
    vocab = {}
    bm25 = BM25(term_counts([["x", "y"], ["x"], ["z"]], vocab))
    once = bm25.scores([vocab["x"]])
    assert bm25.scores([vocab["x"], vocab["x"]]) == pytest.approx(2 * once)
    assert (bm25.scores([]) == 0).all()


# ---- reciprocal rank fusion --------------------------------------------------

def test_rrf_scores_and_order():
    s = rrf([[5, 1], [1, 2]], 6)
    assert s[1] == pytest.approx(1 / 61 + 1 / 62)
    assert s[5] == pytest.approx(1 / 61)
    assert s[2] == pytest.approx(1 / 62)
    assert s[0] == s[3] == s[4] == 0
    # second in both lists beats first in one list and missing from the other
    assert list(np.argsort(-s, kind="stable")[:3]) == [1, 5, 2]


def test_rrf_swapped_places_tie():
    s = rrf([[0, 1, 2], [1, 0, 2]], 3)
    assert s[0] == s[1] > s[2]


# ---- the index, on a toy corpus ---------------------------------------------

@pytest.mark.parametrize("method", METHODS)
def test_search_stays_inside_the_filing(method):
    res = toy_index().search("BETA", "purchases of equipment", k=None, method=method)
    assert res and {r["doc"] for r in res} == {"BETA"}
    assert [r["rank"] for r in res] == list(range(1, len(res) + 1))
    scores = [r["score"] for r in res]
    assert scores == sorted(scores, reverse=True)
    assert set(res[0]) == {"doc", "page", "chunk", "text", "score", "rank"}


@pytest.mark.parametrize("method", METHODS)
def test_search_finds_the_obvious_chunk(method):
    index = toy_index()
    top = index.search("ALPHA", "purchases of property and equipment", k=1, method=method)
    assert [(r["page"], r["chunk"]) for r in top] == [(3, 1)]
    # the filing writes $1,577; the query doesn't use a separator
    top = index.search("ALPHA", "capital spending of 1577 million", k=1, method=method)
    assert top[0]["chunk"] == 1


def test_search_all_pools_the_filings():
    index = toy_index()
    for method in METHODS:
        res = index.search_all("purchases of equipment", k=None, method=method)
        assert {r["doc"] for r in res} == {"ALPHA", "BETA"}
    top = index.search_all("beta quarterly", k=1, method="bm25")
    assert (top[0]["doc"], top[0]["chunk"]) == ("BETA", 0)


def test_single_filing_bm25_uses_that_filings_statistics():
    index = toy_index()
    query = "purchases of equipment million"
    beta = [tokenize(c["text"]) for c in TOY if c["doc"] == "BETA"]
    vocab = {}
    own = BM25(term_counts(beta, vocab)).scores(vocab[t] for t in tokenize(query) if t in vocab)
    res = index.search("BETA", query, k=None, method="bm25")
    assert {r["chunk"]: r["score"] for r in res} == pytest.approx({i: s for i, s in enumerate(own) if s > 0})
    # the pooled index gives the same chunk a different score (different idf and average length)
    pooled = {(r["doc"], r["chunk"]): r["score"] for r in index.search_all(query, k=None, method="bm25")}
    assert pooled[("BETA", 2)] != pytest.approx(res[0]["score"])


def test_bm25_leaves_out_chunks_without_a_query_term():
    index = toy_index()
    res = index.search("ALPHA", "competition", k=None, method="bm25")
    assert [r["chunk"] for r in res] == [3]
    assert index.search("ALPHA", "zebra", k=None, method="bm25") == []
    # dense ranks everything, so hybrid still returns the whole filing
    assert len(index.search("ALPHA", "zebra", k=None, method="hybrid")) == 4


def test_hybrid_is_rrf_of_bm25_and_dense():
    index = toy_index()
    query = "cash flow and purchases of equipment"
    bm25 = [r["chunk"] for r in index.search("ALPHA", query, k=None, method="bm25")]
    dense = [r["chunk"] for r in index.search("ALPHA", query, k=None, method="dense")]
    expected = rrf([bm25, dense], 4)
    hybrid = index.search("ALPHA", query, k=None, method="hybrid")
    assert [r["chunk"] for r in hybrid] == list(np.argsort(-expected, kind="stable"))
    assert [r["score"] for r in hybrid] == pytest.approx(sorted(expected, reverse=True))


def test_k_and_bad_arguments():
    index = toy_index()
    assert len(index.search("ALPHA", "million", k=2, method="dense")) == 2
    assert len(index.search("ALPHA", "million", k=50, method="dense")) == 4
    with pytest.raises(ValueError):
        index.search("ALPHA", "million", method="tfidf")
    with pytest.raises(KeyError):
        index.search("GAMMA", "million")
    with pytest.raises(ValueError):
        Index(TOY, emb=np.zeros((3, 4)), encode=fake_encode)


def test_results_are_copies():
    index = toy_index()
    index.search("ALPHA", "million", k=1, method="bm25")[0]["text"] = "changed"
    assert all(c["text"] != "changed" for c in index.chunks)
    assert "score" not in index.chunks[0]


def test_index_reads_chunks_jsonl(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text("".join(json.dumps(c) + "\n" for c in TOY), encoding="utf-8")
    index = Index(path, encode=fake_encode)
    assert index.chunks == TOY
    assert sorted(index.rows) == ["ALPHA", "BETA"]
    assert index.search("BETA", "revenue", k=1, method="bm25")[0]["chunk"] == 1


# ---- embedding cache ---------------------------------------------------------

def test_embedding_cache_round_trip_and_staleness(tmp_path):
    texts = [c["text"] for c in TOY]
    emb = fake_encode(texts)
    path = tmp_path / "emb.npy"
    with pytest.raises(FileNotFoundError):
        load_embeddings(texts, path)
    save_embeddings(emb, texts, path)
    assert np.array_equal(load_embeddings(texts, path), emb)
    with pytest.raises(ValueError):
        load_embeddings(texts[:-1] + ["a different last chunk"], path)
    with pytest.raises(ValueError):
        load_embeddings(texts[1:] + texts[:1], path)

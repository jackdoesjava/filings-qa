"""Retrieval over the filing chunks: BM25, bge-small embeddings, and reciprocal
rank fusion of the two.

BM25 is written out by hand so every number can be checked against the
formula. The idf is Robertson-Sparck Jones with +1 inside the log (the form
Lucene uses), so a term that is in more than half the chunks still counts for
a little instead of going negative. Embeddings are unit length, so dense
scores are cosine similarities."""

import hashlib
import json
import re
from collections.abc import Iterable
from functools import cache, cached_property
from pathlib import Path

import numpy as np
from scipy import sparse

from fqa.data import DATA

CHUNKS = DATA / "processed" / "chunks.jsonl"
EMB = DATA / "processed" / "emb_bge-small.npy"
MODEL = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
METHODS = ("bm25", "dense", "hybrid")
K1 = 1.5
B = 0.75
RRF_K = 60

STOPWORDS = frozenset(
    "a about an and are as at be been but by can could did do does for from had has have how i if in into is it "
    "its of on or our s so t than that the their them then there these they this those to was we were what when "
    "where which while who why will with would you your".split()
)
THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
# The questions say FY2018 or FY18; the filings almost always just say 2018.
FISCAL_YEAR = re.compile(r"\bfy ?(\d{4}|\d{2})\b")
WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    text = THOUSANDS.sub("", text.lower())
    text = FISCAL_YEAR.sub(lambda m: m[1] if len(m[1]) == 4 else "20" + m[1], text)
    return [t for t in WORD.findall(text) if t not in STOPWORDS]


def term_counts(docs: Iterable[list[str]], vocab: dict[str, int]) -> sparse.csr_matrix:
    """Term counts, one row per tokenised passage. New terms are added to vocab."""
    indptr, indices = [0], []
    for tokens in docs:
        indices.extend(vocab.setdefault(t, len(vocab)) for t in tokens)
        indptr.append(len(indices))
    counts = sparse.csr_matrix((np.ones(len(indices)), indices, indptr), shape=(len(indptr) - 1, len(vocab)))
    counts.sum_duplicates()
    return counts


class BM25:
    def __init__(self, counts, k1: float = K1, b: float = B):
        counts = sparse.csr_matrix(counts, dtype=np.float64, copy=True)
        counts.sum_duplicates()
        counts.eliminate_zeros()
        n, n_terms = counts.shape
        dl = np.asarray(counts.sum(axis=1)).ravel()
        avgdl = dl.mean() if dl.sum() > 0 else 1.0
        df = np.bincount(counts.indices, minlength=n_terms)
        self.idf = np.log((n - df + 0.5) / (df + 0.5) + 1)
        tf = counts.data
        row = np.repeat(np.arange(n), np.diff(counts.indptr))
        w = self.idf[counts.indices] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[row] / avgdl))
        self.weights = sparse.csr_matrix((w, counts.indices, counts.indptr), shape=counts.shape).tocsc()

    def scores(self, term_ids) -> np.ndarray:
        """Score of every passage for a query given as term ids. A term that is
        repeated in the query counts once per repeat."""
        term_ids = list(term_ids)
        if not term_ids:
            return np.zeros(self.weights.shape[0])
        return np.asarray(self.weights[:, term_ids].sum(axis=1)).ravel()


def rrf(rankings, n: int, k: int = RRF_K) -> np.ndarray:
    """Reciprocal rank fusion score for items 0..n-1. Each ranking lists item
    ids best first; an item missing from a ranking gets nothing from it."""
    score = np.zeros(n)
    for ranking in rankings:
        ranking = np.asarray(ranking, dtype=int)
        score[ranking] += 1.0 / (k + np.arange(1, len(ranking) + 1))
    return score


def best_first(scores: np.ndarray) -> np.ndarray:
    return np.argsort(-scores, kind="stable")


@cache
def load_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL, device="cpu")


def embed_queries(queries: list[str]) -> np.ndarray:
    return load_model().encode([QUERY_PREFIX + q for q in queries], normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=False)


def embed_passages(texts: list[str], batch_size: int = 16) -> np.ndarray:
    # no prefix on this side; bge only wants the instruction on queries
    vecs = load_model().encode(texts, batch_size=batch_size, normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=False)
    return np.asarray(vecs, dtype=np.float32)


def texts_sha1(texts: list[str]) -> str:
    h = hashlib.sha1()
    for t in texts:
        h.update(t.encode("utf-8") + b"\0")
    return h.hexdigest()


def save_embeddings(emb: np.ndarray, texts: list[str], path: Path = EMB) -> None:
    """Writes the vectors plus a .json beside them saying which chunk texts they came from."""
    meta = path.with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    meta.unlink(missing_ok=True)
    np.save(path, np.asarray(emb, dtype=np.float32))
    meta.write_text(json.dumps({"model": MODEL, "n": len(texts), "sha1": texts_sha1(texts)}), encoding="utf-8")


def load_embeddings(texts: list[str], path: Path = EMB) -> np.ndarray:
    """The cached vectors for these chunk texts. Refuses vectors built from different chunks."""
    meta = path.with_suffix(".json")
    if not path.exists() or not meta.exists():
        raise FileNotFoundError(f"no embeddings at {path}; run scripts/03_index.py")
    info = json.loads(meta.read_text(encoding="utf-8"))
    if info.get("model") != MODEL or info.get("sha1") != texts_sha1(texts):
        raise ValueError(f"{path} was built from different chunks; re-run scripts/03_index.py")
    emb = np.load(path)
    if len(emb) != len(texts):
        raise ValueError(f"{path} has {len(emb)} rows for {len(texts)} chunks; re-run scripts/03_index.py")
    return emb


def read_chunks(path: Path | str = CHUNKS) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class Index:
    """Every chunk of every filing. search() looks inside one filing, like the
    per-filing vector store in the FinanceBench paper, with BM25 statistics
    from that filing alone; search_all() pools every filing.

    Results are the chunk dicts plus "score" and "rank" (1 = best). k=None
    returns the whole ranking. BM25 leaves out chunks that share no term with
    the query, so it can return fewer than k.

    The cached embeddings and the query encoder are only loaded once a dense
    or hybrid search needs them; tests pass their own."""

    def __init__(self, chunks: list[dict] | Path | str | None = None, emb: np.ndarray | None = None,
                 encode=None):
        if chunks is None or isinstance(chunks, (str, Path)):
            chunks = read_chunks(chunks or CHUNKS)
        self.chunks = list(chunks)
        rows = {}
        for i, c in enumerate(self.chunks):
            rows.setdefault(c["doc"], []).append(i)
        self.rows = {doc: np.array(r) for doc, r in rows.items()}
        if emb is not None:
            emb = np.asarray(emb, dtype=np.float32)
            if len(emb) != len(self.chunks):
                raise ValueError(f"{len(emb)} embeddings for {len(self.chunks)} chunks")
            self.emb = emb
        self.encode = encode or embed_queries

    @cached_property
    def emb(self) -> np.ndarray:
        return load_embeddings([c["text"] for c in self.chunks])

    @cached_property
    def terms(self) -> tuple[dict[str, int], sparse.csr_matrix]:
        vocab = {}
        counts = term_counts((tokenize(c["text"]) for c in self.chunks), vocab)
        return vocab, counts

    @cached_property
    def bm25_all(self) -> BM25:
        return BM25(self.terms[1])

    def search(self, doc: str, query: str, k: int | None = 5, method: str = "hybrid") -> list[dict]:
        if doc not in self.rows:
            raise KeyError(f"no chunks for {doc!r}")
        return self._search(self.rows[doc], query, k, method)

    def search_all(self, query: str, k: int | None = 5, method: str = "hybrid") -> list[dict]:
        return self._search(None, query, k, method)

    # rows=None means every chunk; otherwise positions below are into rows
    def _bm25(self, rows, query):
        vocab, counts = self.terms
        bm25 = self.bm25_all if rows is None else BM25(counts[rows])
        s = bm25.scores(vocab[t] for t in tokenize(query) if t in vocab)
        order = best_first(s)
        order = order[s[order] > 0]
        return order, s[order]

    def _dense(self, rows, query):
        q = np.asarray(self.encode([query]), dtype=np.float32)[0]
        s = (self.emb if rows is None else self.emb[rows]) @ q
        order = best_first(s)
        return order, s[order]

    def _search(self, rows, query, k, method):
        if method == "bm25":
            order, scores = self._bm25(rows, query)
        elif method == "dense":
            order, scores = self._dense(rows, query)
        elif method == "hybrid":
            n = len(self.chunks) if rows is None else len(rows)
            fused = rrf([self._bm25(rows, query)[0], self._dense(rows, query)[0]], n)
            order = best_first(fused)
            scores = fused[order]
        else:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        ids = order if rows is None else rows[order]
        return [{**self.chunks[i], "score": float(s), "rank": r}
                for r, (i, s) in enumerate(zip(ids[:k], scores[:k]), start=1)]

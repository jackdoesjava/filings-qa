"""Embeds every chunk with bge-small-en-v1.5 and caches the vectors.

Run after 02_ingest.py. Writes data/processed/emb_bge-small.npy (one unit-length
row per line of chunks.jsonl, in the same order) and emb_bge-small.json beside
it, which records the model and a hash of the chunk texts so that vectors left
over from an older ingest are caught rather than silently misaligned. Does
nothing if the cache is already up to date.

This is the slow step on a CPU: bge-small manages about 14 chunks a second on a
6-core Ryzen 5 5500 (a chunk is about 240 tokens), so the whole corpus takes
the best part of an hour. Progress is saved after every block of chunks, so an
interrupted run carries on where it stopped. BM25 has no build step; an Index
tokenises every chunk the first time BM25 is used, which takes about five
seconds.

Ends by putting the first question through each method, as a check that the
page numbers line up with the gold evidence pages."""

import sys
import time

import numpy as np

from fqa.data import load_questions
from fqa.retrieve import (CHUNKS, EMB, METHODS, Index, embed_passages, load_embeddings, read_chunks,
                          save_embeddings, texts_sha1)

BLOCK = 512
PARTIAL = EMB.with_name(EMB.stem + ".partial.npz")
PARTIAL_TMP = EMB.with_name(EMB.stem + ".partial.tmp.npz")

if not CHUNKS.exists():
    sys.exit(f"{CHUNKS} not found; run scripts/02_ingest.py first")

chunks = read_chunks()
texts = [c["text"] for c in chunks]
words = np.array([len(t.split()) for t in texts])
print(f"{len(chunks)} chunks from {len({c['doc'] for c in chunks})} filings, "
      f"{words.mean():.0f} words per chunk on average")

try:
    emb = load_embeddings(texts)
    print(f"{EMB.name} is up to date")
except FileNotFoundError:
    emb = None
except ValueError:
    print(f"{EMB.name} was built from different chunks, embedding again")
    emb = None

if emb is None:
    # Longest chunks first: each batch is padded to about its own length, and
    # the time estimate errs on the slow side.
    sha1 = texts_sha1(texts)
    order = np.argsort([-len(t) for t in texts], kind="stable")
    done = 0
    if PARTIAL.exists():
        with np.load(PARTIAL) as saved:
            if str(saved["sha1"]) == sha1:
                emb, done = saved["emb"], int(saved["done"])
                print(f"carrying on from an interrupted run, {done} chunks already embedded")
    print(f"embedding {len(texts) - done} chunks on the CPU, {BLOCK} at a time", flush=True)
    t0, first = time.time(), done
    for start in range(done, len(texts), BLOCK):
        idx = order[start:start + BLOCK]
        vecs = embed_passages([texts[i] for i in idx])
        if emb is None:
            emb = np.zeros((len(texts), vecs.shape[1]), dtype=np.float32)
        emb[idx] = vecs
        done = start + len(idx)
        np.savez(PARTIAL_TMP, emb=emb, done=done, sha1=sha1)
        PARTIAL_TMP.replace(PARTIAL)
        rate = (done - first) / (time.time() - t0)
        print(f"  {done}/{len(texts)} chunks, {rate:.1f} a second, about {(len(texts) - done) / rate / 60:.0f} min "
              "to go", flush=True)
    save_embeddings(emb, texts)
    PARTIAL.unlink()
    print(f"wrote {EMB} ({(time.time() - t0) / 60:.1f} min)")

norms = np.linalg.norm(emb, axis=1)
print(f"{emb.shape[1]}-dim vectors, norms between {norms.min():.4f} and {norms.max():.4f}")

index = Index(chunks, emb=emb)
t0 = time.time()
vocab, _ = index.terms
print(f"BM25 vocabulary of {len(vocab)} terms, {time.time() - t0:.0f}s to tokenise every chunk")

q = load_questions()[0]
print(f"\n{q['id']} on {q['doc']}, gold pages {q['gold_pages']}")
for method in METHODS:
    top = index.search(q["doc"], q["question"], k=5, method=method)
    print(f"  {method:6s} top 5 chunks are on pages {[r['page'] for r in top]}")

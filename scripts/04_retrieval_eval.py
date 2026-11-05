"""Page-level retrieval evaluation on the FinanceBench questions.

One filing at a time: each question searches only the chunks of its own
filing (the paper's "single vector store"). A hit at k means at least one of
the top k chunks is from a gold evidence page. MRR uses the first chunk from a
gold page anywhere in the ranking, 0 if there is none.

Every filing at once: the same three methods over all chunks (the paper's
"shared vector store"), checking whether the right filing, the right company
and a gold page come back.

95% percentile bootstrap intervals over questions; differences between methods
are paired. Only the question text is used as the query. Writes
results/retrieval.json and reports/retrieval.md."""

import json
from collections import Counter
from math import comb

import numpy as np
import pandas as pd

from fqa.data import REPORTS, RESULTS, load_questions
from fqa.retrieve import METHODS, Index
from fqa.stats import bootstrap_ci, paired_diff_ci

KS = (1, 3, 5, 10)
CURVE = range(1, 21)
PAIRS = (("hybrid", "bm25"), ("dense", "bm25"))
SHARED = {"doc": "right filing", "company": "right company", "page": "gold page"}

questions = load_questions()
index = Index()
doc_company = {q["doc"]: q["company"] for q in questions}
pages_of = {doc: [index.chunks[i]["page"] for i in rows] for doc, rows in index.rows.items()}


def first_rank(results, hit):
    return next((r["rank"] for r in results if hit(r)), None)


def all_gold_rank(results, gold):
    """Rank by which every gold page has had a chunk returned; None if one never does."""
    first = {}
    for r in results:
        if r["page"] in gold:
            first.setdefault(r["page"], r["rank"])
    return max(first.values()) if len(first) == len(gold) else None


per_q = []
for i, q in enumerate(questions):
    doc, gold = q["doc"], set(q["gold_pages"])
    rec = {"id": q["id"], "doc": doc, "qtype": q["qtype"], "gold_pages": q["gold_pages"],
           "reachable": bool(gold & set(pages_of.get(doc, []))), "single": {}, "shared": {}}
    for m in METHODS:
        res = index.search(doc, q["question"], k=None, method=m) if doc in index.rows else []
        rec["single"][m] = {"first": first_rank(res, lambda r: r["page"] in gold),
                            "all": all_gold_rank(res, gold)}
        top = index.search_all(q["question"], k=max(CURVE), method=m)
        rec["shared"][m] = {"doc": first_rank(top, lambda r: r["doc"] == doc),
                            "company": first_rank(top, lambda r: doc_company.get(r["doc"]) == q["company"]),
                            "page": first_rank(top, lambda r: r["doc"] == doc and r["page"] in gold)}
    per_q.append(rec)
    if (i + 1) % 25 == 0 or i + 1 == len(questions):
        print(f"{i + 1}/{len(questions)} questions searched", flush=True)


def ranks(setting, method, key, recs=per_q):
    return [r[setting][method][key] for r in recs]


def hit_at(rank_list, k):
    return np.array([r is not None and r <= k for r in rank_list], dtype=float)


def recip(rank_list):
    return np.array([1 / r if r else 0.0 for r in rank_list])


def ci(x):
    mean, lo, hi = bootstrap_ci(x)
    return {"mean": mean, "lo": lo, "hi": hi}


def diff_ci(a, b):
    diff, lo, hi = paired_diff_ci(a, b)
    return {"diff": diff, "lo": lo, "hi": hi}


def random_recall(q, k):
    """Chance that k of the filing's chunks drawn at random include one from a gold page."""
    pages = pages_of.get(q["doc"], [])
    n, g = len(pages), sum(p in q["gold_pages"] for p in pages)
    k = min(k, n)
    return 1 - comb(n - g, k) / comb(n, k) if n else 0.0


single, single_diff, curves = {}, {}, {"k": list(CURVE), "single": {}, "shared": {}}
for m in METHODS:
    first = ranks("single", m, "first")
    single[m] = {f"recall@{k}": ci(hit_at(first, k)) for k in KS}
    single[m]["mrr"] = ci(recip(first))
    curves["single"][m] = [float(hit_at(first, k).mean()) for k in CURVE]
for a, b in PAIRS:
    fa, fb = ranks("single", a, "first"), ranks("single", b, "first")
    single_diff[f"{a} - {b}"] = {f"recall@{k}": diff_ci(hit_at(fa, k), hit_at(fb, k)) for k in KS}
    single_diff[f"{a} - {b}"]["mrr"] = diff_ci(recip(fa), recip(fb))

random_baseline = {f"recall@{k}": float(np.mean([random_recall(q, k) for q in questions])) for k in KS}

multi = [r for r in per_q if len(r["gold_pages"]) > 1]
multi_page = {}
for m in METHODS:
    every = ranks("single", m, "all", multi)
    multi_page[m] = {"n": len(multi), **{f"all_gold@{k}": ci(hit_at(every, k)) for k in KS}}

qtypes = sorted({r["qtype"] for r in per_q})
by_qtype = {}
for qt in qtypes:
    recs = [r for r in per_q if r["qtype"] == qt]
    by_qtype[qt] = {m: {"n": len(recs), "recall@5": ci(hit_at(ranks("single", m, "first", recs), 5)),
                        "mrr": ci(recip(ranks("single", m, "first", recs)))} for m in METHODS}

shared, shared_diff = {}, {}
for m in METHODS:
    shared[m] = {f"{key}@{k}": ci(hit_at(ranks("shared", m, key), k)) for key in SHARED for k in KS}
    curves["shared"][m] = [float(hit_at(ranks("shared", m, "page"), k).mean()) for k in CURVE]
for a, b in PAIRS:
    pa, pb = ranks("shared", a, "page"), ranks("shared", b, "page")
    shared_diff[f"{a} - {b}"] = {f"page@{k}": diff_ci(hit_at(pa, k), hit_at(pb, k)) for k in KS}

unreachable = [r["id"] for r in per_q if not r["reachable"]]
missing = sorted({q["doc"] for q in questions} - set(index.rows))
per_doc = np.array([len(rows) for rows in index.rows.values()])
words = np.array([len(c["text"].split()) for c in index.chunks])
filings_per_company = Counter(doc_company.values())

out = {
    "n_questions": len(questions), "n_chunks": len(index.chunks), "n_filings": len(index.rows), "ks": list(KS),
    "single": single, "single_diff": single_diff, "random": random_baseline, "multi_page": multi_page,
    "by_qtype": by_qtype, "shared": shared, "shared_diff": shared_diff, "curves": curves,
    "unreachable": unreachable, "missing_docs": missing, "per_question": per_q,
}
RESULTS.mkdir(parents=True, exist_ok=True)
(RESULTS / "retrieval.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


# ---- write up ---------------------------------------------------------------
def fmt(c):
    return f"{c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]"


def fmt_diff(c):
    return f"{c['diff']:+.3f} [{c['lo']:+.3f}, {c['hi']:+.3f}]"


def md(rows):
    return pd.DataFrame(rows).to_markdown(index=False)


main_rows = [{"method": m, **{f"recall@{k}": fmt(single[m][f"recall@{k}"]) for k in KS},
              "MRR": fmt(single[m]["mrr"])} for m in METHODS]
main_rows.append({"method": "random order (expected)",
                  **{f"recall@{k}": f"{random_baseline[f'recall@{k}']:.3f}" for k in KS}, "MRR": ""})
diff_rows = [{"comparison": name, **{f"recall@{k}": fmt_diff(d[f"recall@{k}"]) for k in KS},
              "MRR": fmt_diff(d["mrr"])} for name, d in single_diff.items()]
multi_rows = [{"method": m, **{f"all gold pages@{k}": fmt(multi_page[m][f"all_gold@{k}"]) for k in KS}}
              for m in METHODS]
qtype_rows = [{"method": m, **{f"{qt} (n={by_qtype[qt][m]['n']})": fmt(by_qtype[qt][m]["recall@5"])
                               for qt in qtypes}} for m in METHODS]
shared_rows = [{"method": m, "measure": label, **{f"@{k}": fmt(shared[m][f"{key}@{k}"]) for k in KS}}
               for m in METHODS for key, label in SHARED.items()]
shared_diff_rows = [{"comparison": name, **{f"gold page@{k}": fmt_diff(d[f"page@{k}"]) for k in KS}}
                    for name, d in shared_diff.items()]

notes = []
if unreachable:
    notes.append(f"Questions with no parsed text on any gold page, which no method can find and which count as "
                 f"misses above: {len(unreachable)} ({', '.join(unreachable)}).")
if missing:
    notes.append(f"Filings with no chunks at all, whose questions count as misses: {', '.join(missing)}.")

REPORTS.mkdir(parents=True, exist_ok=True)
with open(REPORTS / "retrieval.md", "w", encoding="utf-8") as fh:
    fh.write(
        "# Retrieval\n\n"
        f"Generated by scripts/04_retrieval_eval.py. {len(questions)} questions; {len(index.chunks)} chunks "
        f"(overlapping windows of up to {words.max()} words that never cross a page, {words.mean():.0f} words on "
        f"average) from {len(index.rows)} filings, between {per_doc.min()} and {per_doc.max()} chunks per filing "
        f"(median {np.median(per_doc):.0f}).\n\n"
        "A hit at k means at least one of the top k chunks is from a gold evidence page. k counts chunks, so k = 5 "
        f"is about {5 * words.mean():.0f} words, the amount of context the RAG answerer gets. MRR is the mean of "
        "1/rank of the first chunk from a gold page, 0 if none comes back. Intervals are 95% percentile bootstraps "
        "over questions (2,000 resamples); differences are paired, using the same resampled questions for both "
        "methods.\n\n"
        "## One filing at a time\n\n"
        "Each question searches only its own filing, as in the FinanceBench paper's single vector store, and BM25 "
        "takes its statistics (idf, average chunk length) from that filing alone. The random row is what a random "
        "ordering of the filing's chunks would get on average.\n\n"
        + md(main_rows) + "\n\n" + "".join(n + "\n\n" for n in notes)
        + "Paired differences:\n\n" + md(diff_rows) + "\n\n"
        f"Questions with more than one gold page (n={len(multi)}): share where every gold page has a chunk in the "
        "top k.\n\n" + md(multi_rows) + "\n\n"
        "Recall@5 by question type:\n\n" + md(qtype_rows) + "\n\n"
        "## Every filing in one pool\n\n"
        f"All {len(index.chunks)} chunks in one index, as in the paper's shared vector store. Right filing: a chunk "
        "of the question's filing is in the top k. Right company: a chunk from any filing of the same company "
        f"(companies with more than one filing here: {sum(n > 1 for n in filings_per_company.values())} of "
        f"{len(filings_per_company)}). Gold page: a chunk from a gold page of the right filing.\n\n"
        + md(shared_rows) + "\n\n"
        "Gold page, paired differences:\n\n" + md(shared_diff_rows) + "\n"
    )
print(md(main_rows))
print(f"wrote {RESULTS / 'retrieval.json'} and {REPORTS / 'retrieval.md'}")

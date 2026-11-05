"""Answers the FinanceBench questions with one configuration and one model.

    python scripts/05_answer.py --config rag_hybrid --model qwen2.5:3b [--limit 20]

Writes results/answers/<config>__<model>.jsonl (":" in the model name becomes
"-"), one line per question, as each answer comes back. Questions already in
the file are skipped, so a run can be stopped and started again. Calls go to
Ollama one at a time; the GPU only holds one model anyway.

Needs data/processed from 02_ingest.py, and for rag_dense, rag_hybrid and agent the
embeddings from 03_index.py."""

import argparse
import json
import sys
import time

from fqa.agent import agent, closed_book, oracle, rag
from fqa.data import PROCESSED, RESULTS, load_questions, read_jsonl

CONFIGS = ("closed_book", "rag_bm25", "rag_dense", "rag_hybrid", "oracle", "agent")


def out_path(config, model):
    return RESULTS / "answers" / f"{config}__{model.replace(':', '-')}.jsonl"


def load_pages():
    pages = {}
    for row in read_jsonl(PROCESSED / "pages.jsonl"):
        pages.setdefault(row["doc"], {})[row["page"]] = row["text"]
    return pages


def answerer(config, model):
    if config == "closed_book":
        return lambda q: closed_book(q, model)
    if config == "oracle":
        pages = load_pages()
        return lambda q: oracle(q, model, pages)
    from fqa.retrieve import Index

    index = Index()
    if config == "agent":
        return lambda q: agent(q, model, index)
    return lambda q: rag(q, model, index, config.removeprefix("rag_"))


def run(questions, answer, path):
    done = {row["id"] for row in read_jsonl(path)} if path.exists() else set()
    todo = [q for q in questions if q["id"] not in done]
    print(f"{path.name}: {len(done)} already answered, {len(todo)} to go")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for i, q in enumerate(todo, 1):
            t0 = time.time()
            rec = answer(q)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            last = rec["answer"].splitlines()[-1] if rec["answer"] else "(empty)"
            print(f"[{i}/{len(todo)}] {q['id']} {time.time() - t0:5.1f}s {rec['steps']} call(s)  {last[:100]}")


def main():
    p = argparse.ArgumentParser(description="Answer the FinanceBench questions with one config and one model.")
    p.add_argument("--config", required=True, choices=CONFIGS)
    p.add_argument("--model", required=True, help="Ollama model name, e.g. qwen2.5:3b")
    p.add_argument("--limit", type=int, help="only the first N questions")
    args = p.parse_args()
    questions = load_questions()[: args.limit]
    run(questions, answerer(args.config, args.model), out_path(args.config, args.model))


if __name__ == "__main__":
    # model replies can hold characters the Windows console can't print
    sys.stdout.reconfigure(errors="replace")
    main()

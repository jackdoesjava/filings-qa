"""What the agent actually did, from results/answers/agent__*.jsonl, against
rag_hybrid for the same model (same retriever, same k, but the question itself
as the query), split by whether the model searched at all. Accuracy comes
from results/scores.jsonl when it is there. Writes reports/agent.md. No model
calls."""

import json
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

from fqa.agent import arg
from fqa.data import REPORTS, RESULTS, load_questions
from fqa.score import has_answer_line, retrieval_hit
from fqa.stats import paired_diff_ci

ANSWERS = RESULTS / "answers"
SCORES = RESULTS / "scores.jsonl"
# the first runs, made while the calculate tool's description had a worked example
# with real numbers in it, (2453 - 2108) / 2108 * 100
FIRST_RUNS = RESULTS / "agent_tool_example"
PLACEHOLDER = "(new - old) / old * 100"


def expression(call):
    return arg(call["arguments"], "expression")


def copies_placeholder(call):
    # the description's formula word for word
    return call["name"] == "calculate" and " ".join(str(expression(call)).split()) == PLACEHOLDER


def uses_placeholder_names(call):
    # its variable names, new and old, however garbled the rest
    return call["name"] == "calculate" and bool(re.search(r"\bnew\b", str(expression(call)))) \
        and bool(re.search(r"\bold\b", str(expression(call))))


def copies_example(call):
    return call["name"] == "calculate" and "2453" in expression(call) and "2108" in expression(call)


def read(path):
    with open(path, encoding="utf-8") as fh:
        return {r["id"]: r for r in map(json.loads, fh)}


def read_scores():
    if not SCORES.exists():
        return {}
    with open(SCORES, encoding="utf-8") as fh:
        return {(r["id"], r["config"], r["model"]): r["correct"] for r in map(json.loads, fh)}


def searched(rec):
    return any(c["name"] == "search" for c in rec["tool_calls"])


def summary(model, agent, rag, questions):
    ids = [i for i in agent if i in rag and i in questions]
    searches = [[c for c in agent[i]["tool_calls"] if c["name"] == "search"] for i in ids]
    calcs = [c for i in ids for c in agent[i]["tool_calls"] if c["name"] == "calculate"]
    first = [arg(s[0]["arguments"], "query") for s in searches if s]
    hit_agent = np.array([float(retrieval_hit(questions[i], agent[i])) for i in ids])
    hit_rag = np.array([float(retrieval_hit(questions[i], rag[i])) for i in ids])
    d, lo, hi = paired_diff_ci(hit_agent, hit_rag)
    n_search = Counter(min(len(s), 3) for s in searches)
    row = {
        "model": model,
        "questions": len(ids),
        "no search": n_search[0],
        "1 search": n_search[1],
        "2 searches": n_search[2],
        "3+ searches": n_search[3],
        "words in first query (mean)": np.mean([len(str(q).split()) for q in first]) if first else None,
        "words in question (mean)": np.mean([len(questions[i]["question"].split()) for i in ids]),
        "gold page shown, agent %": 100 * hit_agent.mean(),
        "gold page shown, rag_hybrid %": 100 * hit_rag.mean(),
        "difference, points [95% CI]": f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]",
        "calculate calls": len(calcs),
        "calculate errors": sum(str(c["result"]).startswith("Error") for c in calcs),
        "questions where calculate got the description's formula word for word":
            sum(any(copies_placeholder(c) for c in agent[i]["tool_calls"]) for i in ids),
        "questions where calculate got its variable names (new, old)":
            sum(any(uses_placeholder_names(c) for c in agent[i]["tool_calls"]) for i in ids),
        "tool calls written as text": sum(c["from_text"] for i in ids for c in agent[i]["tool_calls"]),
        "replies with no Answer line": sum(not has_answer_line(agent[i]["answer"]) for i in ids),
        "model calls / q": np.mean([agent[i]["steps"] for i in ids]),
    }
    return {k: f"{v:.1f}" if isinstance(v, float) else v for k, v in row.items()}


def split(model, agent, rag, questions, scores):
    """The same comparison for the questions where the model searched and the ones where it did not."""
    ids = [i for i in agent if i in rag and i in questions]
    row = {"model": model}
    for label, part in (("with a search", [i for i in ids if searched(agent[i])]),
                        ("with no search", [i for i in ids if not searched(agent[i])])):
        row[f"questions {label}"] = len(part)
        if not part:
            continue
        hit_agent = np.array([float(retrieval_hit(questions[i], agent[i])) for i in part])
        hit_rag = np.array([float(retrieval_hit(questions[i], rag[i])) for i in part])
        row[f"{label}: gold page shown, agent %"] = 100 * hit_agent.mean()
        row[f"{label}: gold page shown, rag_hybrid %"] = 100 * hit_rag.mean()
        d, lo, hi = paired_diff_ci(hit_agent, hit_rag)
        row[f"{label}: difference, points [95% CI]"] = f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
        right = [(scores.get((i, "agent", model)), scores.get((i, "rag_hybrid", model))) for i in part]
        if all(a is not None and b is not None for a, b in right):
            row[f"{label}: correct, agent %"] = 100 * np.mean([a for a, _ in right])
            row[f"{label}: correct, rag_hybrid %"] = 100 * np.mean([b for _, b in right])
    return {k: f"{v:.1f}" if isinstance(v, float) else v for k, v in row.items()}


def main():
    questions = {q["id"]: q for q in load_questions()}
    scores = read_scores()
    rows, splits = [], []
    for path in sorted(ANSWERS.glob("agent__*.jsonl")):
        rag = ANSWERS / f"rag_hybrid__{path.stem.partition('__')[2]}.jsonl"
        if rag.exists():
            agent, rag = read(path), read(rag)
            model = next(iter(agent.values()))["model"]
            rows.append(summary(model, agent, rag, questions))
            splits.append(split(model, agent, rag, questions, scores))
    if not rows:
        print("no agent and rag_hybrid answer files for the same model yet")
        return
    table = pd.DataFrame(rows).set_index("model").T
    out = ["# What the agent did\n",
           "Generated by scripts/08_agent_report.py from results/answers/. The agent's search tool is the same "
           "hybrid retriever with the same k as rag_hybrid, so each search differs from rag_hybrid's only in "
           "its query. The model also decides whether to search at all and how often, and a question it never "
           "searches shows it nothing. Gold page shown: some chunk the model was shown is from a gold evidence "
           "page. Differences are paired over the same questions (95% bootstrap interval). An error from "
           "calculate is a message sent back to the model, such as a variable name in the expression; the "
           "model can try again.\n",
           table.to_markdown() + "\n",
           "## Questions with and without a search\n",
           "The same comparison, split by whether the model made at least one search. With no search the agent "
           "is shown nothing, so the rag_hybrid column there is what the question itself would have found."
           + ("" if scores else " Accuracy is left out until results/scores.jsonl exists.") + "\n",
           pd.DataFrame(splits).set_index("model").T.to_markdown() + "\n"]
    first = []
    for path in sorted(FIRST_RUNS.glob("agent__*.jsonl")):
        runs = read(path)
        first.append({"model": next(iter(runs.values()))["model"], "questions": len(runs),
                      "calculate called with the example's numbers":
                          sum(any(copies_example(c) for c in r["tool_calls"]) for r in runs.values())})
    if first:
        out += ["## The first runs\n",
                "The first agent runs, in results/agent_tool_example/, were made while the calculate tool's "
                "description had a worked example with real numbers in it, (2453 - 2108) / 2108 * 100. They "
                "were replaced by the runs above once it was clear the small models were copying it.\n",
                pd.DataFrame(first).to_markdown(index=False) + "\n"]
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "agent.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"wrote {REPORTS / 'agent.md'}")


if __name__ == "__main__":
    sys.stdout.reconfigure(errors="replace")
    main()

"""Figures for the write-up, into reports/figures/:

- accuracy.png: answer accuracy for every config and model with 95% bootstrap
  intervals, from results/scores.jsonl (scripts/06_score.py)
- recall_at_k.png: page-level recall@k for each retrieval method, one filing at
  a time and all filings pooled, from results/retrieval.json
  (scripts/04_retrieval_eval.py)

A figure is skipped if its input is not there yet."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fqa.data import REPORTS, RESULTS
from fqa.stats import bootstrap_ci

FIG = REPORTS / "figures"
# Okabe-Ito blue, vermillion and bluish green, as in the last project; the three
# stay apart under the common kinds of colour blindness and clear 3:1 on white.
# Slots go by name, not by what happens to be plotted, so a model keeps its colour.
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
MARKERS = ["o", "s", "^", "D", "v", "P"]
MODELS = ["qwen2.5:3b", "qwen2.5:1.5b", "llama3.2:3b"]
METHODS = ["bm25", "dense", "hybrid"]
CONFIGS = ["closed_book", "rag_bm25", "rag_dense", "rag_hybrid", "agent", "oracle"]
INK, MUTED, GREY = "#222222", "#6b6b6b", "#9a9a9a"


def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#b5b5b5",
        "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": "#e4e4e4", "grid.linewidth": 0.6,
        "lines.linewidth": 1.8, "font.size": 9, "legend.frameon": False,
    })


def in_order(names, preferred):
    known = [n for n in preferred if n in set(names)]
    return known + sorted(set(names) - set(known))


def slots(names, preferred):
    extra = [n for n in in_order(names, preferred) if n not in preferred]
    return {n: preferred.index(n) if n in preferred else len(preferred) + extra.index(n) for n in names}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name)
    plt.close(fig)
    print(f"wrote {FIG / name}")


def accuracy_figure(scores: pd.DataFrame):
    df = scores[scores["correct"].notna()]
    models = in_order(df["model"].unique(), MODELS)
    configs = in_order(df["config"].unique(), CONFIGS)
    slot = slots(models, MODELS)
    offsets = np.linspace(-0.22, 0.22, len(models)) if len(models) > 1 else [0.0]
    fig, ax = plt.subplots(figsize=(6.4, 0.6 + 0.55 * len(configs)))
    sizes = set()
    for j, (m, dy) in enumerate(zip(models, offsets)):
        color, marker, labelled = COLORS[slot[m] % len(COLORS)], MARKERS[slot[m] % len(MARKERS)], False
        for i, c in enumerate(configs):
            x = df.loc[(df["model"] == m) & (df["config"] == c), "correct"].astype(float)
            if x.empty:
                continue
            sizes.add(len(x))
            mean, lo, hi = (100 * v for v in bootstrap_ci(x))
            ax.plot([lo, hi], [i + dy, i + dy], color=color, lw=2, solid_capstyle="round", clip_on=False)
            ax.plot(mean, i + dy, marker, color=color, ms=7, mec="white", mew=1.2, clip_on=False, zorder=3,
                    label=None if labelled else m)
            labelled = True
            if j == 0:  # values for the first model only; the rest are in reports/accuracy.md
                ax.text(hi + 1.5, i + dy, f"{mean:.0f}%", va="center", color=MUTED, fontsize=8)
    ax.set_yticks(range(len(configs)), configs, color=INK)
    ax.set_ylim(len(configs) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("accuracy (%): dot = estimate, bar = 95% bootstrap interval over questions")
    n = f"{sizes.pop()} questions per row" if len(sizes) == 1 else "partial runs, so the number of questions differs by row"
    title = "Answer accuracy on FinanceBench" + (f", {models[0]}" if len(models) == 1 else "")
    ax.set_title(f"{title} ({n})", loc="left", pad=24 if len(models) > 1 else 6)
    if len(models) > 1:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(models), borderaxespad=0.3,
                  handletextpad=0.3, columnspacing=1.5)
    save(fig, "accuracy.png")


def recall_figure(ret: dict):
    k = ret["curves"]["k"]
    ks = [x for x in ret["ks"] if x in k]
    panels = [("single", "One filing at a time", "recall@{}"), ("shared", "All filings in one pool", "page@{}")]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), sharey=True)
    for ax, (key, title, name) in zip(axes, panels):
        methods = in_order(ret["curves"][key], METHODS)
        slot = slots(methods, METHODS)
        for m in methods:
            color, marker = COLORS[slot[m] % len(COLORS)], MARKERS[slot[m] % len(MARKERS)]
            cis = [ret[key][m][name.format(x)] for x in ks]
            mean = np.array([c["mean"] for c in cis])
            ax.errorbar(ks, mean, yerr=[mean - [c["lo"] for c in cis], [c["hi"] for c in cis] - mean],
                        fmt="none", ecolor=color, elinewidth=1.2, zorder=1)
            ax.plot(k, ret["curves"][key][m], color=color, marker=marker, ms=6, mec="white", mew=1.0,
                    markevery=[k.index(x) for x in ks], label=m, zorder=3)
        if key == "single" and "random" in ret:
            ax.plot(ks, [ret["random"][f"recall@{x}"] for x in ks], color=GREY, lw=1.2, marker=".",
                    label="chunks in random order")
        ax.axvline(5, color=GREY, lw=0.8, zorder=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("k, chunks retrieved")
        ax.set_xticks([1, 5, 10, 15, 20])
        ax.set_xlim(0.5, max(k) + 0.5)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("questions with a gold page in the top k")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=len(labels))
    fig.suptitle("Page-level recall@k. Bars are 95% bootstrap intervals over questions; the grey line is k = 5, "
                 "what the RAG configs are given.", x=0.06, y=1.02, ha="left", fontsize=9, color=INK)
    save(fig, "recall_at_k.png")


def main():
    setup()
    path = RESULTS / "scores.jsonl"
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            scores = pd.DataFrame([json.loads(line) for line in fh if line.strip()])
        if len(scores) and scores["correct"].notna().any():
            accuracy_figure(scores)
        else:
            print("skipping accuracy.png: nothing graded in results/scores.jsonl")
    else:
        print("skipping accuracy.png: run scripts/06_score.py first")

    path = RESULTS / "retrieval.json"
    if path.exists():
        recall_figure(json.loads(path.read_text(encoding="utf-8")))
    else:
        print("skipping recall_at_k.png: run scripts/04_retrieval_eval.py first")


if __name__ == "__main__":
    main()

# Question answering over company filings

FinanceBench (Islam et al., 2023) is a set of questions about real company filings, written by financial analysts, each with the answer and the page of the filing it comes from. In the paper, GPT-4-Turbo answered 19% of them correctly with one vector store shared by all the filings (the "81% wrong or refused" in its abstract), 50% with a separate store for each filing, which is the setting used here, and 85% with the evidence pages pasted in. The paper graded by hand and counted answers in the wrong unit as correct; here they are wrong and 88 of the questions are graded by a 7B judge, so the figures are only roughly comparable. I wanted to see how far small models running on my own machine get (1.5 to 3 billion parameters, on a 4GB GTX 1050 Ti), and more usefully, where their answers go wrong: not finding the right page, or getting it wrong with the right page in front of them.

The open-source sample is 150 questions over 84 filings from 32 companies. Each question is answered six ways: with no filing at all (closed book); with the top five passages from BM25, from a dense retriever or from a hybrid of the two (RAG); with the gold evidence pages pasted in (the oracle, which shows what a model does once retrieval is perfect); and by a small agent that can search the filing and use a calculator. Questions whose gold answer is a single number are graded by comparing numbers. The rest go to a larger local model as a judge, and its agreement with the automatic grade is measured on the questions where both apply.

The short version, accuracy % over the 150 questions with 95% bootstrap intervals:

| config | qwen2.5:3b | llama3.2:3b | qwen2.5:1.5b |
|:--|:--|:--|:--|
| closed_book | 14.7 [9.3, 20.7] | 8.7 [4.7, 13.3] | 10.0 [5.3, 15.3] |
| rag_bm25 | 26.7 [19.3, 33.3] | 22.0 [15.3, 28.7] | 24.0 [17.3, 30.7] |
| rag_dense | 32.0 [24.7, 39.3] | 28.7 [22.0, 36.7] | 23.3 [16.7, 30.0] |
| rag_hybrid | 30.7 [23.3, 38.0] | 28.7 [21.3, 36.0] | 24.0 [17.3, 31.3] |
| agent | 21.3 [15.3, 28.0] | 16.0 [10.7, 22.0] | 9.3 [5.3, 14.0] |
| oracle | 49.3 [42.0, 58.0] | 50.0 [42.0, 58.7] | 30.0 [22.7, 38.0] |

![accuracy](reports/figures/accuracy.png)

Everything here is one run on 150 questions. The intervals cover the choice of questions, not run-to-run variation: the first qwen2.5:3b agent run, kept in `results/agent_tool_example/` and differing mainly in the calculator's description, changed 119 of 150 replies and flipped 19 verdicts, though its accuracy moved by under a point. A paired interval of about ±8 points is the resolution, so differences of 4 or 5 points between configs are not findings.

- With the dense or hybrid retriever every model gets 13 to 20 points more questions right than it does on its own, roughly two to three times as many, and nearly all of that is on the 62 numeric questions, where closed book gets almost nothing. For qwen2.5:3b, RAG with the hybrid retriever is 16.0 points above closed book [+6.7, +25.3], paired over the same questions.
- Even handed the right pages, the 3B models get half the questions wrong, so better retrieval on its own has a ceiling. The oracle is 18.7 points above hybrid RAG for qwen2.5:3b [+9.3, +28.7]. Of the 104 questions qwen2.5:3b got wrong with hybrid RAG, 63 never had a gold page in front of the model, and 41 had at least one (12 of those needed two or three pages and were not shown all of them).
- The agent is clearly worse than plain RAG with the same retriever for llama3.2:3b (-12.7 [-20.7, -5.3]) and qwen2.5:1.5b (-14.7 [-22.0, -8.0]), and borderline worse for qwen2.5:3b (-9.3 [-17.3, -1.3]; on the strict grade -8.0 [-16.0, +0.7]). For the first two most of the loss is questions they never searched (47 and 143 of the 150). qwen2.5:3b searched in 141, and its loss is mostly not retrieval (see The agent).
- Small models copy examples. llama3.2:3b called the calculator with the numbers from the worked example in its description in 32 of its 150 questions, and an example figure in the system prompt came back as a whole answer in a smoke test. The prompts now use placeholders, and those get copied too (see Answering).
- The judge agrees with the automatic grade on 93.5% [90.9, 95.8] of the 1,116 numeric answers where both apply (Cohen's kappa 0.81). That is the only check on it here: nothing in this repo measures how it does on the text answers it grades alone.

## Reproduce

```
pip install -r requirements.txt
pip install -e .
python scripts/01_download.py        # questions and the 84 filings from the FinanceBench repo
python scripts/02_ingest.py          # pages and chunks, and the page-number check
python scripts/03_index.py           # bge-small embeddings, about 45 minutes on a CPU
python scripts/04_retrieval_eval.py
ollama pull qwen2.5:3b               # and llama3.2:3b, qwen2.5:1.5b, and qwen2.5:7b for the judge
python scripts/05_answer.py --config rag_hybrid --model qwen2.5:3b   # once per config and model
python scripts/06_score.py           # grading, and the judge
python scripts/07_report.py          # figures
python scripts/08_agent_report.py    # what the agent did
python -m pytest
```

The tables and intervals below are copied from `reports/`: `page_alignment.md` and `retrieval.md`, which `02_ingest.py` and `04_retrieval_eval.py` write from the parsed filings and the retrieval runs, and `accuracy.md` and `agent.md`, which `06_score.py` and `08_agent_report.py` write from the answers and verdicts in `results/`. If a number here disagrees with a report, the report is right. Some counts in the text that are in no report (the gold-page split, the placeholder replies, the longest oracle prompt and so on) I took straight from `results/` and the raw data. Under Speed, the 2.9GB and the 92% on the card are what Ollama's `/api/ps` reported with llama3.2:3b loaded, and the layer split and slowdown come from its server logs. Generation uses temperature 0 and a fixed seed, but that does not make it exactly repeatable: for question 00746, rag_dense and rag_hybrid retrieved the same five chunks, so qwen2.5:3b got the same prompt twice, and its two replies differ. `05_answer.py` picks up where it stopped, so a run can be killed and restarted.

My GPU driver turned out to be too old for the CUDA build Ollama ships: every model load crashed with "the provided PTX was compiled with an unsupported toolchain". Rather than update the driver mid-project, I ran a second Ollama server with CUDA hidden and the Vulkan library selected (`CUDA_VISIBLE_DEVICES=-1 OLLAMA_LLM_LIBRARY=vulkan`), so it runs on Ollama's Vulkan backend on the same card, and pointed the scripts at it with `OLLAMA_HOST`. The judge ran on the CPU through a third server (`OLLAMA_VULKAN=0 OLLAMA_LLM_LIBRARY=cpu`) at the same time, which slowed llama3.2:3b down (see Speed).

## The data

The 150 questions come in three kinds of 50:

- metrics-generated: a figure worked out from the financial statements. FinanceBench tags 14 of them as information extraction, a single line item like "What is the FY2018 capital expenditure amount (in USD millions) for 3M? Give a response to the question by relying on the details shown in the cash flow statement." The other 36 need arithmetic, mostly ratios and margins, some averaged over three years, and 23 of them use two statements.
- domain-relevant: what an analyst would ask about any company. "Is 3M a capital-intensive business based on FY2022 data?"
- novel-generated: written by the annotators for that filing. "If we exclude the impact of M&A, which segment has dragged down 3M's overall growth in 2022?"

The 84 filings are 64 10-Ks, 8 10-Qs, 6 8-Ks and 6 earnings releases, from 2015 to 2024. I parse them with pymupdf: 12,013 pages, 65 of them with no text layer, none of them an evidence page. 29 of those are in Walmart's 2019 10-K, where an exhibit at the end is stored as page images; its text is on the page after each image in white, so the parse still has it.

Retrieval is scored against FinanceBench's evidence page numbers, so before using them I checked them. Each evidence item comes with the text of its page as FinanceBench extracted it, and I compared that with my parse at the stated page and one either side. All 189 evidence items match at the stated page with no offset, with a character 5-gram overlap of 1.00 on every one; the best score on a neighbouring page is 0.54. Word overlap would not have given one clean cutoff: on some statement pages FinanceBench's text has the spaces dropped ("Cashandcashequivalents"), and the right page scores as low as 0.35 on words, while a neighbouring page elsewhere reaches 0.63. 115 questions have one gold page, 33 have two and 2 have three.

Chunks are 200-word windows with a 40-word overlap that never cross a page, so every chunk has exactly one page number: 41,525 chunks, 177 words on average, between 7 and 1,565 per filing (median 447).

## Retrieval

- BM25, written out by hand with numpy and scipy.sparse (k1 = 1.5, b = 0.75, the Lucene form of the idf so it never goes negative). The tokeniser lowercases, drops thousands separators so "1,577" matches "1577", turns "FY2018" and "FY18" into "2018" because the filings almost never say FY, and drops a short stopword list.
- Dense: bge-small-en-v1.5 (384 dimensions) through sentence-transformers, with its query prefix. Embedding the 41,525 chunks took 46 minutes on the CPU.
- Hybrid: reciprocal rank fusion of the two rankings, with the usual constant of 60.

Each question searches only its own filing, as in the paper's setting with a separate store for each filing, and BM25 takes its statistics from that filing alone. A hit at k means at least one of the top k chunks is on a gold page. Intervals are 95% bootstraps over questions, and every comparison is paired, using the same resampled questions for both methods.

| method | recall@1 | recall@3 | recall@5 | recall@10 | MRR |
|:--|:--|:--|:--|:--|:--|
| bm25 | 0.167 [0.107, 0.227] | 0.307 [0.233, 0.380] | 0.347 [0.273, 0.427] | 0.473 [0.393, 0.553] | 0.269 [0.211, 0.328] |
| dense | 0.333 [0.260, 0.407] | 0.480 [0.407, 0.560] | 0.540 [0.467, 0.620] | 0.653 [0.580, 0.727] | 0.439 [0.376, 0.505] |
| hybrid | 0.260 [0.193, 0.333] | 0.473 [0.393, 0.553] | 0.527 [0.440, 0.607] | 0.667 [0.587, 0.740] | 0.398 [0.337, 0.461] |
| random order | 0.020 | 0.055 | 0.084 | 0.142 | |

- Dense retrieval is the clear winner over BM25: +19 points of recall@5 [+10, +29] and +0.17 MRR [+0.10, +0.24].
- Fusing in BM25 does not help. Against dense alone, hybrid is 7 points worse at rank 1 [-14, -1] and no different at 3, 5 or 10 (recall@5 -1 point [-9, +5]).
- The gap is all in the metrics-generated questions. On the 50 metrics-generated questions BM25 finds a gold page in its top 5 for 18% of them and dense for 76%. The questions use an analyst's words and the statements use their own: one asks for Amcor's "net AR", and the balance sheet says "Trade receivables, net". BM25 first reaches that page at rank 56, dense at rank 1. On the novel-generated questions, which tend to name the thing they ask about, BM25 is ahead (60% against 54%, within the noise).
- Questions that need two or three pages are where everything struggles. Of the 35, BM25 never gets all of their gold pages into its top 10, and dense manages it for 34%.

With every filing in one pool (the paper's shared store) the right filing comes back in the top 5 for 80% of questions with dense retrieval and 55% with BM25, and a gold page for 37% and 15%. Most of the loss from pooling is the same company's other years: dense finds the right company in its top 5 for 94% of questions. In the pool the hybrid is worse than dense alone at every k past 1 (gold page@5 -13 points [-19, -7]), because BM25 on its own is so poor there. The answering experiments use the single-filing setting.

![recall at k](reports/figures/recall_at_k.png)

## Answering

Every config uses the same system prompt (answer from the filing, show the working briefly, end with one line `Answer: <value with unit>`), an 8,192-token window, temperature 0 and seed 0.

- closed_book: the question, the company and the name of the filing.
- rag_bm25, rag_dense, rag_hybrid: the top 5 chunks from that retriever, labelled with their page numbers, then the question. Five chunks come to about 920 words.
- oracle: the full text of the gold evidence pages. The code cuts them to fit the window if it has to, but none came close: for qwen2.5:3b the longest oracle prompt was 3,122 tokens. This is the only config that ever sees which pages are gold.
- agent: the question and two tools, through Ollama's native tool calling (below).

The agent's `search(query)` returns the top 5 hybrid chunks of this filing for whatever query the model writes, and `calculate(expression)` does arithmetic by walking the expression's syntax tree, so nothing the model writes is ever run as Python. Its prompt carries the tool descriptions and a line asking it to use them, and the passages come back as tool results after the question. The model gets up to five calls with the tools on offer, and its first reply without a tool call is the answer. If all five end in tool calls, or a reply is empty or an unreadable tool call, one more call without the tools asks for the final answer; in the runs in the tables no question took more than four calls. Tool calls that a small model writes into its reply as JSON instead of through the API are picked up too. If the conversation would overflow the window, search results the model has already seen are dropped, oldest first; new ones are never dropped, only cut short if they still do not fit.

Models, all through Ollama with its default 4-bit quantisation: qwen2.5:3b is the main one, with llama3.2:3b and qwen2.5:1.5b for comparison.

Two things about prompting small models caught me out. The system prompt first showed the answer format with real figures, and in the smoke test qwen2.5:3b's whole reply to a question whose excerpts lacked the answer was the example itself, "Answer: $2.4 billion" (those answers are in `results/smoke_test/`). The examples became placeholders, `$X.X billion` and `X.X%`, before any full run. Those get copied too, but they cannot pass for a figure: 4 of the 2,700 final replies have the placeholder in their answer line, and none is graded correct. The yes/no template, "Answer: Yes, ..." or "Answer: No, ...", gets copied far more: 301 of the 2,700 replies have it, 202 of them llama3.2:3b's, mostly with a real reason after the dots. 20 of qwen2.5:1.5b's replies are the bare template with no reason; in all 20 the yes or no matched the reference and the judge counted them correct, which is worth 3 to 4 points of its RAG accuracy (without them 20.7, 19.3 and 20.7% with BM25, dense and hybrid).

The second one cost me a rerun of the agent. The calculator's description had a worked example, `(2453 - 2108) / 2108 * 100`, and llama3.2:3b called the tool with those numbers in 32 of its 150 questions, often without having searched, and gave the result as its answer. Those first runs are kept in `results/agent_tool_example/`, and the ones in the tables use a description with no figures in it. llama3.2:3b then copied the placeholder formula instead, `(new - old) / old * 100`, word for word in 14 questions and garbled in a 15th (qwen2.5:1.5b copied it once too). The calculator rejects that, so no number comes back from the tool, but in 7 of those 15 it was the only tool call, the model never searched, and its answer used figures it had not been shown. A test now checks that no prompt text the models see has a figure in it.

## Grading

62 of the 150 questions have a gold answer that is a single number (50 metrics-generated, 3 domain-relevant, 9 novel-generated). Those are graded automatically from the reply's `Answer:` line, or from its last non-empty line if it has none:

- correct if a number there is within 1% of the gold, or within half a unit of the gold's last decimal place, because some golds are rounded to one significant figure (a gold of 0.8 accepts 0.75 to 0.85). If the question asks for more decimal places than the gold has, half a unit of that place is used instead, and a whole-number gold gets only the 1%;
- a percentage matches the same fraction, so 25% and 0.25 count as equal;
- a bare number is read in the unit the question asks for, so "1,577" answers "in USD millions";
- a money figure off by a factor of a thousand, million or billion is wrong, and labelled a unit error;
- for capex and dividends, a minus sign copied from the cash flow statement ("Dividends paid (389)") does not count against the answer;
- two different figures of the same kind joined by "or" ("$389 million or $3.89 billion") are a hedge, and wrong however close one of them is.

One gold is corrected before grading: 05718 asks for American Water Works' FY2020 dividends in USD billions, the filing says 389 million, and the gold is "$0.40", which marks the filing's own figure wrong at two decimal places. It is graded against $0.39 (`GOLD_FIXES` in `fqa/score.py`).

The 1% is generous for ratios: it passes 31.0% for a gold of 30.8%, and 1.74 for 1.73. So the report also has a strict grade, where the answer and the gold must agree at the coarser of their two precisions (63.9 days still passes for 63.86; 63.59 does not). FinanceBench writes the money golds of the metrics questions to two decimals whatever the filing says ("$1616.00" for Amcor's $1,615.9 million, "$4.60" for PepsiCo's $4,625 million), so trailing zeros there are not counted as precision.

The other 88 questions go to qwen2.5:7b as a judge. It sees the question, the gold answer, the analyst's working where FinanceBench gives one (half of these 88 have none) and the reply, never which config or model wrote it. It answers two yes/no questions in JSON: does the reply commit to an answer, and does that answer agree with the reference. The verdict is worked out from those: refused if it does not commit, correct if it agrees, incorrect otherwise. The first version asked the judge for the verdict directly, and on the closed-book answers it would write "the answer does not commit to an answer" as its reason and then give a verdict of correct. Its verdicts are in `results/judge_v1/`.

To check the judge, I also had it mark every numerically graded answer, where the automatic grade is the reference. On those 1,116 answers it agrees 93.5% of the time [90.9, 95.8] (Cohen's kappa 0.81, correct against not). Its errors go both ways: of the 73 disagreements, 41 are answers the grader accepts and the judge rejects and 32 the reverse, not a significant difference. About half of the 41 are near misses that the strict grade also rejects, such as 31.0% for 30.8%, and most of the rest are format slips on the judge's side, such as rejecting "$5,818 million" against a gold of "$5818.00", or 80.13% against 0.8. It also accepts near misses the grader rejects, such as 0.69 for 0.68, and 9 of the 26 unit errors. I would expect it to do worse on the questions it grades alone, since fewer of them come with the analyst's working (56 of the 62 numeric ones have it) and a sentence is harder to compare than a number, but I have not measured that.

## Results

Most of these are in `reports/accuracy.md`, and the agent's in `reports/agent.md`. Intervals are 95% bootstraps over questions, and differences are paired: both configs scored on the same resampled questions.

Paired differences for qwen2.5:3b, in points of accuracy:

| comparison | difference [95% CI] |
|:--|:--|
| rag_bm25 - closed_book | +12.0 [+4.0, +19.3] |
| rag_dense - closed_book | +17.3 [+8.0, +26.7] |
| rag_hybrid - closed_book | +16.0 [+6.7, +25.3] |
| rag_dense - rag_hybrid | +1.3 [-5.3, +8.0] |
| agent - rag_hybrid | -9.3 [-17.3, -1.3] |
| oracle - rag_hybrid | +18.7 [+9.3, +28.7] |

**Retrieval.** Over all 150 questions the choice of retriever makes no clear difference to accuracy: dense and hybrid are 5 and 4 points above BM25 for qwen2.5:3b and 7 and 7 for llama3.2:3b, all with intervals that include zero, and qwen2.5:1.5b gains nothing though dense retrieval shows it a gold page more often (54% of questions against 35%). The one clear effect is where the retrieval eval predicted it, on the 50 metrics-generated questions: dense beats BM25 by 18 points for qwen2.5:3b [+6, +30] and 22 for llama3.2:3b [+8, +36].

**Reading.** With the gold pages pasted in, qwen2.5:3b gets 49.3% and llama3.2:3b 50.0%, and qwen2.5:1.5b 30.0% on the same pages. Part of that gap is behaviour, not just size: qwen2.5:1.5b ignores the instruction to show its working and answers 147 of its 150 oracle questions in one line. For the two Qwen models most of the oracle's wrong numeric answers are not near misses (24 of qwen2.5:3b's 35 are more than 25% away from the gold, and 42 of qwen2.5:1.5b's 49); llama3.2:3b's are closer. On the strict grade llama3.2:3b's oracle answers lose the most of any config and model (numeric accuracy goes from 51.6% to 33.9%), which leaves it no better on the numeric questions than its RAG configs; overall its oracle is still 15.3 points ahead of hybrid RAG [+7.3, +23.3]. For qwen2.5:3b it is rag_dense that loses most (37.1% to 25.8%), and qwen2.5:1.5b loses nothing anywhere.

**Closed book.** On the 62 numeric questions closed book is right 0 to 3 times out of 62 for every model; the models do not know 3M's 2018 capex. On the other 88 they get 14% to 22%.

**Why answers were wrong**, qwen2.5:3b (each wrong answer gets the first label that applies):

| config | wrong | unit error | retrieval miss | refusal | other | gold page shown % |
|:--|--:|--:|--:|--:|--:|--:|
| closed_book | 128 | 0 | - | 33 | 95 | |
| rag_bm25 | 110 | 1 | 82 | 2 | 25 | 34.7 |
| rag_dense | 102 | 0 | 59 | 0 | 43 | 54.0 |
| rag_hybrid | 104 | 2 | 63 | 2 | 37 | 52.7 |
| agent | 118 | 0 | 76 | 2 | 40 | 40.7 |
| oracle | 76 | 2 | - | 4 | 70 | |

A retrieval miss means no chunk the model was shown came from a gold page. "Other" is any other wrong answer: in the configs that show filing text, the model had a gold page in front of it and still got it wrong, and closed book is shown nothing, so its 95 are wrong from memory.

### The agent

`reports/agent.md` has the details. The search tool is the same hybrid retriever with the same k as rag_hybrid, so a search differs from rag_hybrid's only in its query. What the model adds is the choice of whether and how often to search, and on a question where it never searches it sees no filing text.

| | qwen2.5:3b | llama3.2:3b | qwen2.5:1.5b |
|:--|:--|:--|:--|
| questions with no search | 9 | 47 | 143 |
| one search | 125 | 97 | 7 |
| two or more | 16 | 6 | 0 |
| words in the first query (questions average 27) | 5.4 | 7.1 | 6.0 |
| gold page shown, agent | 40.7% | 32.7% | 1.3% |
| gold page shown, rag_hybrid | 52.7% | 52.7% | 52.7% |
| difference, points [95% CI] | -12.0 [-22.0, -3.3] | -20.0 [-28.7, -12.0] | -51.3 [-59.3, -43.3] |
| the same, on the questions it searched | -9.2 [-19.9, +0.7] | +2.9 [-2.9, +8.7] | -28.6 [-57.1, +0.0] |
| right on the questions it did not search, agent vs rag_hybrid | 11.1% vs 55.6% | 0.0% vs 34.0% | 9.8% vs 25.2% |
| calculate calls (of which failed) | 10 (0) | 71 (71) | 3 (3) |

For llama3.2:3b and qwen2.5:1.5b most of the gap comes from questions they never searched, though the prompt told them to search. llama3.2:3b skipped the search in 47 questions and got none of them right, although on 70% of them the question itself would have found a gold page. When it did search, its queries found a gold page about as often as the question did. It also called the calculator 71 times, and every call failed: it wrote the formula in words, "(capex / revenue) * 100", rather than putting in numbers from the filing, and 14 of the calls were the description's placeholder word for word. qwen2.5:1.5b searched in only 7 of the 150 questions and answered the rest from memory, so as an agent it is close to a closed-book model. qwen2.5:3b searched in 141 questions with a short keyword query, 5.4 words on average ("FY2018 capital expenditure 3M"), and on those its searches showed it a gold page 9 points less often than the question did, though the interval just includes zero. That explains little of its accuracy gap: it lost 14 answers against rag_hybrid, and even on the questions where both showed it a gold page the agent got fewer right, so the agent's format costs it answers too. One guess worth testing: the agent gets the passages after the question, and the FinanceBench paper found context after the question hurt GPT-4-Turbo.

### Speed

On the 1050 Ti through Vulkan, qwen2.5:3b processes about 207 prompt tokens a second and generates about 32, so a RAG answer (about 1,700 to 2,000 prompt tokens and 100 generated) takes 10 to 13 seconds, closed book 3 seconds and the agent 15. qwen2.5:1.5b is about twice as fast on both counts, 4 to 5 seconds a RAG answer. llama3.2:3b generates at 21 to 22 tokens a second on the RAG configs, slower than qwen2.5:3b. With an 8,192-token context it needed 2.9GB, which would have fitted, but Ollama keeps 1GB of the card free by default, so it put 25 of the 29 layers (Ollama reported 92% of its memory on the GPU) on the card and ran the other four on the CPU, where they competed with the judge. In the few minutes the judge was idle, llama3.2:3b generated about 27 tokens a second on RAG-length prompts. The two Qwen models fitted on the card, and the judge cost them under 1% of their answering time. Prompt speed counts only the tokens Ollama processed, not the ones it reused from the previous request, which for qwen2.5:3b's agent is more than a third of its prompt. The full table is in `reports/accuracy.md`.

## What I would do next

- Retrieval for statement pages. Dense retrieval still misses a gold page on 12 of the 50 metrics questions, and on the 38 where it puts one in front of qwen2.5:3b, the model does as well as with the whole gold pages (18 right either way), so what is left there is recall. The chunks keep each line item next to its figures, but the later chunks of a page lose the statement's title and the years at the top of its columns. Putting those back at the start of every chunk is the first thing I would try.
- A bigger reader. The oracle ceiling at 50% for 3B models is the hard limit here, and a 7B model fits on this card only partly.
- Make the agent start where RAG does: run the question itself as the first search, and let the model refine from there. That would fix most of llama3.2:3b's and qwen2.5:1.5b's gap, which is skipped searches, but not most of qwen2.5:3b's.
- Check the judge by hand on a sample of the text questions.

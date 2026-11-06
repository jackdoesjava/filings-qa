"""Grading answers against the FinanceBench gold answers.

A question whose gold answer is a single number is graded by comparing numbers
(grade_numeric). Everything else goes to a local judge model with a strict
rubric (judge). The judge is shown the question, the gold answer and its
justification, and the answer. It never sees which config or model wrote it.
"""

import hashlib
import json
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

from .llm import chat

JUDGE = "qwen2.5:7b"
NO_RETRIEVAL = ("closed_book", "oracle")


class Num(NamedTuple):
    value: float  # as written, with its sign: "12.3%" -> 12.3, "$1.2bn" -> 1.2
    scale: float  # 1e9 for "$1.2bn", 1.0 when there is no scale word
    pct: bool
    money: bool
    decimals: int


_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?!\d)(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+")
_SCALE = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "mm": 1e6, "m": 1e6,
          "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12, "tn": 1e12}
_SCALE_WORD = re.compile(r"\s?(thousand|million|billion|trillion|bn|mn|mm|tn)s?\b", re.I)
_PERCENT = re.compile(r"\s?(%|percent\b|per cent\b|percentage points?\b|pp\b|ppts?\b)", re.I)
_BPS = re.compile(r"\s?(bps?|basis points?)\b", re.I)
_CURRENCY = re.compile(r"(US\$|USD|\$|€|£)\s?$", re.I)
_MONTH = re.compile(r"\b(january|february|march|april|may|june|july|august|september|october|november|december"
                    r"|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec)\.?\s+$", re.I)
_MINUS = ("-", "−", "–")  # hyphen, minus sign, en dash


def _numbers(text: str) -> list[Num]:
    out = []
    for m in _NUM.finditer(text):
        token, before, after = m.group(), text[:m.start()], text[m.end():]
        money = neg = paren = False
        # currency, an opening bracket and a minus sign can come in any order: -$5, $(5), (-5)
        while True:
            if c := _CURRENCY.search(before):
                money, before = True, before[:c.start()]
            elif before.endswith("("):
                paren, before = True, before[:-1]
            elif before.endswith(_MINUS) and not (before[-2:-1].isalnum() or before[-2:-1] == "%"):
                neg, before = True, before[:-1]
            else:
                break
        if re.search(r"\bnegative\s+$", before, re.I):
            neg = True
        # part of a label or a date rather than a quantity: FY2022, Q2, COVID-19, Jun'23, May 26
        if re.search(r"[\w.'’]$|\w[-–]$", before):
            continue
        if _MONTH.search(before) and token.isdigit() and int(token) <= 31:
            continue
        if re.match(r"[-–]\s?[A-Za-z]", after):  # 10-K, 3-year
            continue

        # a bracket round the bare figure makes it negative, and the unit can follow: (1,577) million
        if paren and after.startswith(")"):
            neg, after = True, after[1:]
        scale, pct, bps = 1.0, False, False
        if p := _PERCENT.match(after):
            pct, after = True, after[p.end():]
        elif b := _BPS.match(after):
            pct, bps, after = True, True, after[b.end():]
        elif g := re.match(r"[A-Za-z]+", after):
            word = g.group().lower()
            if word.endswith("s") and word[:-1] in _SCALE:
                word = word[:-1]
            # a lone k/m/b only counts after a currency sign or on a figure written with a
            # comma or a decimal point (1,577M, 8.74B), otherwise 3M is 3 million
            if word in _SCALE and (len(word) > 1 or money or "," in token or "." in token):
                scale = _SCALE[word]
            elif word != "x":  # 2.7x is fine; 3M, 1st, 2Q are not quantities
                continue
            after = after[g.end():]
        elif s := _SCALE_WORD.match(after):
            scale, after = _SCALE[s.group(1).lower()], after[s.end():]
        if after.lstrip().startswith("$"):
            money = True
        # so is (12.5%), but the bracket in ($5,466 million) only restates a figure
        if paren and after.startswith(")") and scale == 1:
            neg = True
        if not (money or pct) and scale == 1 and re.fullmatch(r"(19|20)\d\d", token):
            continue  # a year

        value = float(token.replace(",", ""))
        decimals = len(token.partition(".")[2])
        if bps:
            value, decimals = value / 100, decimals + 2
        out.append(Num(-value if neg else value, scale, pct, money, decimals))
    return out


def parse_numbers(text: str) -> list[float]:
    """Every quantity in the text with its scale applied: "$(1,234.5)" -> -1234.5,
    "1.2 billion" -> 1.2e9, "12.3%" -> 12.3. Years and labels such as FY2022 are skipped."""
    return [n.value * n.scale for n in _numbers(text)]


# only at the start of a line, so "To answer:" or "prior answer:" mid-reply is not taken for it
_ANSWER = re.compile(r"^[ \t*#>_-]*(?:final\s+)?answer\s*\**\s*:", re.I | re.M)


def extract_answer_line(text: str) -> str:
    text = text or ""
    hits = list(_ANSWER.finditer(text))
    if hits:
        text = text[hits[-1].end():]
        lines = [s.strip(" *_#`\t") for s in text.splitlines()]
        return next((s for s in lines if s), "")
    lines = [s.strip(" *_#`\t") for s in text.splitlines()]
    lines = [s for s in lines if s]
    return lines[-1] if lines else ""


def question_unit(question: str) -> float:
    """1e6 for "Answer in USD millions", 1.0 when the question names no unit."""
    m = re.search(r"\b(?:USD|in)\s+(thousand|million|billion)s?\b", question, re.I)
    return _SCALE[m.group(1).lower()] if m else 1.0


def asked_decimals(question: str) -> int:
    m = re.search(r"\bround\w*\s+(?:\w+\s+){0,2}?to\s+(one|two|three|\d)\s+decimal", question, re.I)
    if not m:
        return 0
    word = m.group(1).lower()
    return {"one": 1, "two": 2, "three": 3}.get(word) or int(word)


def _gold_numbers(gold: str) -> list[Num]:
    # a gold that opens with its figure is that figure, and the rest is a note:
    # "36%. The answer here assumes FY2023 refers to the 12 months ended ..."
    if re.match(r"\W*\d", gold):
        gold = re.split(r"\.\s", gold, maxsplit=1)[0]
    return _numbers(gold)


# a figure off by exactly one of these is the right number in the wrong unit
_UNIT_SLIPS = (1e3, 1e6, 1e9, 1e-3, 1e-6, 1e-9)
# cash flow statements print these as negatives in brackets, "Dividends paid (389)",
# but a measure that takes them off something else, such as free cash flow, keeps its sign
_OUTFLOW = re.compile(r"capital expenditure|capex|dividend", re.I)
_NET = re.compile(r"\bless\b|free cash flow|\bFCF\b|retention", re.I)


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * (1 + 1e-9)


_FIGURE = r"\(?-?\$?\s?\(?\d[\d,]*(?:\.\d+)?\)?(?:\s?(?:%|[A-Za-z]+))?"
_EITHER = re.compile(rf"({_FIGURE})\s*,?\s+or\s+({_FIGURE})", re.I)


def is_hedge(line: str) -> bool:
    """Two different figures of the same kind offered as alternatives, as in
    "$389 million or $3.89 billion". The filings' own "$210 million, or 12.5%"
    restates a change as a percentage and is not a hedge, and nor is the same
    figure in two units."""
    for m in _EITHER.finditer(line):
        a, b = _numbers(m.group(1)), _numbers(m.group(2))
        if a and b and a[0].pct == b[0].pct:
            x, y = a[0].value * a[0].scale, b[0].value * b[0].scale
            if not _close(x, y, 0.01 * max(abs(x), abs(y))):
                return True
    return False


def grade_numeric(pred: str, gold: str, rel_tol: float = 0.01, question: str = "", strict: bool = False) -> str:
    """'correct', 'unit_error' or 'incorrect'.

    pred can be the whole reply or just its answer line; only the answer line is
    read. A number there is correct if it is within rel_tol of a gold number, or
    within half a unit of the gold's last decimal place (the golds are rounded,
    some to a single significant figure, so 1% alone would fail a correct 1.93%
    against a gold of 1.9%). 25% and 0.25 count as equal, and a percentage never
    matches a money figure. A bare number is read in the unit the question asks
    for, so "1,577" answers "in USD millions". For capex and dividends the sign
    is ignored when the gold is positive, since the cash flow statement prints
    them as negatives. A money figure that is off by a factor of 1e3, 1e6 or 1e9
    is a unit error. Offering two figures, "$389 million or $3.89 billion", is
    wrong however close one of them is, as it is for the judge.

    strict=True also asks the answer and the gold to agree at the coarser of
    their two precisions, so 31.0% misses a gold of 30.8% even though it is
    within 1%, while 63.9 days still passes for 63.86. A money gold whose
    decimals are all zero counts to the whole unit, because FinanceBench pads
    every money gold to two places ("$1616.00" for a filing's $1,615.9 million).
    Anything strict accepts, the main rule accepts too.
    """
    unit = question_unit(question)
    asked = asked_decimals(question)
    outflow = bool(_OUTFLOW.search(question)) and not _NET.search(question)
    line = extract_answer_line(pred)
    if is_hedge(line):
        return "incorrect"
    preds = _numbers(line)
    verdict = "incorrect"
    for g in _gold_numbers(gold):
        scale = g.scale if (g.scale != 1 or g.pct) else unit
        target = g.value * scale
        decimals = max(g.decimals, asked)
        tol = max(rel_tol * abs(target), 0.5 * 10.0 ** -decimals * scale if decimals else 0.0)
        money = g.money or scale != 1
        # half a unit of the gold's last place, for strict
        gold_half = 0.5 * 10.0 ** -(0 if money and g.value == round(g.value) else g.decimals) * scale
        # a bare predicted number may be in the unit the question asked for, else the gold's
        bare_unit = 1.0 if g.pct else (unit if unit != 1 else scale)
        for p in preds:
            if p.pct and money:
                continue
            readings = [(p.value * p.scale, p.scale)] if p.scale != 1 else [(p.value, 1.0), (p.value * bare_unit, bare_unit)]
            for x, x_unit in readings:
                if outflow and target > 0:
                    x = abs(x)
                pred_half = 0.5 * 10.0 ** -p.decimals * x_unit
                # a percentage against a plain gold is only ever x / 100: 0.014% is not 0.014
                if p.pct and not g.pct:
                    pairs = [(x / 100, target, tol, gold_half, pred_half / 100)]
                else:
                    pairs = [(x, target, tol, gold_half, pred_half)]
                if g.pct and not p.pct:
                    pairs.append((x, target / 100, tol / 100, gold_half / 100, pred_half))
                for a, b, t, hg, hp in pairs:
                    if strict:
                        # agree at the coarser of the two precisions (63.9 is fine for a gold of
                        # 63.86, 63.6 is not), and never more loosely than the main rule
                        t = min(t, max(hg, hp))
                    if _close(a, b, t):
                        return "correct"
                    if money and any(_close(a, b * f, t * f) for f in _UNIT_SLIPS):
                        verdict = "unit_error"
    return verdict


_QUANTITY = re.compile(r"\bhow (?:much|many)\b|\bamount\b|\bquantity\b|\bwhat percent"
                       r"|\bwhat (?:is|was)\b[^?]*\bfor FY", re.I)
_YES_NO = re.compile(r"^\W*(?:yes|no)\b", re.I)


def is_numeric_gold(gold: str, question: str = "") -> bool:
    """True when the gold answer is essentially a single number, so it can be graded
    without a judge: exactly one non-zero quantity (years and labels such as FY2022
    don't count, nor does a note after a gold that opens with its figure), not a
    yes/no answer, and either the gold is nothing but that number or the question
    asks for a quantity. The second condition is what keeps out "Corporate. Its net
    revenue was -$473 million." as the answer to "which segment". Zero is left to
    the judge because a relative tolerance means nothing there."""
    nums = _gold_numbers(gold)
    if len(nums) != 1 or nums[0].value == 0 or _YES_NO.match(gold):
        return False
    return not re.search(r"[A-Za-z]", gold) or bool(_QUANTITY.search(question))


_REFUSAL = re.compile(
    r"\b(cannot|can't|can not|unable|not possible|insufficient|unknown|n/a"
    r"|not (?:provided|available|given|included|mentioned|disclosed|stated|specified|shown|found|present|enough)"
    r"|no (?:information|data|mention)"
    r"|(?:does|do)(?: not|n't) (?:contain|include|provide|mention|show|specify|state|disclose|have))\b", re.I)


def looks_like_refusal(text: str) -> bool:
    return bool(_REFUSAL.search(text or ""))


# The fixed part of the judge's prompt goes first, as a system message, so that
# Ollama can reuse it from one call to the next instead of processing it again.
# On the CPU that saves about six seconds a verdict.
#
# The judge answers two yes/no questions and the verdict is worked out from them
# (parse_verdict). An earlier version asked for the verdict itself, and on the
# closed-book answers qwen2.5:7b sometimes wrote a reason saying the answer did
# not commit to anything and then a verdict of "correct" anyway.
JUDGE_SYSTEM = """You are checking answers to questions about company filings (10-Ks, 10-Qs, 8-Ks and earnings releases). Each message gives a question, a reference answer written by a financial analyst, sometimes the analyst's working, and an answer to check.

Compare the answer with the reference answer only, not with what you know about the company. The answer was asked to end with a line starting "Answer:". Judge that final answer, and use any working above it only to understand what it means.

Answer two questions about it.

1. commits: does the answer commit to an answer to the question? It does not if it says the information is not available or not provided, that it cannot be worked out, or if it gives no answer at all. An answer that says something does not exist or does not apply has committed, if that is its answer.
2. matches: does that answer agree with the reference answer? Figures must match the reference allowing for rounding (about 1%); the same figure in the wrong unit or scale, such as millions for billions, does not match. For a yes/no question the yes or no must match and the reasons must not contradict the reference. For an explanation or a list it must contain the main points of the reference; extra points are fine as long as they do not contradict it. Partly right answers, answers that hedge between options and answers that give several conflicting figures do not match. An answer that does not commit does not match.

Reply with JSON only, in this form: {"reason": "<one short sentence>", "commits": true or false, "matches": true or false}"""


def judge_prompt(question: str, gold: str, justification: str, pred: str) -> str:
    working = f"\nThe analyst's working:\n{justification.strip()}\n" if justification.strip() else ""
    return f"""Question:
{question.strip()}

Reference answer, written by a financial analyst:
{gold.strip()}
{working}
Answer to check:
{pred.strip()[-4000:]}

Give your verdict as JSON."""


def _ask_json(model: str, prompt: str) -> str:
    messages = [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}]
    reply = chat(model, messages, num_ctx=4096, num_predict=256, format="json")
    return reply["message"]["content"]


def _flag(value) -> bool | None:
    if isinstance(value, str):
        value = {"true": True, "yes": True, "false": False, "no": False}.get(value.strip().lower())
    return value if isinstance(value, bool) else None


def parse_verdict(content: str) -> dict:
    """refused if the answer does not commit, correct if it matches, incorrect
    otherwise. A reply that is cut off or garbled counts against the answer."""
    try:
        out = json.loads(content)
        if not isinstance(out, dict):
            raise ValueError
    except ValueError:
        out = {k: m.group(1) for k in ("commits", "matches")
               if (m := re.search(rf'"{k}"\s*:\s*"?(true|false)', content, re.I))}
        out["reason"] = content
    commits, matches = _flag(out.get("commits")), _flag(out.get("matches"))
    reason = str(out.get("reason", "")).strip()
    if commits is None or matches is None:
        reason = f"unclear reply: {reason}"
    verdict = "refused" if commits is False else "correct" if matches is True else "incorrect"
    return {"verdict": verdict, "reason": reason[:500], "commits": commits, "matches": matches}


def judge(question: str, gold: str, justification: str, pred: str, model: str = JUDGE) -> dict:
    return parse_verdict(_ask_json(model, judge_prompt(question, gold, justification, pred)))


def answer_sha1(answer: str) -> str:
    return hashlib.sha1(answer.encode("utf-8")).hexdigest()


_KEY = ("id", "config", "model", "answer_sha1", "judge")


def load_judgements(path) -> dict:
    cache = {}
    path = Path(path)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    cache[tuple(rec[k] for k in _KEY)] = rec
    return cache


def judge_cached(q: dict, row: dict, cache: dict, path, model: str = JUDGE, call: bool = True) -> dict | None:
    """The judge's verdict on row["answer"], from the cache if this exact answer to this
    question from this config and model has been judged before. Only the question, gold
    and answer go to the judge. With call=False a missing verdict comes back as None."""
    key = (q["id"], row["config"], row["model"], answer_sha1(row["answer"]), model)
    if key not in cache:
        if not call:
            return None
        rec = dict(zip(_KEY, key)) | judge(q["question"], q["answer"], q["justification"], row["answer"], model)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        cache[key] = rec
    return cache[key]


# Gold answers that disagree with the filing, put right before grading. 05718 asks
# for American Water Works' FY2020 cash dividends in USD billions, and the cash flow
# statement shows "Dividends paid (389)" in millions. The gold, "$0.40", is that
# rounded to one decimal place and padded to two; read to two places, as every
# other gold is, it marks the filing's own figure wrong.
GOLD_FIXES = {"financebench_id_05718": "$0.39"}


def gold_answer(q: dict) -> str:
    return GOLD_FIXES.get(q["id"], q["answer"])


def retrieval_hit(q: dict, row: dict) -> bool | None:
    """Did any chunk shown to the model come from a gold evidence page? None for
    configs that do not retrieve."""
    if row["config"] in NO_RETRIEVAL:
        return None
    gold = set(q["gold_pages"])
    return any(r["doc"] == q["doc"] and r["page"] in gold for r in row.get("retrieved") or [])


def failure_type(verdict: str | None, hit: bool | None) -> str:
    """Why a wrong answer was wrong, first match wins. A unit error goes first because
    the model found the right figure, so retrieval was not the problem."""
    if verdict in (None, "correct"):
        return ""
    if verdict == "unit_error":
        return "unit error"
    if hit is False:
        return "retrieval miss"
    if verdict == "refused":
        return "refusal"
    return "other"


def grade_row(q: dict, row: dict, cache: dict | None = None, path=None, judge_model: str = JUDGE,
              call_judge: bool = True, validate: bool = False) -> dict:
    """One line of results/scores.jsonl. With cache=None no judge is used, so only
    numerically graded questions get a verdict. validate=True also asks the judge
    about numerically graded answers, to measure how far it agrees with the grader;
    verdicts already in the cache are used either way. The judge's view of a
    numerically graded answer never replaces the automatic grade."""
    q = q | {"answer": gold_answer(q)}
    answer = row.get("answer") or ""
    line = extract_answer_line(answer)
    numeric = is_numeric_gold(q["answer"], q["question"])
    verdict = judged = strict = None
    if numeric:
        verdict = grade_numeric(answer, q["answer"], question=q["question"])
        if verdict == "incorrect" and not _numbers(line) and looks_like_refusal(answer):
            verdict = "refused"
        strict = int(grade_numeric(answer, q["answer"], question=q["question"], strict=True) == "correct")
    if answer.strip() and cache is not None:
        judged = judge_cached(q, row, cache, path, judge_model, call=call_judge and (validate or not numeric))
    if not numeric:
        verdict = judged["verdict"] if judged else (None if answer.strip() else "incorrect")
    hit = retrieval_hit(q, row)
    calls = row.get("tool_calls")
    return {
        "id": q["id"], "config": row["config"], "model": row["model"], "qtype": q["qtype"],
        "method": "numeric" if numeric else "judge",
        "verdict": verdict,
        "correct": None if verdict is None else int(verdict == "correct"),
        "strict_correct": strict,  # numeric questions only: grade_numeric(strict=True)
        "judge_verdict": judged["verdict"] if judged else None,
        "judge_reason": judged["reason"] if judged else "",
        "answer_line": line,
        "retrieval_hit": hit,
        "failure": failure_type(verdict, hit),
        "prompt_tokens": row.get("prompt_tokens") or 0,
        "cached_tokens": row.get("cached_tokens") or 0,
        "output_tokens": row.get("output_tokens") or 0,
        "prompt_s": row.get("prompt_s") or 0.0,
        "gen_s": row.get("gen_s") or 0.0,
        "total_s": row.get("total_s") or 0.0,
        "steps": row.get("steps") or 0,
        "tool_calls": len(calls) if isinstance(calls, list) else int(calls or 0),
    }


def cohen_kappa(a, b) -> float:
    """Agreement between two binary raters beyond what chance would give."""
    a, b = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    observed = np.mean(a == b)
    chance = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((observed - chance) / (1 - chance)) if chance < 1 else float("nan")

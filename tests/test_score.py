from collections import Counter

import pytest

from fqa import score
from fqa.data import DATA, load_questions
from fqa.score import (cohen_kappa, extract_answer_line, failure_type, gold_answer, grade_numeric, grade_row,
                       is_numeric_gold, judge_cached, load_judgements, parse_numbers, parse_verdict, retrieval_hit)

needs_data = pytest.mark.skipif(not (DATA / "raw" / "financebench_open_source.jsonl").exists(),
                                reason="FinanceBench questions not downloaded")

Q_MILLIONS = "What is the FY2018 capital expenditure amount (in USD millions) for 3M?"
Q_PERCENT = "Answer in units of percents and round to one decimal place."
Q_TWO_DP = "Round your answer to two decimal places."
Q_BILLIONS = "What is the year end FY2018 net PPNE for 3M? Answer in USD billions."
Q_LIABILITIES = "What is Netflix's year end FY2017 total current liabilities (in USD millions)?"
Q_FCF = ("What is the FY2020 free cash flow (FCF) for General Mills? FCF here is defined as: "
         "(cash from operations - capex). Answer in USD millions.")


@pytest.mark.parametrize("text, want", [
    ("$(1,234.5)", [-1234.5]),
    ("12.3%", [12.3]),
    ("1.2 billion", [1.2e9]),
    ("USD 1577.00 million", [1.577e9]),
    ("$2,018mn", [2.018e9]),
    ("$ 1.6Bn", [1.6e9]),
    ("-$1561M", [-1.561e9]),
    ("0.01$ per share", [0.01]),
    ("negative 3.7 days", [-3.7]),
    ("(3.4% jump)", [3.4]),
    ("20 basis points", [0.2]),
    ("1.5%-2.0%", [1.5, 2.0]),
    # a bracket round the bare figure is a negative, and its unit can come after it
    ("$(1,577) million", [-1.577e9]),
    ("(1,577) million", [-1.577e9]),
    ("(12.5%)", [-12.5]),
    ("Total current liabilities ($5,466 million)", [5.466e9]),  # the bracket only restates it
    # suffixes written straight after the figure
    ("1pp", [1.0]),
    ("+1pp (from 8% to 9%)", [1.0, 8.0, 9.0]),
    ("20bps", [0.2]),
    ("50bp", [0.5]),
    ("1.9percent", [1.9]),
    ("1,577M", [1.577e9]),
    ("8.74B", [8.74e9]),
    # labels, dates and years are not quantities
    ("3M's FY2018 capex was $1,577 million", [1.577e9]),
    ("Form 10-K for Q2 2023, filed May 26, 2023", []),
    ("COVID-19 costs fell 3.7% in Jun'23", [3.7]),
])
def test_parse_numbers(text, want):
    assert parse_numbers(text) == pytest.approx(want)


def test_extract_answer_line():
    assert extract_answer_line("Capex is in the cash flow statement.\nAnswer: $1,577 million\n") == "$1,577 million"
    assert extract_answer_line("Answer: 3\nOn reflection:\n**Answer:** 4") == "4"
    assert extract_answer_line("Answer:\n\n$5 million") == "$5 million"
    assert extract_answer_line("Working.\n### Final Answer: $5 million") == "$5 million"
    assert extract_answer_line("no marker here\n\nlast line\n\n") == "last line"
    assert extract_answer_line("") == ""
    # "answer:" partway through a line is not the answer line
    assert extract_answer_line("To answer: we need the cash flow statement.\nCapex was $1,577 million.") == \
        "Capex was $1,577 million."
    assert extract_answer_line("Answer: $1,577 million\nExplanation of the answer: from the cash flows.") == \
        "$1,577 million"


@pytest.mark.parametrize("pred, gold, question, want", [
    ("Answer: $1,577 million", "$1577.00", Q_MILLIONS, "correct"),
    ("Answer: $1.577 billion", "$1577.00", Q_MILLIONS, "correct"),
    ("Answer: 1,577", "$1577.00", Q_MILLIONS, "correct"),  # read in the unit the question asks for
    ("Answer: $1,590 million", "$1577.00", Q_MILLIONS, "correct"),  # 0.8% off
    ("Answer: $1,600 million", "$1577.00", Q_MILLIONS, "incorrect"),  # 1.5% off
    ("Answer: $1,577 billion", "$1577.00", Q_MILLIONS, "unit_error"),
    ("Answer: $1,577 thousand", "$1577.00", Q_MILLIONS, "unit_error"),
    ("Answer: 1.577", "$1577.00", Q_MILLIONS, "unit_error"),
    ("Answer: 25%", "0.25", "", "correct"),
    ("Answer: 0.019", "1.9%", Q_PERCENT, "correct"),
    ("Answer: 1.93%", "1.9%", Q_PERCENT, "correct"),  # 1.6% off, but rounds to the gold
    ("Answer: 2.0%", "1.9%", Q_PERCENT, "incorrect"),
    ("Answer: 0.25%", "25%", "", "incorrect"),
    ("Answer: 660", "0.66", Q_TWO_DP, "incorrect"),  # a ratio, so not a unit error
    ("Answer: -1.53%", "-0.02", Q_TWO_DP, "correct"),
    ("Answer: 0.02", "-0.02", Q_TWO_DP, "incorrect"),
    ("Answer: 0.79", "0.8", Q_TWO_DP, "incorrect"),  # the gold is 0.80 to the two places asked for
    ("Answer: $1.9 billion in FY 2023", "$2,018mn", "", "incorrect"),  # 2023 is a year, not $2,023m
    ("Answer: The filing does not say", "$1577.00", Q_MILLIONS, "incorrect"),
    # a percentage against a plain gold is only ever x / 100, and never a money figure
    ("Answer: 1.42%", "0.01", Q_TWO_DP, "correct"),
    ("Answer: 0.014%", "0.01", Q_TWO_DP, "incorrect"),
    ("Answer: 8.7%", "$8.70", Q_BILLIONS, "incorrect"),
    ("Answer: $5,175 million, 5.4% of revenue", "$5409.00", Q_LIABILITIES, "incorrect"),  # not a unit error
    # capex as the cash flow statement prints it, but free cash flow keeps its sign
    ("Answer: -$1,577 million", "$1577.00", Q_MILLIONS, "correct"),
    ("Answer: $(1,577) million", "$1577.00", Q_MILLIONS, "correct"),
    ("Answer: (1,577)", "$1577.00", Q_MILLIONS, "correct"),
    ("Answer: -$3,215 million", "$3215.00", Q_FCF, "incorrect"),
    ("Answer: -$5,466 million", "$5466.00", Q_LIABILITIES, "incorrect"),
    ("Answer: Total current liabilities ($5,466 million)", "$5466.00", Q_LIABILITIES, "correct"),
    ("Answer: 20bps", "0.2%", Q_PERCENT, "correct"),
    ("Answer: +1pp (from 8% to 9%)", "Pepsico raised its guidance by 1 percentage point.", "", "correct"),
    ("Answer: 1,577M", "$1577.00", Q_MILLIONS, "correct"),
    ("Answer: 8.74B", "$8.70", Q_BILLIONS, "correct"),
    # only the figure a gold opens with counts, not the 12 in its note
    ("Answer: 12%", "36%. The answer here assumes FY2023 refers to the 12 months ended on January 28, 2023.", "",
     "incorrect"),
])
def test_grade_numeric(pred, gold, question, want):
    assert grade_numeric(pred, gold, question=question) == want


def test_grade_numeric_reads_only_the_answer_line():
    reply = "Capex was 1,373 in FY2017 and 1,577 in FY2018.\nAnswer: $1,373 million"
    assert grade_numeric(reply, "$1577.00", question=Q_MILLIONS) == "incorrect"


@pytest.mark.parametrize("gold, question, want", [
    ("$1577.00", Q_MILLIONS, True),
    ("$400,000,000 increase.", "By how much did Pepsico increase its credit agreement?", True),
    ("Corporate. Its net revenue was -$473 million.", "Which of JPM's segments had the lowest net revenue?", False),
    ("No. Microsoft decreased its debt by $2.5bn.", "Has Microsoft increased its debt?", False),
    ("From 24.6% in FY 2021 to 21.6% in FY 2022.", "How much has the effective tax rate changed?", False),
    ("0", "What is the quantity of restructuring costs?", False),
    ("36%. The answer here assumes FY2023 refers to the 12 months ended on January 28, 2023.",
     "What percent of Ulta Beauty's total spend on stock repurchases for FY 2023 occurred in Q4 of FY2023?", True),
    ("AMCOR's Adj. EBITDA was $2,018mn in FY 2023", "What Was AMCOR's Adjusted Non GAAP EBITDA for FY 2023", True),
    ("87% of the total restructuring liability is related Employee liabilities.",
     "What is the nature & purpose of AMCOR's restructuring liability as oF Q2 of FY2023 close?", False),
])
def test_is_numeric_gold(gold, question, want):
    assert is_numeric_gold(gold, question) is want


@needs_data
def test_every_metrics_gold_is_a_number():
    qs = [q for q in load_questions() if q["qtype"] == "metrics-generated"]
    assert len(qs) == 50
    for q in qs:
        assert parse_numbers(q["answer"]), q["answer"]
        assert is_numeric_gold(q["answer"], q["question"]), q["answer"]


@needs_data
def test_numeric_golds_grade_themselves_correct():
    qs = [q for q in load_questions() if is_numeric_gold(gold_answer(q), q["question"])]
    assert Counter(q["qtype"] for q in qs) == {"metrics-generated": 50, "domain-relevant": 3, "novel-generated": 9}
    for q in qs:
        gold = gold_answer(q)
        assert grade_numeric("Answer: " + gold, gold, question=q["question"]) == "correct", q["id"]


@pytest.mark.parametrize("content, want", [
    ('{"reason": "same figure", "verdict": "Correct"}', "correct"),
    ('{"reason": "misses one driver", "verdict": "partially correct"}', "incorrect"),
    ('{"verdict": "refused", "reason": "says it is not in the filing"}', "refused"),
    ('{"reason": "x", "verdict": "refused"', "refused"),  # cut off before the closing brace
    ("not json at all", "incorrect"),
])
def test_parse_verdict(content, want):
    assert parse_verdict(content)["verdict"] == want


TEXT_Q = {"id": "q1", "doc": "ACME_2022_10K", "question": "What drove the change in gross margin?",
          "answer": "Higher input costs.", "justification": "Page 41 says so.", "qtype": "domain-relevant",
          "gold_pages": [41]}
NUMBER_Q = {"id": "q2", "doc": "ACME_2022_10K", "question": Q_MILLIONS, "answer": "$1577.00", "justification": "",
            "qtype": "metrics-generated", "gold_pages": [59]}
ROW = {"config": "rag_hybrid", "model": "qwen2.5:3b", "answer": "Costs went up.\nAnswer: Higher input costs.",
       "retrieved": [{"doc": "ACME_2022_10K", "page": 12, "chunk": 3}], "prompt_tokens": 900, "output_tokens": 60,
       "prompt_s": 1.0, "gen_s": 2.0, "total_s": 3.1, "steps": 1, "tool_calls": []}


def fake_judge(monkeypatch, verdict="correct"):
    prompts = []

    def ask(model, prompt):
        prompts.append(prompt)
        return '{"reason": "fake", "verdict": "%s"}' % verdict

    monkeypatch.setattr(score, "_ask_json", ask)
    return prompts


def test_judge_is_blind_and_cached(tmp_path, monkeypatch):
    prompts = fake_judge(monkeypatch)
    path = tmp_path / "judgements.jsonl"
    assert judge_cached(TEXT_Q, ROW, load_judgements(path), path)["verdict"] == "correct"
    assert len(prompts) == 1
    assert "rag_hybrid" not in prompts[0] and "qwen" not in prompts[0]
    assert "Higher input costs." in prompts[0] and "Page 41 says so." in prompts[0]

    # read back from disk, the same answer is not judged twice
    assert judge_cached(TEXT_Q, ROW, load_judgements(path), path)["verdict"] == "correct"
    assert len(prompts) == 1
    # a different answer, or the same one from another config, is a new judgement
    judge_cached(TEXT_Q, ROW | {"answer": "Answer: Lower prices."}, load_judgements(path), path)
    judge_cached(TEXT_Q, ROW | {"config": "agent"}, load_judgements(path), path)
    assert len(prompts) == 3
    assert prompts[2] == prompts[0]


def test_grade_row_calls_judge_only_when_needed(tmp_path, monkeypatch):
    prompts = fake_judge(monkeypatch, verdict="incorrect")
    path = tmp_path / "judgements.jsonl"
    cache = {}
    number_row = ROW | {"answer": "Answer: $1,577 million"}

    s = grade_row(NUMBER_Q, number_row, cache, path)
    assert (s["method"], s["verdict"], s["correct"], s["judge_verdict"]) == ("numeric", "correct", 1, None)
    assert not prompts

    s = grade_row(NUMBER_Q, number_row, cache, path, validate=True)
    assert (s["verdict"], s["judge_verdict"]) == ("correct", "incorrect")  # the judge never overrides the grader
    assert len(prompts) == 1
    s = grade_row(NUMBER_Q, number_row, cache, path)  # once judged, the cached verdict is kept
    assert s["judge_verdict"] == "incorrect" and len(prompts) == 1

    s = grade_row(TEXT_Q, ROW, cache, path)
    assert (s["method"], s["verdict"], s["correct"]) == ("judge", "incorrect", 0)
    assert len(prompts) == 2

    s = grade_row(TEXT_Q, ROW | {"answer": "Answer: something new"}, cache, path, call_judge=False)
    assert s["verdict"] is None and s["correct"] is None
    assert len(prompts) == 2


def test_grade_row_failure_breakdown():
    s = grade_row(NUMBER_Q, ROW | {"answer": "Answer: $1,577 billion"})
    assert (s["verdict"], s["retrieval_hit"], s["failure"]) == ("unit_error", False, "unit error")

    s = grade_row(NUMBER_Q, ROW | {"answer": "The context does not include the cash flow statement.\nAnswer: N/A"})
    assert (s["verdict"], s["failure"]) == ("refused", "retrieval miss")

    s = grade_row(NUMBER_Q, ROW | {"config": "closed_book", "retrieved": [], "answer": "Answer: I cannot tell."})
    assert (s["verdict"], s["retrieval_hit"], s["failure"]) == ("refused", None, "refusal")

    hit = [{"doc": "ACME_2022_10K", "page": 59, "chunk": 80}]
    s = grade_row(NUMBER_Q, ROW | {"retrieved": hit, "answer": "Answer: $1,373 million"})
    assert (s["verdict"], s["retrieval_hit"], s["failure"]) == ("incorrect", True, "other")


def test_grade_row_grades_against_the_corrected_gold():
    q = {"id": "financebench_id_05718", "doc": "AMERICANWATERWORKS_2020_10K", "answer": "$0.40", "justification": "",
         "question": "How much (in USD billions) did American Water Works pay out in cash dividends for FY2020?",
         "qtype": "metrics-generated", "gold_pages": [85]}
    for answer, want in [("Answer: $389 million", "correct"),  # "Dividends paid (389)" in the filing
                         ("Answer: $0.39 billion", "correct"),
                         ("Answer: $(389) million", "correct"),
                         ("Answer: $353 million", "incorrect")]:  # FY2019
        assert grade_row(q, ROW | {"answer": answer})["verdict"] == want, answer


def test_retrieval_hit_needs_the_right_filing():
    q = {"doc": "A", "gold_pages": [3]}
    assert retrieval_hit(q, {"config": "agent", "retrieved": [{"doc": "B", "page": 3, "chunk": 0}]}) is False
    assert retrieval_hit(q, {"config": "agent", "retrieved": [{"doc": "A", "page": 3, "chunk": 9}]}) is True
    assert retrieval_hit(q, {"config": "agent", "retrieved": []}) is False
    assert retrieval_hit(q, {"config": "oracle", "retrieved": []}) is None


def test_failure_type_order():
    assert failure_type("correct", False) == ""
    assert failure_type(None, False) == ""
    assert failure_type("unit_error", False) == "unit error"
    assert failure_type("refused", False) == "retrieval miss"
    assert failure_type("refused", True) == "refusal"
    assert failure_type("incorrect", None) == "other"


def test_cohen_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    # agreement 3/4, chance 0.5 * 0.25 + 0.5 * 0.75 = 0.5, so kappa = (0.75 - 0.5) / (1 - 0.5)
    assert cohen_kappa([1, 1, 0, 0], [1, 0, 0, 0]) == pytest.approx(0.5)

import copy
import importlib.util
import json
from pathlib import Path

import pytest
import requests

import fqa.agent as fa
from fqa import llm
from fqa.agent import (DROPPED, FINAL_ASK, PROMPT_BUDGET, TOOLS, agent, calculate, calculator, closed_book,
                       n_tokens, oracle, rag, size, text_tool_calls)

DOC = "ACME_2020_10K"
Q = {"id": "q1", "company": "Acme", "doc": DOC, "question": "By how much did Acme's revenue grow in FY2020?",
     "answer": "20%", "justification": "1200 / 1000 - 1", "qtype": "metrics-generated", "reasoning": "",
     "gold_pages": [3], "evidence": ["Revenue was 1,200 in 2020."]}
CHUNKS = [
    {"doc": DOC, "page": 3, "chunk": 7, "text": "Revenue was 1,200 in 2020."},
    {"doc": DOC, "page": 4, "chunk": 9, "text": "Revenue was 1,000 in 2019."},
    {"doc": DOC, "page": 12, "chunk": 30, "text": "Capital expenditure was 150."},
]
FILLER = "Net sales were 1,234,567 thousand. "  # about 16 tokens by n_tokens


class FakeIndex:
    """Same search() as fqa.retrieve.Index, returning the chunks in list order."""

    def __init__(self, chunks=CHUNKS):
        self.chunks = chunks
        self.queries = []

    def search(self, doc, query, k=5, method="hybrid"):
        self.queries.append((doc, query, k, method))
        hits = [c for c in self.chunks if c["doc"] == doc][:k]
        return [{**c, "score": 1.0 / r, "rank": r} for r, c in enumerate(hits, start=1)]


class FakeChat:
    """Stands in for llm.chat: hands out scripted replies and keeps a copy of every request.
    A reply can be a function of (messages, tools)."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []

    def __call__(self, model, messages, tools=None, **options):
        self.requests.append({"model": model, "messages": copy.deepcopy(messages), "tools": tools, **options})
        reply = self.replies.pop(0)
        if callable(reply):
            reply = reply(messages, tools)
        return {"message": reply, "prompt_tokens": 100, "cached_tokens": 40, "output_tokens": 10, "prompt_s": 0.5,
                "gen_s": 0.25, "total_s": 1.0}


@pytest.fixture
def fake_chat(monkeypatch):
    def install(*replies):
        fake = FakeChat(replies)
        monkeypatch.setattr(fa, "chat", fake)
        return fake

    return install


def said(text):
    return {"role": "assistant", "content": text}


def called(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


def tool_messages(request):
    return [m for m in request["messages"] if m["role"] == "tool"]


# ---- calculator -------------------------------------------------------------


def test_calculate_does_arithmetic():
    assert calculate("1 + 2 * 3") == 7
    assert calculate("(10 - 4) / 4") == 1.5
    assert calculate("-3 ** 2") == -9
    assert calculate("2 ** 0.5") == pytest.approx(1.41421356)
    assert calculate("17 % 5") == 2
    assert calculate("-(1577 - 1373) / -1373") == pytest.approx(204 / 1373)


@pytest.mark.parametrize("expr", [
    "revenue * 2",
    "abs(-3)",
    "__import__('os').system('echo hi')",
    "(1).real",
    "[1, 2]",
    "'a' * 3",
    "True + 1",
    "1j * 2",
    "1 if 2 else 3",
    "1 < 2",
    "1 << 10",
    "7 // 2",
    "+5",
    "lambda: 1",
    "1,577 / 2",  # a tuple to Python; the tool wrapper strips the comma first
])
def test_calculate_rejects_everything_else(expr):
    with pytest.raises((ValueError, SyntaxError)):
        calculate(expr)


def test_calculator_tidies_numbers_and_reports_errors():
    assert calculator("$1,577 / 2") == "1577 / 2 = 788.5"
    assert calculator("(1,200 - 1,000) / 1,000 * 100") == "(1200 - 1000) / 1000 * 100 = 20"
    assert calculator("1.05^3") == "1.05**3 = 1.157625"
    assert calculator("revenue / 2").startswith("Error: Name is not allowed")
    assert calculator("open('x')").startswith("Error: Call is not allowed")
    assert calculator("1 / 0").startswith("Error")
    assert calculator("10 ** 10 ** 10").startswith("Error")  # overflows at once instead of hanging
    assert calculator("1e308 * 10") == "Error: the result is not a finite number."
    assert calculator("1 +" * 100 + "1") == "Error: expression too long."
    assert calculator("") == "calculate needs an expression."


@pytest.mark.parametrize("expr", ["(-8) ** (1/3)", "(-120 / 100) ** (1/3) - 1", "(-8)^(1/3)", "(-1) ** 0.5 % 2",
                                  "(-2) ** 0.5 * 0"])
def test_calculator_reports_a_negative_number_to_a_fractional_power(expr):
    # Python makes these complex numbers, which used to escape the error handling and stop the run
    assert calculator(expr) == "Error: a negative number to a fractional power is not a real number."


def test_calculator_powers_that_are_fine():
    assert calculator("(-8) ** 2") == "(-8) ** 2 = 64"
    assert calculator("(-2) ** -1") == "(-2) ** -1 = -0.5"
    growth = (1577 / 1373) ** (1 / 3) - 1
    assert calculator("(1577 / 1373) ** (1 / 3) - 1") == f"(1577 / 1373) ** (1 / 3) - 1 = {growth:.10g}"
    assert calculator("0 ** -1").startswith("Error")


# ---- tool calls written as text ----------------------------------------------


@pytest.mark.parametrize("text, calls", [
    ('{"name": "search", "arguments": {"query": "capex 2020"}}', [("search", {"query": "capex 2020"})]),
    ('<tool_call>\n{"name": "calculate", "arguments": {"expression": "1+1"}}\n</tool_call>',
     [("calculate", {"expression": "1+1"})]),
    ('Let me look.\n```json\n{"name": "search", "parameters": {"query": "revenue"}}\n```',
     [("search", {"query": "revenue"})]),
    ('{"name": "search", "arguments": "{\\"query\\": \\"revenue\\"}"}', [("search", {"query": "revenue"})]),
    ('{"name": "search", "arguments": "revenue 2020"}', [("search", {"query": "revenue 2020"})]),
    ('{"name": "search", "query": "revenue"}', [("search", {"query": "revenue"})]),
    ('{"type": "function", "function": {"name": "search", "arguments": {"query": "x"}}}', [("search", {"query": "x"})]),
    ('[{"name": "search", "arguments": {"query": "a"}}, {"name": "calculate", "arguments": {"expression": "2*3"}}]',
     [("search", {"query": "a"}), ("calculate", {"expression": "2*3"})]),
    ('Revenue grew from 1,000 to 1,200 {roughly}.\nAnswer: 20%', []),
    ('{"answer": "$1,200 million"}', []),
    ('{"name": "final_answer", "arguments": {"answer": "1200"}}', []),
    ('{"name": "search", "arguments": {"query": "unfinished"', []),
])
def test_text_tool_calls(text, calls):
    assert text_tool_calls(text) == calls


# ---- the agent loop ---------------------------------------------------------


def test_agent_searches_calculates_and_answers(fake_chat):
    fake = fake_chat(
        called("search", query="revenue 2020 and 2019"),
        said('<tool_call>\n{"name": "calculate", "arguments": {"expression": "(1,200 - 1,000) / 1,000 * 100"}}\n'
             '</tool_call>\nThe result is 25.'),
        said("Revenue went from 1,000 to 1,200.\nAnswer: 20%"),
    )
    index = FakeIndex()
    rec = agent(Q, "qwen2.5:3b", index, k=2)

    assert rec["answer"] == "Revenue went from 1,000 to 1,200.\nAnswer: 20%"
    assert (rec["id"], rec["config"], rec["model"], rec["steps"]) == ("q1", "agent", "qwen2.5:3b", 3)
    assert (rec["prompt_tokens"], rec["cached_tokens"], rec["output_tokens"], rec["prompt_s"], rec["gen_s"],
            rec["total_s"]) == (300, 120, 30, 1.5, 0.75, 3.0)
    assert index.queries == [(DOC, "revenue 2020 and 2019", 2, "hybrid")]
    assert rec["retrieved"] == [{"doc": DOC, "page": 3, "chunk": 7}, {"doc": DOC, "page": 4, "chunk": 9}]
    assert rec["tool_calls"] == [
        {"name": "search", "arguments": {"query": "revenue 2020 and 2019"}, "from_text": False, "result": [3, 4]},
        {"name": "calculate", "arguments": {"expression": "(1,200 - 1,000) / 1,000 * 100"}, "from_text": True,
         "result": "(1200 - 1000) / 1000 * 100 = 20"},
    ]
    assert all(r["tools"] == TOOLS for r in fake.requests)
    assert all((r["num_ctx"], r["num_predict"]) == (8192, 512) for r in fake.requests)

    history = fake.requests[-1]["messages"]
    assert [m["role"] for m in history] == ["system", "user", "assistant", "tool", "assistant", "tool"]
    assert history[3] == {"role": "tool", "tool_name": "search",
                          "content": "[page 4] Revenue was 1,200 in 2020.\n\n[page 5] Revenue was 1,000 in 2019."}
    # the call written as text goes back as a proper tool call, without the made-up result after it
    assert history[4] == {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "calculate", "arguments": {"expression": "(1,200 - 1,000) / 1,000 * 100"}}}]}
    assert history[5] == {"role": "tool", "tool_name": "calculate", "content": "(1200 - 1000) / 1000 * 100 = 20"}


def test_agent_keeps_going_when_every_call_comes_back_as_text(fake_chat):
    fake_chat(
        said('{"name": "search", "parameters": {"query": "capital expenditure"}}'),
        said('{"name": "calculate", "parameters": {"expression": "150 * 2"}}'),
        said("Twice the capex is 300.\nAnswer: $300"),
    )
    rec = agent(Q, "llama3.2:3b", FakeIndex())
    assert rec["steps"] == 3
    assert [(c["name"], c["from_text"]) for c in rec["tool_calls"]] == [("search", True), ("calculate", True)]
    assert rec["tool_calls"][1]["result"] == "150 * 2 = 300"
    assert rec["answer"].endswith("Answer: $300")


def test_agent_stops_at_max_steps_and_asks_once_without_tools(fake_chat):
    def keeps_searching(messages, tools):
        return called("search", query="revenue") if tools else said("Answer: $1,200")

    fake = fake_chat(*[keeps_searching] * 10)
    rec = agent(Q, "m", FakeIndex(), max_steps=3)

    assert len(fake.requests) == 4
    assert rec["steps"] == 4
    assert [r["tools"] is not None for r in fake.requests] == [True, True, True, False]
    assert fake.requests[-1]["messages"][-1] == {"role": "user", "content": FINAL_ASK}
    assert rec["answer"] == "Answer: $1,200"
    assert [c["name"] for c in rec["tool_calls"]] == ["search"] * 3
    # the same chunks are not pasted in a second time
    first, second, _ = tool_messages(fake.requests[-1])
    assert "Revenue was 1,200" in first["content"]
    assert second["content"].count("(already shown above)") == 3 and "Revenue" not in second["content"]
    assert len(rec["retrieved"]) == 3


def test_agent_asks_again_after_an_empty_reply(fake_chat):
    fake = fake_chat(said("  "), said("Answer: No, it fell."))
    rec = agent(Q, "m", FakeIndex())
    assert rec["answer"] == "Answer: No, it fell."
    assert rec["steps"] == 2
    assert fake.requests[1]["tools"] is None


def test_agent_answers_unknown_tools_and_missing_arguments(fake_chat):
    fake = fake_chat(
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_revenue", "arguments": {"year": 2020}}},
            {"function": {"name": "search", "arguments": {}}},
            {"function": {"name": "calculate", "arguments": {"expr": "2 * 3"}}},
        ]},
        said("Answer: 6"),
    )
    rec = agent(Q, "m", FakeIndex())
    assert [c["result"] for c in rec["tool_calls"]] == [
        "There is no tool called get_revenue. The tools are search and calculate.", [], "2 * 3 = 6"]
    assert [m["content"] for m in tool_messages(fake.requests[-1])][1] == "search needs a query."
    assert rec["retrieved"] == []


def test_agent_drops_old_search_results_to_stay_inside_the_window(fake_chat):
    chunks = [{"doc": DOC, "page": i, "chunk": i, "text": f"block {i} " + FILLER * 100} for i in range(6)]

    class Blocks(FakeIndex):
        # each search returns the next two chunks, and the fourth goes back to the first two
        def search(self, doc, query, k=5, method="hybrid"):
            start = 2 * (len(self.queries) % 3)
            self.queries.append(query)
            return [{**c, "score": 1.0, "rank": r} for r, c in enumerate(self.chunks[start:start + 2], start=1)]

    fake = fake_chat(*[called("search", query=f"query {i}") for i in range(4)], said("Answer: 1"))
    rec = agent(Q, "m", Blocks(chunks), k=2, max_steps=5)

    assert rec["steps"] == 5
    for r in fake.requests:
        assert size(r["messages"], TOOLS) <= PROMPT_BUDGET
    results = [m["content"] for m in tool_messages(fake.requests[-1])]
    assert results[0] == DROPPED
    assert "block 4" in results[2]
    # chunks 0 and 1 were dropped with the first result, so they come back in full, not as "already shown"
    assert "block 0" in results[3] and "already shown" not in results[3]
    assert [r["chunk"] for r in rec["retrieved"]] == [0, 1, 2, 3, 4, 5]


BLOCKS = [{"doc": DOC, "page": i, "chunk": i, "text": f"block {i} " + FILLER * 100} for i in range(6)]


class Scripted(FakeIndex):
    """Search number i returns the chunks listed at script[i]."""

    def __init__(self, script, chunks=BLOCKS):
        super().__init__(chunks)
        self.script = list(script)

    def search(self, doc, query, k=5, method="hybrid"):
        self.queries.append(query)
        return [{**self.chunks[i], "score": 1.0, "rank": r} for r, i in enumerate(self.script.pop(0), start=1)]


def stubs_have_their_text_above(request):
    above = ""
    for m in tool_messages(request):
        for line in m["content"].split("\n\n"):
            if line.endswith("(already shown above)") and line.split(" (")[0] + " block" not in above:
                return False
        above += m["content"]
    return True


def test_agent_keeps_the_text_of_a_chunk_found_again_when_the_first_result_goes(fake_chat):
    # the second search finds chunk 0 again; when the first result is dropped, chunk 0's text moves into it
    fake = fake_chat(*[called("search", query=f"query {i}") for i in range(3)], said("Answer: 1"))
    rec = agent(Q, "m", Scripted([[0, 1], [0, 2], [3, 4]]), k=2)

    first, second, third = [m["content"] for m in tool_messages(fake.requests[-1])]
    assert first == DROPPED
    assert "block 0" in second and "block 2" in second and "already shown" not in second
    assert "block 3" in third
    assert all(stubs_have_their_text_above(r) for r in fake.requests)
    assert all(size(r["messages"], TOOLS) <= PROMPT_BUDGET for r in fake.requests)
    assert [r["chunk"] for r in rec["retrieved"]] == [0, 1, 2, 3, 4]


def test_agent_cuts_new_results_that_do_not_fit_rather_than_dropping_them(fake_chat):
    # three searches in one reply bring back more than the window holds, none of it seen yet
    reply = {"role": "assistant", "content": "", "tool_calls": [
        {"function": {"name": "search", "arguments": {"query": f"query {i}"}}} for i in range(3)]}
    fake = fake_chat(reply, said("Answer: 1"))
    rec = agent(Q, "m", Scripted([[0, 1], [2, 3], [4, 5]]), k=2)

    request = fake.requests[1]
    assert size(request["messages"], TOOLS) <= PROMPT_BUDGET
    for i, text in enumerate(m["content"] for m in tool_messages(request)):
        a, b = text.split("\n\n")
        assert a.startswith(f"[page {2 * i + 1}] block {2 * i} ") and a.endswith(" ...")
        assert b.startswith(f"[page {2 * i + 2}] block {2 * i + 1} ") and b.endswith(" ...")
    # so every chunk it found was shown, if only in part
    assert [r["chunk"] for r in rec["retrieved"]] == [0, 1, 2, 3, 4, 5]


def test_agent_hands_calculator_errors_back_to_the_model(fake_chat):
    fake = fake_chat(called("calculate", expression="(-50 / 100) ** (1/3) - 1"), said("Answer: -20.6%"))
    rec = agent(Q, "m", FakeIndex())
    assert rec["answer"] == "Answer: -20.6%"
    assert tool_messages(fake.requests[-1])[0]["content"].startswith("Error: a negative number")


# ---- one-shot answerers -----------------------------------------------------


def test_closed_book_sends_only_the_question(fake_chat):
    fake = fake_chat(said("Answer: 20%"))
    rec = closed_book(Q, "m")
    assert fake.requests[0]["messages"] == [
        {"role": "system", "content": fa.SYSTEM},
        {"role": "user", "content": "Company: Acme\nFiling: ACME 2020 10K\n\n"
                                    "Question: By how much did Acme's revenue grow in FY2020?"},
    ]
    assert fake.requests[0]["tools"] is None
    assert (rec["config"], rec["steps"], rec["retrieved"], rec["tool_calls"]) == ("closed_book", 1, [], [])
    assert list(rec) == ["id", "config", "model", "answer", "retrieved", "tool_calls", "steps", "prompt_tokens",
                         "cached_tokens", "output_tokens", "prompt_s", "gen_s", "total_s"]


def test_rag_puts_the_top_k_chunks_before_the_question_in_rank_order(fake_chat):
    fake = fake_chat(said("Answer: 20%"))
    index = FakeIndex()
    rec = rag(Q, "m", index, "bm25", k=2)
    assert fake.requests[0]["messages"][1]["content"] == (
        "Company: Acme\nFiling: ACME 2020 10K\n\nExcerpts from the filing:\n\n"
        "[page 4] Revenue was 1,200 in 2020.\n\n[page 5] Revenue was 1,000 in 2019.\n\n"
        "Question: By how much did Acme's revenue grow in FY2020?")
    assert index.queries == [(DOC, Q["question"], 2, "bm25")]
    assert rec["config"] == "rag_bm25"
    assert rec["retrieved"] == [{"doc": DOC, "page": 3, "chunk": 7}, {"doc": DOC, "page": 4, "chunk": 9}]


def test_rag_leaves_out_the_lowest_ranked_chunks_that_do_not_fit(fake_chat):
    big = [{"doc": DOC, "page": i, "chunk": i, "text": FILLER * 200} for i in range(5)]
    fake = fake_chat(said("Answer: 1"))
    rec = rag(Q, "m", FakeIndex(big), "hybrid", k=5)
    assert [r["chunk"] for r in rec["retrieved"]] == [0, 1]
    assert size(fake.requests[0]["messages"]) <= PROMPT_BUDGET


def test_oracle_shows_the_gold_pages_and_nothing_else(fake_chat):
    pages = {DOC: {3: "Revenue was 1,200 in 2020.", 4: "Unrelated page.", 9: "Revenue was 1,000 in 2019.", 10: " "}}
    fake = fake_chat(said("Answer: 20%"))
    rec = oracle(dict(Q, gold_pages=[3, 9, 10]), "m", pages)
    user = fake.requests[0]["messages"][1]["content"]
    assert "Excerpts from the filing:\n\n[page 4] Revenue was 1,200 in 2020.\n\n[page 10] Revenue was 1,000 in 2019." \
           "\n\nQuestion:" in user
    assert "Unrelated" not in user
    # a blank page is not shown, so it is not listed either
    assert rec["retrieved"] == [{"doc": DOC, "page": 3, "chunk": None}, {"doc": DOC, "page": 9, "chunk": None}]
    assert rec["config"] == "oracle"


def test_oracle_cuts_the_long_page_and_keeps_the_short_one_whole(fake_chat):
    pages = {DOC: {3: FILLER * 1000, 9: "Revenue was 1,000 in 2019."}}
    fake = fake_chat(said("Answer: 20%"))
    oracle(dict(Q, gold_pages=[3, 9]), "m", pages)
    messages = fake.requests[0]["messages"]
    assert size(messages) <= PROMPT_BUDGET
    assert size(messages) > PROMPT_BUDGET - 100  # and it uses the room it has
    assert " ...\n\n[page 10] Revenue was 1,000 in 2019.\n\nQuestion:" in messages[1]["content"]


def test_no_gold_answer_or_evidence_in_any_non_oracle_prompt(fake_chat):
    q = dict(Q, answer="GOLD-ANSWER", justification="GOLD-WHY", evidence=["GOLD-EVIDENCE"], reasoning="GOLD-HOW")

    def search_once(messages, tools):
        searched = any(m["role"] == "tool" for m in messages)
        return called("search", query="revenue") if tools and not searched else said("Answer: 20%")

    fake = fake_chat(*[search_once] * 10)
    closed_book(q, "m")
    rag(q, "m", FakeIndex(), "bm25")
    agent(q, "m", FakeIndex())
    assert len(fake.requests) == 4
    for r in fake.requests:
        assert "GOLD" not in json.dumps(r["messages"]) + json.dumps(r["tools"])


def test_prompts_fit_the_window():
    assert n_tokens(fa.SYSTEM) < 150
    assert size(fa.conversation(fa.with_context(Q, []))) + n_tokens(json.dumps(TOOLS)) < 600
    assert PROMPT_BUDGET + fa.NUM_PREDICT < fa.NUM_CTX


# ---- llm.chat ---------------------------------------------------------------


class Response:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self.body = body
        self.text = text

    def json(self):
        return self.body


def test_chat_sends_one_non_streaming_request_and_converts_timings(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    sent = []
    body = {"message": {"role": "assistant", "content": "Answer: 1"}, "prompt_eval_count": 120,
            "prompt_eval_cached_count": 100, "eval_count": 8, "prompt_eval_duration": 1_500_000_000,
            "eval_duration": 400_000_000, "total_duration": 2_000_000_000}

    def post(url, json, timeout):
        sent.append((url, json))
        return Response(200, body)

    monkeypatch.setattr(llm.requests, "post", post)
    out = llm.chat("qwen2.5:3b", [{"role": "user", "content": "hi"}], tools=TOOLS)
    url, req = sent[0]
    assert url == "http://localhost:11434/api/chat"
    assert req["model"] == "qwen2.5:3b" and req["stream"] is False and req["tools"] == TOOLS
    assert req["options"] == {"num_ctx": 8192, "num_predict": 512, "temperature": 0.0, "seed": 0}
    assert out == {"message": body["message"], "prompt_tokens": 120, "cached_tokens": 100, "output_tokens": 8,
                   "prompt_s": 1.5, "gen_s": 0.4, "total_s": 2.0}

    llm.chat("qwen2.5:3b", [{"role": "user", "content": "hi"}])
    assert "tools" not in sent[1][1] and "format" not in sent[1][1]


@pytest.mark.parametrize("value, url", [
    ("127.0.0.1:11435", "http://127.0.0.1:11435/api/chat"),
    ("http://gpu-box:11434/", "http://gpu-box:11434/api/chat"),
])
def test_chat_goes_to_the_server_in_ollama_host(monkeypatch, value, url):
    sent = []
    monkeypatch.setenv("OLLAMA_HOST", value)
    monkeypatch.setattr(llm.requests, "post",
                        lambda url, json, timeout: sent.append((url, json)) or Response(200, {"message": {}}))
    llm.chat("m", [], format="json")
    assert sent[0][0] == url and sent[0][1]["format"] == "json"


def test_chat_retries_connection_errors_three_times_then_gives_up(monkeypatch):
    attempts = []

    def post(url, json, timeout):
        attempts.append(url)
        raise requests.ConnectionError("connection refused")

    monkeypatch.setattr(llm.requests, "post", post)
    monkeypatch.setattr(llm, "RETRY_WAIT", 0)
    with pytest.raises(requests.ConnectionError):
        llm.chat("m", [])
    assert len(attempts) == 4


def test_chat_gets_through_after_a_dropped_connection(monkeypatch):
    replies = [requests.ConnectionError("reset"), Response(200, {"message": {"role": "assistant", "content": "ok"}})]

    def post(url, json, timeout):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(llm.requests, "post", post)
    monkeypatch.setattr(llm, "RETRY_WAIT", 0)
    out = llm.chat("m", [])
    assert out["message"]["content"] == "ok"
    assert (out["prompt_tokens"], out["output_tokens"], out["total_s"]) == (0, 0, 0)


def test_chat_raises_with_ollamas_own_message(monkeypatch):
    monkeypatch.setattr(llm.requests, "post",
                        lambda url, json, timeout: Response(404, text='{"error":"model \'nope\' not found"}'))
    with pytest.raises(RuntimeError, match="not found"):
        llm.chat("nope", [])


# ---- scripts/05_answer.py ---------------------------------------------------


def load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "05_answer.py"
    spec = importlib.util.spec_from_file_location("answer_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_answer_script_names_its_files_and_picks_up_where_it_stopped(tmp_path):
    script = load_script()
    assert script.out_path("rag_hybrid", "qwen2.5:3b").name == "rag_hybrid__qwen2.5-3b.jsonl"

    questions = [dict(Q, id=f"q{i}") for i in range(4)]
    asked = []

    def answer(q):
        asked.append(q["id"])
        return {"id": q["id"], "answer": "Working.\nAnswer: 1", "steps": 1}

    path = tmp_path / "answers" / "closed_book__m.jsonl"
    script.run(questions[:2], answer, path)
    script.run(questions, answer, path)
    assert asked == ["q0", "q1", "q2", "q3"]
    written = [json.loads(line)["id"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert written == ["q0", "q1", "q2", "q3"]

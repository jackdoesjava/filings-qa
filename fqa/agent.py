"""The answerers. Each takes a question from fqa.data.load_questions() and
returns one record:

    id, config, model, answer, retrieved, tool_calls, steps,
    prompt_tokens, cached_tokens, output_tokens, prompt_s, gen_s, total_s

`answer` is the model's whole reply; scoring pulls the "Answer:" line out of
it. `retrieved` is every chunk whose text went to the model in a request, in
the order it first saw them (a passage cut short to fit the window counts).
`steps` is the number of model calls, and the token and time fields are summed
over those calls.

Only oracle() looks at the gold pages. The gold answer, justification and
evidence text never go into a prompt."""

import ast
import json
import math
import operator
import re

from .llm import chat

NUM_CTX = 8192
NUM_PREDICT = 512
# What the prompt may use: the window, less room for the reply and for the
# chat template's own text. Ollama cuts anything longer without an error.
PROMPT_BUDGET = NUM_CTX - NUM_PREDICT - 400

# The examples are placeholders, not figures: given real ones, the small models
# copied them out as the answer when the excerpts did not have it. The same went
# for the calculate tool's description, which once had a worked example in it:
# llama3.2:3b called the tool with that example's numbers in 34 of 150 questions.
# No prompt text here has a digit in it (tests/test_agent.py checks).
SYSTEM = (
    "You answer questions about a company's financial filing. Answer from the filing and show your "
    "working briefly. Finish with one line of the form \"Answer: <value with unit>\", for example "
    "\"Answer: $X.X billion\" or \"Answer: X.X%\". For a yes/no question write \"Answer: Yes, ...\" or "
    "\"Answer: No, ...\" with a short reason, and for any other question one short sentence."
)
AGENT_HINT = (
    "Use the search tool to find what you need in the filing and the calculate tool for any arithmetic. "
    "When you have the answer, reply without calling a tool."
)
FINAL_ASK = "No more tools. Using what you have found, give your final answer now, ending with the Answer line."
DROPPED = "(older search results removed to make room)"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search the filing. Returns the most relevant passages with their page numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for, e.g. a line item and a year"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Work out an arithmetic expression made of figures from the filing, for example "
            "(new - old) / old * 100 with the two figures written in. Numbers, + - * / ** % and brackets only.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "The expression to work out"}},
                "required": ["expression"],
            },
        },
    },
]
ARG = {"search": "query", "calculate": "expression"}

COUNTS = ("prompt_tokens", "cached_tokens", "output_tokens", "prompt_s", "gen_s", "total_s")

# Rough token count, used only to keep prompts inside the window. Qwen2.5 gives
# every digit a token of its own, so a page of financial tables costs far more
# than its length in words suggests. Here each digit is a token, and a run of
# letters or of punctuation is one token plus one per six characters: PDF text
# often runs words together ("Costofsales"), and table rows trail off in long
# runs of dots. Checked against Qwen2.5's tokenizer on the 189 FinanceBench
# evidence pages: median 12% high, at worst 8% low.
PIECES = re.compile(r" ?[A-Za-z]+|\d| ?[^\sA-Za-z\d]+|\s+")


def cost(piece):
    return 1 + len(piece.strip()) // 6


def n_tokens(text):
    return sum(cost(p) for p in PIECES.findall(text))


def truncate(text, n):
    used = 0
    for m in PIECES.finditer(text):
        used += cost(m.group())
        if used > n:
            return text[: m.start()].rstrip() + " ..."
    return text


def size(messages, tools=None):
    total = n_tokens(json.dumps(tools)) if tools else 0
    for m in messages:
        total += 5 + n_tokens(m["content"])
        if m.get("tool_calls"):
            total += n_tokens(json.dumps(m["tool_calls"]))
    return total


def fair_cap(sizes, budget):
    """Largest cap with sum(min(size, cap)) <= budget, so long pages get cut before short ones do."""
    sizes = sorted(sizes)
    for i, s in enumerate(sizes):
        cap = budget // (len(sizes) - i)
        if s > cap:
            return cap
        budget -= s
    return sizes[-1] if sizes else 0


# ---- prompts and records ----------------------------------------------------


def header(q):
    # FinanceBench tells the model which company and which filing it is about
    return f"Company: {q['company']}\nFiling: {q['doc'].replace('_', ' ')}"


def passage(page, text):
    # page numbers as a PDF viewer shows them; the records keep the 0-based index
    return f"[page {page + 1}] {text}"


def with_context(q, passages):
    context = "\n\n".join(passages) or "(none)"
    return f"{header(q)}\n\nExcerpts from the filing:\n\n{context}\n\nQuestion: {q['question']}"


def conversation(user):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def ref(c):
    return {"doc": c["doc"], "page": c["page"], "chunk": c["chunk"]}


def new_record(q, config, model):
    rec = {"id": q["id"], "config": config, "model": model, "answer": "", "retrieved": [], "tool_calls": [], "steps": 0}
    rec.update({key: 0 for key in COUNTS})
    return rec


def ask(rec, messages, tools=None):
    res = chat(rec["model"], messages, tools=tools, num_ctx=NUM_CTX, num_predict=NUM_PREDICT)
    rec["steps"] += 1
    for key in COUNTS:
        rec[key] += res[key]
    return res["message"]


def text_of(reply):
    return (reply.get("content") or "").strip()


def finish(rec, reply):
    rec["answer"] = text_of(reply)
    for key in ("prompt_s", "gen_s", "total_s"):
        rec[key] = round(rec[key], 3)
    return rec


# ---- the answerers ----------------------------------------------------------


def closed_book(q, model):
    rec = new_record(q, "closed_book", model)
    return finish(rec, ask(rec, conversation(f"{header(q)}\n\nQuestion: {q['question']}")))


def rag(q, model, index, method, k=5):
    rec = new_record(q, f"rag_{method}", model)
    room = PROMPT_BUDGET - size(conversation(with_context(q, [])))
    passages = []
    for c in index.search(q["doc"], q["question"], k, method):
        text = passage(c["page"], c["text"])
        room -= n_tokens(text) + 1
        if room < 0:
            break  # everything after this ranks lower anyway
        passages.append(text)
        rec["retrieved"].append(ref(c))
    return finish(rec, ask(rec, conversation(with_context(q, passages))))


def oracle(q, model, pages):
    """Upper bound: the gold evidence pages, whole, as the context. `pages` is {doc: {page: text}}."""
    rec = new_record(q, "oracle", model)
    doc = pages.get(q["doc"], {})
    gold = [p for p in q["gold_pages"] if doc.get(p, "").strip()]
    full = [passage(p, doc[p]) for p in gold]
    room = PROMPT_BUDGET - size(conversation(with_context(q, []))) - len(full)
    cap = fair_cap([n_tokens(t) for t in full], room)
    rec["retrieved"] = [{"doc": q["doc"], "page": p, "chunk": None} for p in gold]
    return finish(rec, ask(rec, conversation(with_context(q, [truncate(t, cap) for t in full]))))


def agent(q, model, index, method="hybrid", k=5, max_steps=5):
    rec = new_record(q, "agent" if method == "hybrid" else f"agent_{method}", model)
    messages = conversation(f"{header(q)}\n\nQuestion: {q['question']}\n\n{AGENT_HINT}")
    found = {}  # position in `messages` of each search result not yet dropped -> the chunks it found
    seen = 0  # how many messages the last request carried; any search result after them is new

    def search(query):
        hits = index.search(q["doc"], query, k, method)
        found[len(messages)] = hits  # where this result is about to go
        return [c["page"] for c in hits]

    def send(tools=None):
        nonlocal seen
        make_room(messages, found, seen)
        seen = len(messages)
        for c in in_full(found):
            if ref(c) not in rec["retrieved"]:
                rec["retrieved"].append(ref(c))
        return ask(rec, messages, tools)

    reply = None
    for _ in range(max_steps):
        msg = send(TOOLS)
        calls, from_text = calls_in(msg)
        if not calls:
            reply = msg
            break
        # Content stays empty: the chat templates show either the text or the
        # calls, and for a call written out as text, whatever followed it (often
        # a made-up result) should not stay in the history.
        messages.append({"role": "assistant", "content": "",
                         "tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in calls]})
        for name, args in calls:
            if name == "search":
                query = arg(args, "query")
                # make_room writes the passages in before the next request
                content, result = ("", search(query)) if query else ("search needs a query.", [])
            elif name == "calculate":
                content = result = calculator(arg(args, "expression"))
            else:
                content = result = f"There is no tool called {name}. The tools are search and calculate."
            messages.append({"role": "tool", "content": content, "tool_name": name})
            # for a search, `result` is the pages it hit; the text is in chunks.jsonl
            rec["tool_calls"].append({"name": name, "arguments": args, "from_text": from_text, "result": result})
    if reply is None or not text_of(reply):
        # out of steps, or an empty reply: one more call, with no tools on offer
        messages.append({"role": "user", "content": FINAL_ASK})
        reply = send()
    return finish(rec, reply)


# ---- agent plumbing ---------------------------------------------------------


def calls_in(msg):
    """The reply's tool calls as (name, arguments) pairs, and whether they had to be read out of the text."""
    calls = [(c["function"]["name"], c["function"].get("arguments") or {}) for c in msg.get("tool_calls") or []]
    if calls:
        return calls, False
    return text_tool_calls(msg.get("content") or ""), True


def text_tool_calls(text):
    """Tool calls written into the reply as JSON instead of coming back as tool_calls.

    Small models do this a lot, and if it went unnoticed the JSON would be
    taken as the final answer. Covers a bare {"name": ..., "arguments": ...},
    the same inside <tool_call> tags or a ```json fence, Llama's "parameters"
    spelling, arguments given as a JSON string, and the argument put next to
    the name. Only our two tool names count, so a JSON final answer stays an
    answer."""
    decoder = json.JSONDecoder()
    calls = []
    i = text.find("{")
    while i != -1:
        try:
            obj, end = decoder.raw_decode(text, i)
            call = as_call(obj)
        except ValueError:
            call = None
        if call:
            calls.append(call)
            i = text.find("{", end)
        else:
            i = text.find("{", i + 1)
    return calls


def as_call(obj):
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):
        obj = obj["function"]
    name = obj.get("name")
    if not isinstance(name, str) or name not in ARG:
        return None
    args = obj.get("arguments", obj.get("parameters"))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            pass
    if args is None:
        args = {k: v for k, v in obj.items() if k not in ("name", "type", "arguments", "parameters")}
    if not isinstance(args, dict):
        args = {ARG[name]: args}
    return name, args


def arg(args, key):
    if key not in args and len(args) == 1:
        # small models sometimes give the argument a different name
        key = next(iter(args))
    value = args.get(key)
    return "" if value is None else str(value).strip()


def first_found(found):
    # chunk -> the earliest search result still in `found` that has it, the one that shows its text
    return {c["chunk"]: at for at in sorted(found, reverse=True) for c in found[at]}


def in_full(found):
    """The chunks whose text the search results show, in the order they appear."""
    first = first_found(found)
    return [c for at in sorted(found) for c in found[at] if first[c["chunk"]] == at]


def write_results(messages, found, cap=None):
    """Write out every search result in `found`. A chunk's text goes in the earliest one
    that has it, and any later one says it is already shown above. With a cap, each
    passage is cut to that many tokens."""
    first = first_found(found)
    for at, hits in found.items():
        lines = []
        for c in hits:
            if first[c["chunk"]] != at:
                lines.append(f"[page {c['page'] + 1}] (already shown above)")
            elif cap is None:
                lines.append(passage(c["page"], c["text"]))
            else:
                lines.append(truncate(passage(c["page"], c["text"]), cap))
        messages[at]["content"] = "\n\n".join(lines) or "Nothing found."


def make_room(messages, found, seen):
    """Write out the search results so that the conversation fits the window.

    Results the model has seen (those before position `seen`) go first, oldest
    first, each swapped for a stub and taken out of `found`. A chunk whose text
    went with one moves to the next result that has it, so no "already shown
    above" is left pointing at text that has gone. Results the model has not
    seen yet are never dropped: if they still do not fit, their passages are
    cut, the longest first, as in oracle()."""
    write_results(messages, found)
    for at in sorted(found):
        if at >= seen or size(messages, TOOLS) <= PROMPT_BUDGET:
            break
        if n_tokens(messages[at]["content"]) > n_tokens(DROPPED):
            messages[at]["content"] = DROPPED
            del found[at]
            write_results(messages, found)
    if size(messages, TOOLS) > PROMPT_BUDGET:
        sizes = [n_tokens(passage(c["page"], c["text"])) for c in in_full(found)]
        write_results(messages, found, cap=0)  # each passage down to " ...", to measure everything else
        write_results(messages, found, fair_cap(sizes, PROMPT_BUDGET - size(messages, TOOLS)))


# ---- calculator -------------------------------------------------------------


def power(a, b):
    # Python makes this a complex number, e.g. (-8) ** (1/3), and the model
    # gets there with a sign slip in a growth rate: (end / start) ** (1 / n)
    if a < 0 and not b.is_integer():
        raise ArithmeticError("a negative number to a fractional power is not a real number")
    return a ** b


OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: power,
    ast.Mod: operator.mod,
}


def calculate(expression):
    """Numbers, + - * / ** %, brackets and unary minus, and nothing else: a
    name, call, attribute or any other node raises ValueError. The syntax tree
    is walked by hand, so nothing the model writes is ever run as Python.
    Numbers become floats, which makes 10 ** 10 ** 10 an OverflowError rather
    than a very long wait."""
    return _eval(ast.parse(expression, mode="eval").body)


def _eval(node):
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_eval(node.operand)
    what = node.op if isinstance(node, (ast.BinOp, ast.UnaryOp)) else node
    raise ValueError(f"{type(what).__name__} is not allowed")


def calculator(expression):
    """What the calculate tool sends back: the result, or what went wrong."""
    # numbers written the way the filing prints them ($1,577) and ^ for powers
    expr = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", expression.replace("$", "")).replace("^", "**").strip()
    if not expr:
        return "calculate needs an expression."
    if len(expr) > 200:
        return "Error: expression too long."
    try:
        value = calculate(expr)
    except (SyntaxError, ValueError) as e:
        return f"Error: {e}. Use only numbers, + - * / ** % and brackets."
    except ArithmeticError as e:  # dividing by zero, overflow, or power() above
        return f"Error: {e}."
    if not math.isfinite(value):
        return "Error: the result is not a finite number."
    return f"{expr} = {value:.10g}"

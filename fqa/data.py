import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"

RAW = DATA / "raw"
PDFS = RAW / "pdfs"
PROCESSED = DATA / "processed"
QUESTIONS = RAW / "financebench_open_source.jsonl"

# evidence_page_num + offset is the 0-based pymupdf page index. scripts/02_ingest.py
# checks this against the parsed text (reports/page_alignment.md): every evidence
# page matches at offset 0, in every filing, so nothing is shifted. A filing that
# ever needs its own offset goes in DOC_PAGE_OFFSET.
PAGE_OFFSET = 0
DOC_PAGE_OFFSET: dict[str, int] = {}


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def page_offset(doc: str) -> int:
    return DOC_PAGE_OFFSET.get(doc, PAGE_OFFSET)


def load_questions(path=QUESTIONS) -> list[dict]:
    questions = []
    for r in read_jsonl(path):
        doc = r["doc_name"]
        questions.append({
            "id": r["financebench_id"],
            "company": r["company"],
            "doc": doc,
            "question": r["question"],
            "answer": r["answer"],
            # null for 50 of the 150 questions, as is question_reasoning
            "justification": r["justification"] or "",
            "qtype": r["question_type"],
            "reasoning": r["question_reasoning"] or "",
            "gold_pages": sorted({e["evidence_page_num"] + page_offset(doc) for e in r["evidence"]}),
            "evidence": [e["evidence_text"] for e in r["evidence"]],
        })
    return questions

import json

import pymupdf
import pytest

import fqa.data
from fqa.data import load_questions, read_jsonl, write_jsonl
from fqa.ingest import build_chunks, char_overlap, chunk_page, offset_scores, read_pdf_pages, word_overlap


def words(n, prefix="w"):
    return " ".join(f"{prefix}{i}" for i in range(n))


def test_chunk_page_windows():
    chunks = [c.split() for c in chunk_page(words(450), size=200, overlap=40)]
    assert [len(c) for c in chunks] == [200, 200, 130]
    assert [c[0] for c in chunks] == ["w0", "w160", "w320"]
    for a, b in zip(chunks, chunks[1:]):
        assert a[-40:] == b[:40]
    assert chunks[-1][-1] == "w449"


@pytest.mark.parametrize("n, lengths", [(0, []), (1, [1]), (200, [200]), (201, [200, 41]), (360, [200, 200])])
def test_chunk_page_lengths(n, lengths):
    assert [len(c.split()) for c in chunk_page(words(n))] == lengths


def test_chunk_page_blank_text_and_bad_arguments():
    assert chunk_page(" \n\t ") == []
    with pytest.raises(ValueError):
        chunk_page("a b c", size=40, overlap=40)
    with pytest.raises(ValueError):
        chunk_page("a b c", size=40, overlap=-1)


def test_build_chunks_stay_on_their_page():
    pages = {"A": [words(250, "a"), "", words(10, "c")], "B": ["x y"]}
    chunks = build_chunks(pages, size=100, overlap=20)
    assert [(c["doc"], c["page"], c["chunk"]) for c in chunks] == [
        ("A", 0, 0), ("A", 0, 1), ("A", 0, 2), ("A", 2, 3), ("B", 0, 0)]
    for c in chunks:
        assert c["text"] in pages[c["doc"]][c["page"]]


def test_read_pdf_pages(tmp_path):
    path = tmp_path / "toy.pdf"
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Total  revenue\n\n1,577   million")
    doc.new_page()
    doc.new_page().insert_text((72, 72), "Net income")
    doc.save(str(path))
    doc.close()
    assert read_pdf_pages(path) == ["Total revenue 1,577 million", "", "Net income"]


def test_overlap_scores():
    ours = "Cash and cash equivalents $ 452,030 Total assets 1,211,362"
    assert word_overlap(ours, ours.upper()) == 1.0
    assert char_overlap(ours, ours.replace(" ", "\n")) == 1.0
    # how FinanceBench's text looks on some statement pages
    theirs = "Cashandcashequivalents $452,030 Totalassets 1,211,362"
    assert char_overlap(ours, theirs) == 1.0
    assert word_overlap(ours, theirs) < 0.5
    assert word_overlap(ours, "Net income 5,363") == 0.0
    assert word_overlap("", ours) == char_overlap("", "") == 0.0


def test_offset_scores_find_the_shifted_page():
    pages = [words(30, f"p{i}x") for i in range(5)]
    theirs = pages[3].replace(" ", "")
    scores = offset_scores(theirs, pages, 2)
    assert scores[1] == 1.0 and scores[0] < 0.5 and scores[-1] < 0.5
    assert offset_scores(theirs, pages, 4)[1] == 0.0
    assert offset_scores(pages[0], pages, 0, score=word_overlap) == {-1: 0.0, 0: 1.0, 1: 0.0}


def toy_question(qid, doc, pages, reasoning=None):
    return {"financebench_id": qid, "company": "3M", "doc_name": doc, "question_type": "metrics-generated",
            "question_reasoning": reasoning, "domain_question_num": None, "question": "What was capex?",
            "answer": "$1577.00", "justification": None, "dataset_subset_label": "OPEN_SOURCE",
            "evidence": [{"evidence_text": f"text on {p}", "doc_name": doc, "evidence_page_num": p,
                          "evidence_text_full_page": f"full page {p}"} for p in pages]}


def test_load_questions(tmp_path, monkeypatch):
    path = tmp_path / "questions.jsonl"
    rows = [toy_question("q1", "A_2018_10K", [59, 12, 59]), toy_question("q2", "B_2020_10K", [3], "Numerical reasoning")]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    q1, q2 = load_questions(path)
    assert q1 == {"id": "q1", "company": "3M", "doc": "A_2018_10K", "question": "What was capex?",
                  "answer": "$1577.00", "justification": "", "qtype": "metrics-generated", "reasoning": "",
                  "gold_pages": [12, 59], "evidence": ["text on 59", "text on 12", "text on 59"]}
    assert q2["reasoning"] == "Numerical reasoning" and q2["gold_pages"] == [3]

    monkeypatch.setattr(fqa.data, "DOC_PAGE_OFFSET", {"B_2020_10K": 1})
    assert [q["gold_pages"] for q in load_questions(path)] == [[12, 59], [4]]


def test_jsonl_round_trip_is_plain_ascii(tmp_path):
    rows = [{"doc": "A", "page": 0, "text": "Stockholders’ equity — €1,577"}]
    path = tmp_path / "processed" / "pages.jsonl"
    write_jsonl(path, rows)
    assert path.read_bytes().isascii() and b"\r" not in path.read_bytes()
    assert read_jsonl(path) == rows

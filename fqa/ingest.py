import re
from collections import Counter

import pymupdf


def read_pdf_pages(path) -> list[str]:
    with pymupdf.open(path) as doc:
        return [" ".join(page.get_text().split()) for page in doc]


def chunk_page(text: str, size: int = 200, overlap: int = 40) -> list[str]:
    if not 0 <= overlap < size:
        raise ValueError(f"need 0 <= overlap < size, got size={size}, overlap={overlap}")
    words = text.split()
    if not words:
        return []
    # a window only starts if it reaches past the end of the previous one
    starts = range(0, max(len(words) - overlap, 1), size - overlap)
    return [" ".join(words[i:i + size]) for i in starts]


def build_chunks(pages_by_doc: dict[str, list[str]], size: int = 200, overlap: int = 40) -> list[dict]:
    """pages_by_doc maps a filing to its page texts in page order. `chunk` counts
    up from 0 within each filing and no chunk crosses a page."""
    chunks = []
    for doc, pages in pages_by_doc.items():
        n = 0
        for page, text in enumerate(pages):
            for piece in chunk_page(text, size, overlap):
                chunks.append({"doc": doc, "page": page, "chunk": n, "text": piece})
                n += 1
    return chunks


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _jaccard(a: Counter, b: Counter) -> float:
    union = sum((a | b).values())
    return sum((a & b).values()) / union if union else 0.0


def word_overlap(a: str, b: str) -> float:
    """Weighted Jaccard between the word counts of two page texts."""
    return _jaccard(Counter(_words(a)), Counter(_words(b)))


def char_overlap(a: str, b: str, n: int = 5) -> float:
    """The same over character n-grams with spaces and punctuation removed, so it
    does not care where a PDF parser put the word breaks."""
    grams = []
    for text in (a, b):
        s = "".join(_words(text))
        grams.append(Counter(s[i:i + n] for i in range(len(s) - n + 1)))
    return _jaccard(*grams)


def offset_scores(full_page: str, pages: list[str], page_num: int, score=char_overlap,
                  offsets=(-1, 0, 1)) -> dict[int, float]:
    """score(full_page, our page at page_num + offset) for each offset, 0 when that page does not exist."""
    return {o: score(full_page, pages[page_num + o]) if 0 <= page_num + o < len(pages) else 0.0
            for o in offsets}

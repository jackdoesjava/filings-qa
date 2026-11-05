"""Downloads FinanceBench into data/raw: the 150 open-source questions, the
document metadata and the PDFs of the filings the questions use. Anything
already on disk is skipped, so it is safe to re-run."""

import time

import requests

from fqa.data import PDFS, QUESTIONS, RAW, read_jsonl

BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main"


def fetch(url, path, tries=3):
    if path.exists() and path.stat().st_size > 0:
        return False
    for attempt in range(1, tries + 1):
        try:
            r = requests.get(url, timeout=120)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == tries:
                raise
            print(f"  {e}, retrying")
            time.sleep(5 * attempt)
    if path.suffix == ".pdf" and not r.content.startswith(b"%PDF"):
        raise ValueError(f"{url} did not return a PDF")
    path.parent.mkdir(parents=True, exist_ok=True)
    # written under another name first so an interrupted download never looks like a finished one
    part = path.with_name(path.name + ".part")
    part.write_bytes(r.content)
    part.replace(path)
    print(f"downloaded {path.name} ({len(r.content) / 1e6:.1f} MB)")
    return True


got = 0
for name in ("financebench_open_source.jsonl", "financebench_document_information.jsonl"):
    got += fetch(f"{BASE}/data/{name}", RAW / name)

docs = sorted({r["doc_name"] for r in read_jsonl(QUESTIONS)})
for doc in docs:
    got += fetch(f"{BASE}/pdfs/{doc}.pdf", PDFS / f"{doc}.pdf")

print(f"{got} files downloaded, {len(docs) + 2 - got} already there; the questions use {len(docs)} filings")

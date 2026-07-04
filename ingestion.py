"""
Stage 1: Ingestion & Preprocessing (LangChain-native)
------------------------------------------------------
Reads every PDF in config.RAW_PDF_DIR and returns a list of LangChain
`Document` objects, one per page:

    Document(
        page_content = cleaned page text,
        metadata = {
            "filename": "policy.pdf",
            "pdf_id": "a1b2c3...",      # stable id derived from filename
            "page": 12,                  # 1-indexed page number
            "source_type": "native" | "ocr",
            "lang": "en",
        }
    )

Per-page text is cached to PROCESSED_DIR/<pdf_id>.json so that expensive
OCR does not re-run on every pipeline execution (cache invalidates
automatically if the PDF's mtime/size changes).

This module is imported by build_index.py. It can also be run directly:
    python -m src.ingestion
"""
import os
import re
import json
import time
import hashlib
from collections import Counter
from typing import List

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langdetect import detect, LangDetectException
from langchain_core.documents import Document

from src import config

if config.TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


def _pdf_id_for(filepath: str) -> str:
    return hashlib.md5(os.path.basename(filepath).encode()).hexdigest()[:10]


def _fingerprint(filepath: str) -> str:
    """Detect if a cached PDF has changed on disk (size + mtime)."""
    stat = os.stat(filepath)
    return f"{stat.st_size}-{int(stat.st_mtime)}"


def _extract_native_text(page: "fitz.Page") -> str:
    return page.get_text("text")


def _ocr_page(page: "fitz.Page") -> str:
    zoom = config.OCR_DPI / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img)


def _detect_repeated_lines(pages_text: List[str], threshold_ratio: float = 0.6) -> set:
    """Lines repeating across most pages -> likely headers/footers."""
    line_counts = Counter()
    for text in pages_text:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        candidates = lines[:2] + lines[-2:]
        for line in set(candidates):
            if 3 <= len(line) <= 90:
                line_counts[line] += 1
    n_pages = max(len(pages_text), 1)
    return {line for line, cnt in line_counts.items() if cnt / n_pages >= threshold_ratio}


def _clean_text(text: str, repeated_lines: set) -> str:
    lines = text.split("\n")
    kept = [l for l in lines if l.strip() not in repeated_lines]
    text = "\n".join(kept)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _safe_detect_lang(text: str) -> str:
    sample = text[:500].strip()
    if len(sample) < 10:
        return "unknown"
    try:
        return detect(sample)
    except LangDetectException:
        return "unknown"


def _process_pdf(filepath: str) -> dict:
    """Full extraction of one PDF -> cache-able dict."""
    filename = os.path.basename(filepath)
    pid = _pdf_id_for(filepath)
    doc = fitz.open(filepath)

    raw_pages, sources = [], []
    for page_num in range(len(doc)):
        page = doc[page_num]
        native = _extract_native_text(page)
        if len(native.strip()) >= config.MIN_NATIVE_CHARS:
            raw_pages.append(native)
            sources.append("native")
        else:
            raw_pages.append(_ocr_page(page))
            sources.append("ocr")

    repeated_lines = _detect_repeated_lines(raw_pages)

    pages = []
    for page_num, (text, source) in enumerate(zip(raw_pages, sources), start=1):
        cleaned = _clean_text(text, repeated_lines)
        pages.append({
            "page": page_num,
            "text": cleaned,
            "source_type": source,
            "lang": _safe_detect_lang(cleaned),
        })

    doc.close()
    return {
        "pdf_id": pid,
        "filename": filename,
        "fingerprint": _fingerprint(filepath),
        "num_pages": len(pages),
        "num_ocr_pages": sum(1 for p in pages if p["source_type"] == "ocr"),
        "pages": pages,
    }


def _load_or_process(filepath: str, verbose: bool = True) -> dict:
    pid = _pdf_id_for(filepath)
    cache_path = os.path.join(config.PROCESSED_DIR, f"{pid}.json")
    fp = _fingerprint(filepath)

    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if cached.get("fingerprint") == fp:
            if verbose:
                print(f"[cache hit] {os.path.basename(filepath)} "
                      f"({cached['num_pages']} pages, {cached['num_ocr_pages']} OCR)")
            return cached

    if verbose:
        print(f"[processing] {os.path.basename(filepath)} ...")
    t0 = time.time()
    result = _process_pdf(filepath)
    if verbose:
        print(f"  -> done in {time.time() - t0:.2f}s "
              f"({result['num_pages']} pages, {result['num_ocr_pages']} OCR)")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return result


def load_pdfs_as_documents(raw_dir: str = None, verbose: bool = True) -> List[Document]:
    """
    Main entry point. Ingests every PDF in raw_dir (default: config.RAW_PDF_DIR)
    and returns one LangChain Document per page.
    """
    raw_dir = raw_dir or config.RAW_PDF_DIR
    pdf_files = sorted(
        os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.lower().endswith(".pdf")
    )
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in {raw_dir}. Add PDFs there and re-run."
        )

    documents: List[Document] = []
    for filepath in pdf_files:
        result = _load_or_process(filepath, verbose=verbose)
        for page in result["pages"]:
            documents.append(Document(
                page_content=page["text"],
                metadata={
                    "filename": result["filename"],
                    "pdf_id": result["pdf_id"],
                    "page": page["page"],
                    "source_type": page["source_type"],
                    "lang": page["lang"],
                },
            ))

    if verbose:
        total_ocr = sum(1 for d in documents if d.metadata["source_type"] == "ocr")
        print(f"\nIngestion complete: {len(pdf_files)} PDFs, "
              f"{len(documents)} pages total, {total_ocr} OCR pages.")
    return documents


if __name__ == "__main__":
    docs = load_pdfs_as_documents()
    print(f"\nSample document metadata: {docs[0].metadata}")
    print(f"Sample content (first 200 chars): {docs[0].page_content[:200]}")
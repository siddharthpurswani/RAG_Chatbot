"""
Stage 2: Chunking & Metadata (LangChain-native)
-------------------------------------------------
Splits page-level Documents (from ingestion.py) into overlapping passage
chunks of ~config.CHUNK_SIZE_TOKENS tokens with config.CHUNK_OVERLAP_RATIO
overlap, using a tiktoken-based length function so sizes are measured in
actual LLM tokens rather than raw characters.

Each output chunk keeps the parent page's metadata (filename, pdf_id,
page, source_type, lang) and adds a unique chunk_id.

Run directly:
    python -m src.chunking
"""
import hashlib
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import config

try:
    import tiktoken
    _encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    # Falls back to a ~4-chars-per-token heuristic if tiktoken's encoding
    # file can't be downloaded (e.g. restricted/offline network). Slightly
    # less precise but keeps chunking fully functional.
    _encoding = None


def _token_len(text: str) -> int:
    if _encoding is not None:
        return len(_encoding.encode(text))
    return max(1, len(text) // 4)


def _make_chunk_id(filename: str, page: int, chunk_index: int, text: str) -> str:
    h = hashlib.md5(f"{filename}-{page}-{chunk_index}-{text[:50]}".encode()).hexdigest()[:12]
    return h


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE_TOKENS,
        chunk_overlap=config.CHUNK_OVERLAP_TOKENS,
        length_function=_token_len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def chunk_documents(page_documents: List[Document], verbose: bool = True) -> List[Document]:
    splitter = get_splitter()
    chunks = splitter.split_documents(page_documents)

    # add stable chunk_id + chunk_index (per source page) to metadata
    per_page_counter = {}
    for chunk in chunks:
        key = (chunk.metadata["filename"], chunk.metadata["page"])
        idx = per_page_counter.get(key, 0)
        per_page_counter[key] = idx + 1
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_id"] = _make_chunk_id(
            chunk.metadata["filename"], chunk.metadata["page"], idx, chunk.page_content
        )
        chunk.metadata["token_count"] = _token_len(chunk.page_content)

    if verbose:
        avg_tokens = sum(c.metadata["token_count"] for c in chunks) / max(len(chunks), 1)
        print(f"Chunking complete: {len(page_documents)} pages -> {len(chunks)} chunks "
              f"(avg {avg_tokens:.0f} tokens/chunk, target {config.CHUNK_SIZE_TOKENS})")
    return chunks


if __name__ == "__main__":
    from src.ingestion import load_pdfs_as_documents

    pages = load_pdfs_as_documents()
    chunks = chunk_documents(pages)
    print(f"\nSample chunk metadata: {chunks[0].metadata}")
    print(f"Sample chunk content:\n{chunks[0].page_content[:300]}")

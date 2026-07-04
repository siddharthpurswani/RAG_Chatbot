"""
Orchestrates Stages 1-4: Ingestion -> Chunking -> Embedding -> Persist to Chroma.

Run this ONCE (or whenever you add/change PDFs in data/raw_pdfs/):
    python -m src.build_index
"""
import time

from src.ingestion import load_pdfs_as_documents
from src.chunking import chunk_documents
from src.vectorstore import build_vectorstore


def main():
    t_start = time.time()

    print("=" * 60)
    print("STAGE 1: Ingestion & Preprocessing")
    print("=" * 60)
    pages = load_pdfs_as_documents()

    print("\n" + "=" * 60)
    print("STAGE 2: Chunking & Metadata")
    print("=" * 60)
    chunks = chunk_documents(pages)

    print("\n" + "=" * 60)
    print("STAGE 3+4: Embedding & Indexing")
    print("=" * 60)
    build_vectorstore(chunks)

    print("\n" + "=" * 60)
    print(f"INDEX BUILD COMPLETE in {time.time() - t_start:.1f}s")
    print(f"  Pages ingested : {len(pages)}")
    print(f"  Chunks indexed : {len(chunks)}")
    print("=" * 60)


if __name__ == "__main__":
    main()

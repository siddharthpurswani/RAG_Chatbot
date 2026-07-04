"""
Stage 3 (Embedding) + Stage 4 (Indexing & Retrieval) - LangChain-native
-------------------------------------------------------------------------
- Embedding model: BAAI/bge-small-en-v1.5 via HuggingFaceBgeEmbeddings
  (free, no login/API key required, runs locally on CPU).
- Vector DB: Chroma (free, open-source, persisted to disk, built-in HNSW
  ANN indexing under the hood).

Two entry points:
    build_vectorstore(chunks)  -> builds/persists a fresh Chroma index
    load_vectorstore()         -> loads an existing persisted index
    get_retriever(vs, k)       -> plain similarity-search retriever
"""
from typing import List

from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma

from src import config

_embeddings_singleton = None


def get_embeddings() -> HuggingFaceBgeEmbeddings:
    """
    BGE models expect a specific instruction prefix on the QUERY side
    (not on documents) for best retrieval quality. HuggingFaceBgeEmbeddings
    handles this automatically.
    """
    global _embeddings_singleton
    if _embeddings_singleton is None:
        _embeddings_singleton = HuggingFaceBgeEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
            query_instruction="Represent this sentence for searching relevant passages: ",
        )
    return _embeddings_singleton


def build_vectorstore(chunks: List[Document], verbose: bool = True) -> Chroma:
    """Embeds all chunks and persists them into a fresh Chroma collection."""
    embeddings = get_embeddings()
    if verbose:
        print(f"Embedding {len(chunks)} chunks with '{config.EMBEDDING_MODEL}' "
              f"and writing to Chroma at {config.CHROMA_DIR} ...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=config.CHROMA_DIR,
    )
    if verbose:
        print("Vector store built and persisted.")
    return vectorstore


def load_vectorstore() -> Chroma:
    """Loads an already-persisted Chroma collection (no re-embedding)."""
    embeddings = get_embeddings()
    return Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DIR,
    )


def get_retriever(vectorstore: Chroma, k: int = None):
    k = k or config.TOP_K_RETRIEVE
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})


if __name__ == "__main__":
    from src.ingestion import load_pdfs_as_documents
    from src.chunking import chunk_documents

    pages = load_pdfs_as_documents()
    chunks = chunk_documents(pages)
    vs = build_vectorstore(chunks)

    retriever = get_retriever(vs, k=3)
    results = retriever.invoke("What is RAG and why does chunking matter?")
    print(f"\nTop {len(results)} retrieved chunks:")
    for i, doc in enumerate(results, 1):
        print(f"\n[{i}] {doc.metadata['filename']} p.{doc.metadata['page']} "
              f"(source={doc.metadata['source_type']})")
        print(doc.page_content[:200])

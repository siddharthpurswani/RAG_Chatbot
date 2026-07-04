"""
Stage 5: Reranking (LangChain-native)
----------------------------------------
Wraps the base similarity retriever with a cross-encoder reranker.
The base retriever pulls TOP_K_RETRIEVE broad candidates (cheap, bi-encoder
similarity), then the cross-encoder re-scores each (query, chunk) pair
jointly (slower but far more accurate) and keeps only the best
TOP_K_RERANKED to send to the LLM.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 - free, no login, CPU-friendly.
"""
import copy
from typing import Sequence, Optional

from langchain_core.documents import Document
from langchain_core.documents.compressor import BaseDocumentCompressor
from langchain_core.callbacks import Callbacks
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.cross_encoders.base import BaseCrossEncoder

from src import config

_reranker_model_singleton = None


def get_cross_encoder() -> HuggingFaceCrossEncoder:
    global _reranker_model_singleton
    if _reranker_model_singleton is None:
        _reranker_model_singleton = HuggingFaceCrossEncoder(
            model_name=config.RERANKER_MODEL,
            model_kwargs={"device": "cpu"},
        )
    return _reranker_model_singleton


class ScoredCrossEncoderReranker(BaseDocumentCompressor):
    """
    Like LangChain's built-in CrossEncoderReranker, but also writes the
    cross-encoder's relevance score into each returned Document's metadata
    (as 'relevance_score') so it can be shown in the retrieval-visualization
    panel of the UI. LangChain's stock version sorts by score but discards it.
    """
    model: BaseCrossEncoder
    top_n: int = 3

    model_config = {"arbitrary_types_allowed": True}

    def compress_documents(
        self,
        documents: Sequence[Document],
        query: str,
        callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        if not documents:
            return []
        scores = self.model.score([(query, doc.page_content) for doc in documents])
        scored_docs = []
        for doc, score in zip(documents, scores):
            new_doc = copy.deepcopy(doc)
            new_doc.metadata["relevance_score"] = float(score)
            scored_docs.append(new_doc)
        scored_docs = sorted(scored_docs, key=lambda d: d.metadata["relevance_score"], reverse=True)
        return scored_docs[: self.top_n]


def get_reranking_retriever(base_retriever, top_n: int = None) -> ContextualCompressionRetriever:
    top_n = top_n or config.TOP_K_RERANKED
    compressor = ScoredCrossEncoderReranker(model=get_cross_encoder(), top_n=top_n)
    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever,
    )


if __name__ == "__main__":
    from src.vectorstore import load_vectorstore, get_retriever

    vs = load_vectorstore()
    base_retriever = get_retriever(vs, k=config.TOP_K_RETRIEVE)
    reranking_retriever = get_reranking_retriever(base_retriever, top_n=3)

    query = "What is RAG and why does chunking matter?"
    results = reranking_retriever.invoke(query)
    print(f"Reranked top {len(results)} chunks for query: {query!r}\n")
    for i, doc in enumerate(results, 1):
        print(f"[{i}] {doc.metadata['filename']} p.{doc.metadata['page']} "
              f"relevance_score={doc.metadata.get('relevance_score', 'n/a')}")
        print(doc.page_content[:200], "\n")

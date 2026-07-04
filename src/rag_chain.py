"""
Stage 6 (Generation) + Stage 7 (Latency) - LangChain-native
--------------------------------------------------------------
Builds the full RAG chain:
    question -> retrieve (Chroma) -> rerank (cross-encoder)
             -> format context with citation tags
             -> Groq LLM -> answer (instructed to cite [filename, p.X])

`ask()` is the main entry point used by the Streamlit app and eval script.
It returns a dict with the answer, structured sources, the raw retrieved
chunks (for the "retrieval visualization" panel), and a latency breakdown
so we can track the 2-5s end-to-end budget.
"""
import time
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

from src import config
from src.vectorstore import load_vectorstore, get_retriever
from src.reranker import get_reranking_retriever

SYSTEM_PROMPT = """You are a careful research assistant answering questions using ONLY \
the provided context excerpts from a private PDF corpus.

Rules:
1. Answer ONLY using information found in the context below. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say so plainly instead \
of guessing.
3. Every factual claim in your answer MUST be followed by an inline citation in the \
exact format [filename, p.X], where filename and page number come from the context \
excerpt you used.
4. If multiple excerpts support a claim, cite all of them, e.g. [a.pdf, p.4][b.pdf, p.12].
5. Be concise and direct. Do not repeat the question back.

Context excerpts:
{context}
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


def _format_context(chunks: List[Document]) -> str:
    blocks = []
    for i, doc in enumerate(chunks, 1):
        tag = f"[{doc.metadata['filename']}, p.{doc.metadata['page']}]"
        blocks.append(f"Excerpt {i} {tag}:\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _extract_sources(chunks: List[Document]) -> List[Dict[str, Any]]:
    seen = set()
    sources = []
    for doc in chunks:
        key = (doc.metadata["filename"], doc.metadata["page"])
        if key not in seen:
            seen.add(key)
            sources.append({
                "filename": doc.metadata["filename"],
                "page": doc.metadata["page"],
                "source_type": doc.metadata.get("source_type", "unknown"),
            })
    return sources


class RAGChatbot:
    """Holds the retriever + LLM so models are loaded once, not per-query."""

    def __init__(self):
        if not config.GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
                "free key from https://console.groq.com/keys"
            )
        vectorstore = load_vectorstore()
        base_retriever = get_retriever(vectorstore, k=config.TOP_K_RETRIEVE)
        self.retriever = get_reranking_retriever(base_retriever, top_n=config.TOP_K_RERANKED)

        self.llm = ChatGroq(
            model=config.GROQ_MODEL,
            api_key=config.GROQ_API_KEY,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
        )
        self.chain = PROMPT | self.llm | StrOutputParser()

    def ask(self, question: str) -> Dict[str, Any]:
        t_start = time.time()

        t0 = time.time()
        retrieved_chunks = self.retriever.invoke(question)
        retrieval_latency = time.time() - t0

        if not retrieved_chunks:
            return {
                "answer": "I couldn't find any relevant information in the document "
                          "corpus to answer this question.",
                "sources": [],
                "retrieved_chunks": [],
                "latency": {
                    "retrieval_s": round(retrieval_latency, 3),
                    "generation_s": 0.0,
                    "total_s": round(time.time() - t_start, 3),
                },
            }

        context = _format_context(retrieved_chunks)

        t0 = time.time()
        answer = self.chain.invoke({"context": context, "question": question})
        generation_latency = time.time() - t0

        total_latency = time.time() - t_start

        return {
            "answer": answer,
            "sources": _extract_sources(retrieved_chunks),
            "retrieved_chunks": [
                {
                    "filename": d.metadata["filename"],
                    "page": d.metadata["page"],
                    "source_type": d.metadata.get("source_type"),
                    "relevance_score": d.metadata.get("relevance_score"),
                    "text": d.page_content,
                }
                for d in retrieved_chunks
            ],
            "latency": {
                "retrieval_s": round(retrieval_latency, 3),
                "generation_s": round(generation_latency, 3),
                "total_s": round(total_latency, 3),
            },
        }


if __name__ == "__main__":
    bot = RAGChatbot()
    question = "What is RAG and why does chunking matter?"
    result = bot.ask(question)

    print(f"Q: {question}\n")
    print(f"A: {result['answer']}\n")
    print("Sources:")
    for s in result["sources"]:
        print(f"  - {s['filename']} p.{s['page']} ({s['source_type']})")
    print(f"\nLatency: {result['latency']}")

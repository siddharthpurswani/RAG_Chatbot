"""
Streamlit demo UI for the RAG chatbot.

Run:
    streamlit run app.py

Shows:
  - Sidebar: ingestion pipeline stats (PDFs, pages, chunks, OCR count)
  - Main: chat interface
  - Per answer: retrieval visualization (top retrieved/reranked chunks)
    + final generated answer with inline citations + latency breakdown
"""
import os
import json
import glob

import streamlit as st

from src import config

st.set_page_config(page_title="RAG Chatbot (Hackathon Demo)", layout="wide")


@st.cache_resource(show_spinner="Loading models (embeddings, reranker, Groq)...")
def get_bot():
    from src.rag_chain import RAGChatbot
    return RAGChatbot()


def pipeline_stats():
    """Reads cached ingestion JSONs to summarize what's been indexed."""
    cache_files = glob.glob(os.path.join(config.PROCESSED_DIR, "*.json"))
    n_pdfs = len(cache_files)
    n_pages = 0
    n_ocr = 0
    filenames = []
    for cf in cache_files:
        with open(cf, "r", encoding="utf-8") as f:
            d = json.load(f)
        n_pages += d["num_pages"]
        n_ocr += d["num_ocr_pages"]
        filenames.append(d["filename"])
    return n_pdfs, n_pages, n_ocr, filenames


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("📚 Ingestion Pipeline")
    n_pdfs, n_pages, n_ocr, filenames = pipeline_stats()

    if n_pdfs == 0:
        st.warning(
            "No ingested PDFs found.\n\n"
            "Run this first:\n```\npython -m src.build_index\n```"
        )
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("PDFs", n_pdfs)
        col2.metric("Pages", n_pages)
        col3.metric("OCR pages", n_ocr)

        with st.expander("Indexed files"):
            for fn in filenames:
                st.write(f"- {fn}")

    st.divider()
    st.header("⚙️ Config")
    st.caption(f"Embedding: `{config.EMBEDDING_MODEL}`")
    st.caption(f"Reranker: `{config.RERANKER_MODEL}`")
    st.caption(f"LLM (Groq): `{config.GROQ_MODEL}`")
    st.caption(f"Retrieve top-{config.TOP_K_RETRIEVE} -> rerank top-{config.TOP_K_RERANKED}")

    if not config.GROQ_API_KEY:
        st.error("GROQ_API_KEY not set. Add it to your .env file.")


# ---------------- Main chat ----------------
st.title("🤖 RAG Chatbot — Private PDF Knowledge Base")
st.caption("Open-source embeddings + Chroma + cross-encoder reranking + Groq generation")

if "history" not in st.session_state:
    st.session_state.history = []

if n_pdfs == 0 or not config.GROQ_API_KEY:
    st.info("Complete setup (ingest PDFs + set GROQ_API_KEY) to start chatting.")
    st.stop()

bot = get_bot()

# render past turns
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["question"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])

question = st.chat_input("Ask a question about your PDF corpus...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating..."):
            result = bot.ask(question)

        st.write(result["answer"])

        lat = result["latency"]
        badge_color = "green" if 2 <= lat["total_s"] <= 5 else "orange"
        st.markdown(
            f":{badge_color}[**Latency:** total {lat['total_s']}s "
            f"(retrieval {lat['retrieval_s']}s + generation {lat['generation_s']}s)]"
        )

        with st.expander(f"📎 Sources ({len(result['sources'])})"):
            for s in result["sources"]:
                st.write(f"- **{s['filename']}**, page {s['page']} "
                         f"({'OCR' if s['source_type'] == 'ocr' else 'native text'})")

        with st.expander(f"🔍 Retrieval visualization - top {len(result['retrieved_chunks'])} chunks"):
            for i, chunk in enumerate(result["retrieved_chunks"], 1):
                score = chunk.get("relevance_score")
                score_str = f"{score:.3f}" if score is not None else "n/a"
                st.markdown(
                    f"**[{i}] {chunk['filename']} — p.{chunk['page']}** "
                    f"(source: {chunk['source_type']}, rerank score: {score_str})"
                )
                st.text(chunk["text"][:500] + ("..." if len(chunk["text"]) > 500 else ""))
                st.markdown("---")

    st.session_state.history.append({
        "question": question,
        "answer": result["answer"],
    })

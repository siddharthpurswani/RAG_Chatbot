# RAG Chatbot — Hackathon Project (LangChain edition)

Free/open-source Retrieval-Augmented Generation chatbot over a large private
PDF corpus (10+ PDFs, 200+ pages each), built entirely with **LangChain**.

**Stack**
| Component        | Choice                                   | Why |
|-------------------|-------------------------------------------|-----|
| PDF text extraction | PyMuPDF (`fitz`)                        | Fast native text extraction |
| OCR               | Tesseract (`pytesseract`)                 | Free, handles scanned pages |
| Chunking          | LangChain `RecursiveCharacterTextSplitter` (token-aware) | Preserves semantic boundaries |
| Embeddings        | `BAAI/bge-small-en-v1.5` (`HuggingFaceBgeEmbeddings`) | Free, no login, strong small retrieval model |
| Vector DB         | Chroma (`langchain-chroma`)               | Free, open-source, built-in HNSW ANN index |
| Reranker          | `cross-encoder/ms-marco-MiniLM-L-6-v2`    | Free, sharpens top-k precision |
| LLM (generation)  | Groq (`langchain-groq`, `llama-3.1-8b-instant`) | Free tier, very low latency |
| UI                | Streamlit                                 | Fast to build/demo |

## Pipeline stages (files in `src/`)
1. `ingestion.py` — native text extraction + OCR fallback → LangChain `Document`s (one per page), cached as JSON so OCR doesn't rerun every time.
2. `chunking.py` — splits pages into ~700-token chunks (20% overlap) via LangChain's splitter, keeps `filename`/`page`/`pdf_id` metadata, adds `chunk_id`.
3. `vectorstore.py` — embeds chunks (`bge-small-en-v1.5`) and persists to Chroma.
4. `reranker.py` — cross-encoder reranking on top of Chroma similarity search (wrapped as a LangChain `ContextualCompressionRetriever`).
5. `rag_chain.py` — full RAG chain: retrieve → rerank → build a citation-enforcing prompt → Groq LLM → answer + sources + latency breakdown.
6. `build_index.py` — orchestrates stages 1–4 in one command (run once per corpus update).
7. `evaluate.py` — latency (p50/p95 vs the 2–5s target) and optional Recall@k / MRR if you supply labeled test cases.
8. `app.py` (project root) — Streamlit chat UI: ingestion stats, chat, retrieval visualization, citations, latency badge.

---

## One-time local setup (VSCode / your machine)

### 1. System dependencies (needed for OCR)
**Windows**
- Tesseract: https://github.com/UB-Mannheim/tesseract/wiki

**macOS**
```bash
brew install tesseract poppler
```

**Linux (Ubuntu/Debian)**
```bash
sudo apt-get install tesseract-ocr poppler-utils
```

### 2. Python virtual environment
```bash
cd rag-chatbot
python -m venv venv

# activate:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```
> Note: `requirements.txt` pins `langchain==0.3.*`. LangChain 1.0 shipped a
> breaking restructure of `langchain.retrievers` etc. — 0.3.x is the stable,
> widely-documented line this project (and most current tutorials) use.

### 3. Configure secrets
```bash
cp .env.example .env
```
Edit `.env` and paste your **free** Groq API key from https://console.groq.com/keys.
All other settings (models, chunk size, top-k, etc.) are also tunable there.

### 4. Add your PDFs
Drop your 10+ real PDFs into `data/raw_pdfs/`. They're gitignored (private, large — never committed).

### 5. (Optional) sanity-check with a synthetic PDF first
```bash
python tests/make_sample_pdf.py
```
This generates `data/raw_pdfs/sample_test.pdf` — 2 native text pages + 1
scanned (image-only) page — so you can verify OCR fallback works before
touching your real corpus. Delete it before indexing your real PDFs.

### 6. Build the index (run once, or whenever PDFs change)
```bash
python -m src.build_index
```
This runs ingestion → chunking → embedding → Chroma persistence, and prints
stats (pages ingested, OCR pages, chunks indexed).

### 7. Run the chatbot UI
```bash
streamlit run app.py
```

### 8. (Optional) run evaluation
```bash
python -m src.evaluate
```
Reports p50/p95 latency against the 2–5s target. Add entries to
`LABELED_CASES` in `src/evaluate.py` for Recall@k / MRR retrieval metrics.

---

## Testing individual stages directly
Every stage file is runnable standalone for debugging:
```bash
python -m src.ingestion      # Stage 1 only
python -m src.chunking       # Stages 1+2
python -m src.vectorstore    # Stages 1-4, runs a sample query
python -m src.reranker       # Stages 1-5, runs a sample query (needs an existing index)
python -m src.rag_chain      # Full pipeline, one sample Q&A (needs GROQ_API_KEY)
```

## Project structure
```
rag-chatbot/
├── data/
│   ├── raw_pdfs/          # your source PDFs (gitignored)
│   └── processed/         # cached per-page ingestion JSON (gitignored)
├── chroma_db/             # persistent vector store (gitignored)
├── src/
│   ├── config.py
│   ├── ingestion.py
│   ├── chunking.py
│   ├── vectorstore.py
│   ├── reranker.py
│   ├── rag_chain.py
│   ├── build_index.py
│   └── evaluate.py
├── tests/
│   └── make_sample_pdf.py
├── app.py
├── .env / .env.example
├── requirements.txt
└── README.md
```

## Known limitations / things to tune for your hardware
- **Latency target (2–5s):** on CPU-only machines, embedding + reranking + Groq
  generation should comfortably land in this range for `TOP_K_RETRIEVE=20` /
  `TOP_K_RERANKED=5`. If you're over 5s, lower `TOP_K_RETRIEVE`, switch
  `GROQ_MODEL` to a smaller/faster model, or reduce `LLM_MAX_TOKENS`.
- **First run is slow:** the embedding + reranker models download from
  Hugging Face the first time they're used (cached afterward).
- **OCR-heavy corpora:** ~1–2s/page. This only happens once per PDF (cached
  to `data/processed/*.json`), not on every query.

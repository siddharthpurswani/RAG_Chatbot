"""
Central configuration. Loads from .env (see .env.example for the template).
Every other module imports from here instead of reading os.environ directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _path(rel_path: str) -> str:
    return os.path.join(BASE_DIR, rel_path)


# --- Secrets ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# --- Models ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# --- Paths ---
RAW_PDF_DIR = _path(os.getenv("RAW_PDF_DIR", "data/raw_pdfs"))
PROCESSED_DIR = _path(os.getenv("PROCESSED_DIR", "data/processed"))
CHROMA_DIR = _path(os.getenv("CHROMA_DIR", "chroma_db"))
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_hackathon")

# --- Ingestion ---
MIN_NATIVE_CHARS = int(os.getenv("MIN_NATIVE_CHARS", 20))
OCR_DPI = int(os.getenv("OCR_DPI", 200))
# Full path to the tesseract executable. Leave blank if tesseract is on your
# system PATH already (common on macOS/Linux via brew/apt). On Windows, if
# you didn't add it to PATH during install, set this explicitly, e.g.:
#   TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "").strip()

# --- Chunking ---
CHUNK_SIZE_TOKENS = int(os.getenv("CHUNK_SIZE_TOKENS", 700))
CHUNK_OVERLAP_RATIO = float(os.getenv("CHUNK_OVERLAP_RATIO", 0.2))
CHUNK_OVERLAP_TOKENS = int(CHUNK_SIZE_TOKENS * CHUNK_OVERLAP_RATIO)

# --- Retrieval ---
TOP_K_RETRIEVE = int(os.getenv("TOP_K_RETRIEVE", 10))   # candidates before reranking
TOP_K_RERANKED = int(os.getenv("TOP_K_RERANKED", 3))    # final chunks sent to LLM

# --- Generation ---
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.1))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", 1024))

# Make sure directories exist
os.makedirs(RAW_PDF_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "iuh_data"
PROCESSED_DIR = ROOT_DIR / "processed"
EVALUATION_DIR = ROOT_DIR / "evaluation"
INDEX_DIR = ROOT_DIR / "indexes"
RESULTS_DIR = ROOT_DIR / "results"
QDRANT_STORAGE_DIR = ROOT_DIR / "qdrant_storage"

CHUNKS_PATH = PROCESSED_DIR / "chunks.json"
CANDIDATE_QUESTIONS_PATH = PROCESSED_DIR / "candidate_questions.json"
FAILED_CHUNKS_PATH = PROCESSED_DIR / "failed_chunks.json"
FINAL_QUESTIONS_PATH = EVALUATION_DIR / "questions_final.json"
TRIPLES_PATH = PROCESSED_DIR / "knowledge_graph_triples.json"

DEFAULT_TOP_K = 5
DEFAULT_HISTORY_K = 3
DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"
DEFAULT_EMBEDDING_DEVICE = "auto"
DEFAULT_VECTOR_STORE = os.getenv("VECTOR_STORE", "qdrant")
DEFAULT_QDRANT_MODE = os.getenv("QDRANT_MODE", "local")
DEFAULT_QDRANT_PATH = os.getenv("QDRANT_PATH", str(QDRANT_STORAGE_DIR))
DEFAULT_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
DEFAULT_QDRANT_COLLECTION_PREFIX = os.getenv("QDRANT_COLLECTION_PREFIX", "iuh_rag")

EMBEDDING_MODELS = {
    "miniLM": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "e5": "intfloat/multilingual-e5-base",
    "bge_m3": "BAAI/bge-m3",
    "vietnamese_bi": "bkai-foundation-models/vietnamese-bi-encoder",
}

DATA_TYPES = {
    "cac_khoa",
    "cac_trung_tam",
    "dao_tao",
    "faq",
    "tong_quan_ve_truong",
    "tuyen_sinh",
}

VIETNAMESE_STOPWORDS = {
    "a",
    "ai",
    "anh",
    "ban",
    "bang",
    "bi",
    "biet",
    "boi",
    "cac",
    "can",
    "cho",
    "co",
    "con",
    "cua",
    "da",
    "dang",
    "de",
    "den",
    "di",
    "do",
    "duoc",
    "em",
    "gi",
    "hay",
    "hien",
    "hoi",
    "hoc",
    "khong",
    "la",
    "lam",
    "minh",
    "mot",
    "nao",
    "nay",
    "neu",
    "nhung",
    "o",
    "tai",
    "thi",
    "thay",
    "the",
    "toi",
    "trong",
    "truong",
    "tu",
    "va",
    "ve",
    "voi",
    "xin",
}

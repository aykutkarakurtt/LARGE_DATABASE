import os


def _env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")
VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
SIMILARITY_THRESHOLD = 0.5
DEFAULT_TOP_K = 5
SERVER_NAME = "0.0.0.0"
SERVER_PORT = 7861

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct")
OLLAMA_TIMEOUT = _env_int("OLLAMA_TIMEOUT", 120)
RAG_MAX_CONTEXT_CHARS = _env_int("RAG_MAX_CONTEXT_CHARS", 6000)
RAG_CONTEXT_SIZE = _env_int("RAG_CONTEXT_SIZE", 4096)
RAG_TEMPERATURE = _env_float("RAG_TEMPERATURE", 0.2)
RAG_MAX_TOKENS = _env_int("RAG_MAX_TOKENS", 512)

INDEX_FILE = os.path.join(VECTORSTORE_DIR, "faiss_index.bin")
METADATA_FILE = os.path.join(VECTORSTORE_DIR, "metadata.json")

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}

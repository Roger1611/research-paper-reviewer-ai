import os

OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b-instruct-q4_K_M"

ARXIV_MAX_RESULTS = 6

CHUNK_SIZE = 220
CHUNK_OVERLAP = 40
TOP_K_RETRIEVAL = 4
SYNTHESIS_TOP_K = 3

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
FAST_MODEL = "anthropic/claude-3.5-haiku"
STRONG_MODEL = "anthropic/claude-3.5-sonnet"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OLLAMA_FAST_MODEL = "llama3.1:8b-instruct-q4_K_M"
MAX_HYPOTHESIS_ITERATIONS = 2
MIN_CONFIDENCE_THRESHOLD = 0.35
HYPOTHESIS_TOP_K = 5
DEFAULT_BACKEND = "openrouter"

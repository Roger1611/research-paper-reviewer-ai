import logging

import requests
from langchain.tools import tool

import config
from core.chunker import split_text
from core.embedder import get_embeddings
from core.loader import _load_pdf_text
from core.retriever import create_faiss_index

logger = logging.getLogger(__name__)

# keyed by arxiv_id -> {"chunks": [...], "index": faiss_index}
_paper_store: dict = {}


def get_paper_store() -> dict:
    """Return the module-level store of loaded papers."""
    return _paper_store


@tool
def fetch_paper_text(arxiv_id: str) -> str:
    """Download, chunk, and index the full text of an ArXiv paper."""
    url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.error("failed to download %s: %s", arxiv_id, e)
        return f"couldn't download PDF for {arxiv_id}: {e}"

    try:
        text = _load_pdf_text(resp.content)
    except ValueError as e:
        logger.error("pdf extraction failed for %s: %s", arxiv_id, e)
        return f"couldn't extract text from {arxiv_id}: {e}"

    chunks = split_text(text, chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    if not chunks:
        return f"no usable text found in {arxiv_id}"

    embeddings = get_embeddings(chunks)
    index = create_faiss_index(embeddings)
    _paper_store[arxiv_id] = {"chunks": chunks, "index": index}

    return f"Loaded {len(chunks)} chunks from {arxiv_id}"

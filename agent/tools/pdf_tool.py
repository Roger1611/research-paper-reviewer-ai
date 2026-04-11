import requests
from langchain.tools import tool

from core.loader import load_bytes
from core.chunker import split_text
from core.embedder import get_embeddings
from core.retriever import create_faiss_index
import config

# TODO: download PDF from arxiv_id, chunk + embed, return (chunks, faiss_index)


@tool
def load_paper_pdf(arxiv_id: str) -> dict:
    """Download and index the full text of an ArXiv paper by ID."""
    # TODO
    raise NotImplementedError

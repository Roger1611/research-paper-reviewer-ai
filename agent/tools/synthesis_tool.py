import json
from langchain.tools import tool

from core.embedder import get_embeddings
from core.retriever import search_chunks
from agent.prompts import SYNTHESIS_PROMPT
import config

# TODO: query all per-paper indexes, score agreement/contradiction, return report JSON


@tool
def synthesize_papers(topic: str) -> str:
    """Cross-reference loaded papers and return a JSON synthesis report."""
    # TODO
    raise NotImplementedError

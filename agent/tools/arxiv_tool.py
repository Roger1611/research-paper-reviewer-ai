import arxiv
from langchain.tools import tool

import config

# TODO: implement fetch logic; return list of dicts with title/authors/arxiv_id/url/abstract


@tool
def fetch_papers(topic: str) -> list[dict]:
    """Fetch the top-N most relevant ArXiv papers for a research topic."""
    # TODO
    raise NotImplementedError

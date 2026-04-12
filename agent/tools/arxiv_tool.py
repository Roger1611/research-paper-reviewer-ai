import logging

import arxiv
from langchain.tools import tool

import config

logger = logging.getLogger(__name__)


@tool
def arxiv_search(topic: str) -> list[dict] | str:
    """Search ArXiv for papers on a topic. Returns a list of dicts, each with keys: title, authors, arxiv_id, abstract, pdf_url. Use the arxiv_id values to call fetch_paper_text."""
    try:
        client = arxiv.Client()
        query = arxiv.Search(
            query=topic,
            max_results=config.ARXIV_MAX_RESULTS,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = []
        for paper in client.results(query):
            results.append({
                "title": paper.title,
                "authors": ", ".join(a.name for a in paper.authors),
                "arxiv_id": paper.get_short_id(),
                "abstract": paper.summary.replace("\n", " "),
                "pdf_url": paper.pdf_url,
            })
        return results
    except Exception as e:
        logger.error("arxiv_search failed: %s", e)
        return f"couldn't fetch papers from ArXiv: {e}"

import json
import logging
import re

import numpy as np
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

import config
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.tools.arxiv_tool import arxiv_search
from agent.tools.pdf_tool import fetch_paper_text, get_paper_store
from agent.tools.synthesis_tool import synthesize_papers
from core.embedder import get_embeddings

logger = logging.getLogger(__name__)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _filter_relevant(papers: list[dict], topic: str, threshold: float = 0.25) -> list[dict]:
    topic_emb = get_embeddings([topic])[0]
    texts = [f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers]
    embs = get_embeddings(texts)
    kept = []
    for paper, emb in zip(papers, embs):
        sim = _cosine(topic_emb, emb)
        if sim >= threshold:
            kept.append(paper)
        else:
            print(f"[orchestrator] skipping off-topic: {paper.get('arxiv_id')} sim={sim:.3f} — {paper.get('title', '')[:60]}", flush=True)
    return kept


def _extract_arxiv_meta(messages: list) -> list[dict]:
    """Pull the arxiv_search result out of the tool messages, if present."""
    for msg in messages:
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "arxiv_search":
            try:
                parsed = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                parsed = None
            if isinstance(parsed, list):
                return parsed
    return []


def run_research_agent(topic: str, callbacks: list | None = None) -> dict:
    """Fetch papers via the agent, then run synthesis directly and return the report dict."""
    llm = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL)

    invoke_config: dict = {"recursion_limit": 15}
    if callbacks:
        invoke_config["callbacks"] = callbacks

    # Only give the agent arxiv_search — it can't be trusted to reliably chain tool calls
    try:
        graph = create_react_agent(llm, [arxiv_search], prompt=AGENT_SYSTEM_PROMPT)
        result = graph.invoke(
            {"messages": [HumanMessage(content=f"Search ArXiv for papers on this topic: {topic}")]},
            config=invoke_config,
        )
    except Exception as e:
        logger.error("agent execution failed: %s", e)
        return {"error": str(e), "raw": ""}

    paper_meta = _extract_arxiv_meta(result["messages"])
    if not paper_meta:
        logger.error("arxiv_search returned no results")
        return {"error": "no papers found", "raw": "", "_papers_meta": []}

    paper_meta = _filter_relevant(paper_meta, topic)
    if not paper_meta:
        return {"error": "all papers were off-topic after relevance filtering", "raw": "", "_papers_meta": []}

    print(f"[orchestrator] fetching {len(paper_meta)} papers...", flush=True)
    for paper in paper_meta:
        aid = paper.get("arxiv_id", "")
        if not aid:
            continue
        outcome = fetch_paper_text.invoke({"arxiv_id": aid})
        print(f"  {aid}: {outcome}", flush=True)

    store = get_paper_store()
    print(f"[orchestrator] paper store: {len(store)} papers loaded", flush=True)

    if not store:
        return {"error": "no PDFs could be loaded", "raw": "", "_papers_meta": paper_meta}

    # Call synthesis directly — don't trust the model to call it correctly
    print("[orchestrator] running synthesis...", flush=True)
    raw_synthesis = synthesize_papers.invoke({"topic": topic})
    synthesis = _extract_json(raw_synthesis) if isinstance(raw_synthesis, str) else raw_synthesis

    if synthesis is None:
        return {"error": "synthesis returned invalid JSON", "raw": raw_synthesis, "_papers_meta": paper_meta}

    synthesis["_papers_meta"] = paper_meta
    return synthesis

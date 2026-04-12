import json
import logging
import re

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

import config
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.tools.arxiv_tool import arxiv_search
from agent.tools.pdf_tool import fetch_paper_text, get_paper_store
from agent.tools.synthesis_tool import synthesize_papers

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
    # Agent handles only paper fetching — synthesis is called directly below
    fetch_tools = [arxiv_search, fetch_paper_text]
    llm = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL)

    invoke_config: dict = {"recursion_limit": 30}
    if callbacks:
        invoke_config["callbacks"] = callbacks

    try:
        graph = create_react_agent(llm, fetch_tools, prompt=AGENT_SYSTEM_PROMPT)
        result = graph.invoke(
            {"messages": [HumanMessage(content=f"Find and load papers on this topic: {topic}")]},
            config=invoke_config,
        )
    except Exception as e:
        logger.error("agent execution failed: %s", e)
        return {"error": str(e), "raw": ""}

    print("[orchestrator] tool calls made:", flush=True)
    for msg in result["messages"]:
        if isinstance(msg, ToolMessage):
            print(f"  {getattr(msg, 'name', '?')}: {str(msg.content)[:120]}", flush=True)

    paper_meta = _extract_arxiv_meta(result["messages"])

    store = get_paper_store()
    print(f"[orchestrator] paper store after agent run: {len(store)} papers", flush=True)

    if not store:
        return {"error": "agent loaded no papers", "raw": "", "_papers_meta": paper_meta}

    # Call synthesis directly — don't trust the model to call it correctly
    print("[orchestrator] running synthesis...", flush=True)
    raw_synthesis = synthesize_papers.invoke({"topic": topic})
    synthesis = _extract_json(raw_synthesis) if isinstance(raw_synthesis, str) else raw_synthesis

    if synthesis is None:
        return {"error": "synthesis returned invalid JSON", "raw": raw_synthesis, "_papers_meta": paper_meta}

    synthesis["_papers_meta"] = paper_meta
    return synthesis

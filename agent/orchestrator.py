import json
import logging
import re

import requests
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

import config
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.tools.arxiv_tool import arxiv_search
from agent.tools.pdf_tool import fetch_paper_text
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


def run_research_agent(topic: str, callbacks: list | None = None) -> dict:
    """Run the full research pipeline for a topic and return the synthesis report dict."""
    tools = [arxiv_search, fetch_paper_text, synthesize_papers]
    llm = ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL)

    invoke_config: dict = {"recursion_limit": 30}
    if callbacks:
        invoke_config["callbacks"] = callbacks

    try:
        graph = create_react_agent(llm, tools, prompt=AGENT_SYSTEM_PROMPT)
        result = graph.invoke(
            {"messages": [HumanMessage(content=f"Research this topic thoroughly and synthesize findings: {topic}")]},
            config=invoke_config,
        )
    except Exception as e:
        logger.error("agent execution failed: %s", e)
        return {"error": str(e), "raw": ""}

    output = result["messages"][-1].content
    parsed = _extract_json(output)
    if parsed is not None:
        return parsed

    logger.warning("agent output wasn't clean JSON, attempting Ollama extraction")
    try:
        resp = requests.post(
            config.OLLAMA_URL,
            json={
                "model": config.OLLAMA_MODEL,
                "prompt": f"Extract only the JSON object from this text and return nothing else:\n\n{output}",
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        rescued = _extract_json(resp.json()["response"])
        if rescued is not None:
            return rescued
    except Exception as e:
        logger.error("JSON rescue call failed: %s", e)

    return {"error": "agent output was not valid JSON", "raw": output}

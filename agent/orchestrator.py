import json
import logging
import re

import requests
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM

import config
from agent.memory import get_memory
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.tools.arxiv_tool import arxiv_search
from agent.tools.pdf_tool import fetch_paper_text
from agent.tools.synthesis_tool import synthesize_papers

logger = logging.getLogger(__name__)

_REACT_PROMPT = PromptTemplate.from_template(
    AGENT_SYSTEM_PROMPT
    + """
You have access to the following tools:

{tools}

Use this format strictly:

Thought: think about what to do next
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I have the final answer
Final Answer: <paste the raw JSON from synthesize_papers here, unchanged>

Previous conversation:
{chat_history}

Question: {input}
Thought:{agent_scratchpad}"""
)


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
    llm = OllamaLLM(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL)

    try:
        agent = create_react_agent(llm, tools, _REACT_PROMPT)
        executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=get_memory(),
            verbose=True,
            max_iterations=15,
            handle_parsing_errors=True,
            callbacks=callbacks or [],
        )
        result = executor.invoke({
            "input": f"Research this topic thoroughly and synthesize findings: {topic}"
        })
    except Exception as e:
        logger.error("agent execution failed: %s", e)
        return {"error": str(e), "raw": ""}

    output = result.get("output", "")
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

from langchain.agents import AgentExecutor, create_react_agent
from langchain_ollama import OllamaLLM

from agent.memory import build_memory
from agent.prompts import AGENT_SYSTEM_PROMPT
from agent.tools.arxiv_tool import fetch_papers
from agent.tools.pdf_tool import load_paper_pdf
from agent.tools.synthesis_tool import synthesize_papers

# TODO: wire up AgentExecutor with tools, memory, and prompt

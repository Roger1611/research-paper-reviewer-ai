import json
import logging
from typing import Literal

import numpy as np
from langgraph.graph import StateGraph, END

import config
from agent.backend import LLMBackend
from agent.prompts import TOPIC_DECOMPOSITION_PROMPT
from core.json_utils import parse_json
from agent.state import AgentState
from agent.tools.arxiv_tool import arxiv_search
from agent.tools.extract_tool import extract_methods_problems
from agent.tools.gap_tool import detect_gaps as _detect_gaps
from agent.tools.hypothesis_tool import generate_hypotheses as _generate_hypotheses, score_is_sufficient
from agent.tools.pdf_tool import fetch_paper_text, get_paper_store
from core.embedder import get_embeddings

logger = logging.getLogger(__name__)


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


def decompose_topic(state: AgentState) -> dict:
    llm = LLMBackend(state["backend"])
    prompt = TOPIC_DECOMPOSITION_PROMPT.format(topic=state["topic"])
    try:
        result = parse_json(llm.call(prompt, use_strong=False))
        if result is None:
            raise ValueError("empty or unparseable response from decomposition LLM")
        domain_a = result["domain_a"]
        domain_b = result["domain_b"]
    except Exception as e:
        logger.error("topic decomposition failed: %s", e)
        # fall back to treating the whole topic as both domains
        domain_a = state["topic"]
        domain_b = state["topic"]
    print(f"[decompose] Domain A: {domain_a} | Domain B: {domain_b}", flush=True)
    return {"domain_a": domain_a, "domain_b": domain_b}


def search_domains(state: AgentState) -> dict:
    def fetch_domain(domain: str) -> list[dict]:
        raw = arxiv_search.invoke({"topic": domain})
        papers = raw if isinstance(raw, list) else []
        papers = _filter_relevant(papers, domain)
        for paper in papers:
            aid = paper.get("arxiv_id", "")
            if aid:
                fetch_paper_text.invoke({"arxiv_id": aid})
        return papers

    papers_a = fetch_domain(state["domain_a"])
    papers_b = fetch_domain(state["domain_b"])
    print(f"[search] domain_a: {len(papers_a)} papers | domain_b: {len(papers_b)} papers", flush=True)
    return {"papers_a": papers_a, "papers_b": papers_b}


def extract_knowledge(state: AgentState) -> dict:
    store = get_paper_store()
    all_methods: list[str] = []
    all_problems: list[str] = []

    for paper in state["papers_a"]:
        aid = paper.get("arxiv_id", "")
        if aid not in store:
            continue
        result = extract_methods_problems(aid, state["domain_a"], store[aid]["chunks"], state["backend"])
        all_methods.extend(result["methods"])

    for paper in state["papers_b"]:
        aid = paper.get("arxiv_id", "")
        if aid not in store:
            continue
        result = extract_methods_problems(aid, state["domain_b"], store[aid]["chunks"], state["backend"])
        all_problems.extend(result["problems"])

    return {
        "methods": list(set(all_methods)),
        "problems": list(set(all_problems)),
    }


def detect_gaps(state: AgentState) -> dict:
    gaps = _detect_gaps(state["methods"], state["problems"], state["backend"])
    return {"gaps": gaps}


def generate_hypotheses(state: AgentState) -> dict:
    hypotheses = _generate_hypotheses(state["gaps"], get_paper_store(), state["topic"], state["backend"])
    return {"hypotheses": hypotheses, "iteration": state["iteration"] + 1}


def should_loop(state: AgentState) -> Literal["generate_hypotheses", "build_report"]:
    if not score_is_sufficient(state["hypotheses"]) and state["iteration"] < config.MAX_HYPOTHESIS_ITERATIONS:
        print(
            f"[loop] iteration {state['iteration']}: top score insufficient, re-running hypothesis generation",
            flush=True,
        )
        return "generate_hypotheses"
    return "build_report"


def build_report(state: AgentState) -> dict:
    # Combine both domain paper lists into one deduped list for the formatter
    seen: set[str] = set()
    papers: list[dict] = []
    for p in state["papers_a"] + state["papers_b"]:
        aid = p.get("arxiv_id", "")
        if aid not in seen:
            seen.add(aid)
            p.setdefault("url", f"https://arxiv.org/abs/{aid}" if aid else "")
            papers.append(p)

    report = {
        "topic": state["topic"],
        "domain_a": state["domain_a"],
        "domain_b": state["domain_b"],
        "papers": papers,
        "methods": state["methods"],
        "problems": state["problems"],
        "gaps": state["gaps"],
        "hypotheses": state["hypotheses"],
        "iterations": state["iteration"],
    }
    return {"final_report": report}


def _build_graph() -> StateGraph:
    g = StateGraph(AgentState)
    g.add_node("decompose_topic", decompose_topic)
    g.add_node("search_domains", search_domains)
    g.add_node("extract_knowledge", extract_knowledge)
    g.add_node("detect_gaps", detect_gaps)
    g.add_node("generate_hypotheses", generate_hypotheses)
    g.add_node("build_report", build_report)

    g.set_entry_point("decompose_topic")
    g.add_edge("decompose_topic", "search_domains")
    g.add_edge("search_domains", "extract_knowledge")
    g.add_edge("extract_knowledge", "detect_gaps")
    g.add_edge("detect_gaps", "generate_hypotheses")
    g.add_conditional_edges("generate_hypotheses", should_loop)
    g.add_edge("build_report", END)

    return g.compile()


_graph = _build_graph()


def run_research_agent(topic: str, backend: str = config.DEFAULT_BACKEND) -> dict:
    initial: AgentState = {
        "topic": topic,
        "backend": backend,
        "domain_a": "",
        "domain_b": "",
        "papers_a": [],
        "papers_b": [],
        "methods": [],
        "problems": [],
        "gaps": [],
        "hypotheses": [],
        "iteration": 0,
        "final_report": {},
        "error": "",
    }
    result = _graph.invoke(initial)
    return result.get("final_report", {})

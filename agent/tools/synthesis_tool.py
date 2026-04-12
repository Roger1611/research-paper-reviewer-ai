import json
import logging
import re

import numpy as np
import requests
from langchain.tools import tool

import config
from agent.prompts import SYNTHESIS_PROMPT
from agent.tools.pdf_tool import get_paper_store
from core.embedder import get_embeddings
from core.retriever import search_chunks

logger = logging.getLogger(__name__)


def _call_ollama(prompt: str) -> str:
    resp = requests.post(
        config.OLLAMA_URL,
        json={"model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["response"]


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _score_confidence(report: dict, paper_chunks: dict[str, list[str]], topic_emb: np.ndarray) -> dict:
    """Replace placeholder 0.0 confidence values with cosine similarity scores."""
    for finding in report.get("consensus", []):
        if not isinstance(finding, dict):
            continue  # simplified schema returns strings; skip scoring
        cited = [c for aid in finding.get("citations", []) for c in paper_chunks.get(aid, [])]
        if not cited:
            finding["confidence"] = 0.0
            continue
        sims = [_cosine(topic_emb, e) for e in get_embeddings(cited)]
        finding["confidence"] = round(float(np.mean(sims)), 3)
    return report


@tool
def synthesize_papers(topic: str) -> str:
    """Cross-reference all papers loaded so far and return a JSON synthesis report. Pass the research topic as a plain string (e.g. 'knowledge distillation'). Do NOT pass a list of papers — call this after fetch_paper_text has been called for each paper."""
    store = get_paper_store()
    if not store:
        return "no papers loaded — run fetch_paper_text on at least one paper first"

    print(f"[synthesis] paper store has {len(store)} papers: {list(store.keys())}", flush=True)

    topic_emb = get_embeddings([topic])[0]

    paper_chunks: dict[str, list[str]] = {}
    for arxiv_id, data in store.items():
        hits = search_chunks(
            data["index"], data["chunks"],
            topic_emb.reshape(1, -1),
            top_k=config.SYNTHESIS_TOP_K,
        )
        paper_chunks[arxiv_id] = hits
        print(f"[synthesis] {arxiv_id}: retrieved {len(hits)} chunks", flush=True)

    paper_summaries = "\n\n".join(
        f"[{aid}]\n" + "\n---\n".join(chunks)
        for aid, chunks in paper_chunks.items()
    )

    prompt = SYNTHESIS_PROMPT.format(topic=topic, paper_summaries=paper_summaries)

    try:
        raw = _call_ollama(prompt)
    except Exception as e:
        logger.error("ollama call failed: %s", e)
        return f"couldn't reach Ollama: {e}"

    print(f"[synthesis] raw Ollama response:\n{raw}\n---", flush=True)

    report = _parse_json(raw)
    if report is None:
        logger.warning("first parse failed, retrying with strict prompt")
        try:
            raw = _call_ollama(f"Return ONLY valid JSON with no explanation or markdown.\n\n{prompt}")
        except Exception as e:
            return f"couldn't reach Ollama on retry: {e}"
        report = _parse_json(raw)

    if report is None:
        return f"synthesis failed: model returned invalid JSON after retry. Raw output:\n{raw[:500]}"

    report = _score_confidence(report, paper_chunks, topic_emb)
    return json.dumps(report)

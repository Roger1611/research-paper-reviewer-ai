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
from core.json_utils import parse_json
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


def _strip_arxiv_prefix(arxiv_id: str) -> str:
    return re.sub(r"(?i)^arxiv:", "", arxiv_id).strip()


def _normalise_citations(report: dict) -> dict:
    """Strip any 'arXiv:' prefix from citation IDs the model may have added."""
    for item in report.get("consensus", []):
        if isinstance(item, dict):
            item["citations"] = [_strip_arxiv_prefix(c) for c in item.get("citations", [])]
    for item in report.get("contested", []):
        if isinstance(item, dict):
            item["citations"] = [_strip_arxiv_prefix(c) for c in item.get("citations", [])]
    return report


def _validate_citations(report: dict, valid_ids: set[str]) -> dict:
    """Remove hallucinated IDs; resolve version-less IDs (e.g. 2405.09820 → 2405.09820v1)."""
    def resolve(cid: str) -> str | None:
        if cid in valid_ids:
            return cid
        base = re.sub(r"v\d+$", "", cid)
        for vid in valid_ids:
            if re.sub(r"v\d+$", "", vid) == base:
                return vid
        return None

    for item in report.get("consensus", []):
        if isinstance(item, dict):
            item["citations"] = [r for c in item.get("citations", []) if (r := resolve(c))]
    for item in report.get("contested", []):
        if isinstance(item, dict):
            item["citations"] = [r for c in item.get("citations", []) if (r := resolve(c))]
    return report



def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _score_confidence(report: dict, paper_chunks: dict[str, list[str]], topic_emb: np.ndarray) -> dict:
    all_chunks = [c for chunks in paper_chunks.values() for c in chunks]
    all_embs = get_embeddings(all_chunks) if all_chunks else []

    normalized = []
    for item in report.get("consensus", []):
        if isinstance(item, str):
            item = {"finding": item, "confidence": 0.0, "citations": []}

        cited = [c for aid in item.get("citations", []) for c in paper_chunks.get(aid, [])]
        pool_embs = get_embeddings(cited) if cited else all_embs

        if len(pool_embs) > 0:
            finding_emb = get_embeddings([item["finding"]])[0]
            sims = [_cosine(finding_emb, e) for e in pool_embs]
            item["confidence"] = round(float(np.mean(sims)), 3)
        else:
            item["confidence"] = 0.0

        normalized.append(item)

    report["consensus"] = normalized
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
    arxiv_ids = ", ".join(paper_chunks.keys())

    prompt = SYNTHESIS_PROMPT.format(topic=topic, paper_summaries=paper_summaries, arxiv_ids=arxiv_ids)

    try:
        raw = _call_ollama(prompt)
    except Exception as e:
        logger.error("ollama call failed: %s", e)
        return f"couldn't reach Ollama: {e}"

    print(f"[synthesis] raw Ollama response:\n{raw}\n---", flush=True)

    report = parse_json(raw)
    if report is None:
        logger.warning("first parse failed, retrying with strict prompt")
        try:
            raw = _call_ollama(f"Return ONLY valid JSON with no explanation or markdown.\n\n{prompt}")
        except Exception as e:
            return f"couldn't reach Ollama on retry: {e}"
        report = parse_json(raw)

    if report is None:
        return f"synthesis failed: model returned invalid JSON after retry. Raw output:\n{raw[:500]}"

    report = _normalise_citations(report)
    report = _validate_citations(report, set(paper_chunks.keys()))
    report = _score_confidence(report, paper_chunks, topic_emb)
    return json.dumps(report)

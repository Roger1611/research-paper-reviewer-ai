import json
import logging

import config
from agent.backend import LLMBackend
from agent.prompts import HYPOTHESIS_GENERATION_PROMPT
from core.embedder import get_embeddings
from core.json_utils import parse_json
from core.retriever import search_chunks

logger = logging.getLogger(__name__)


def generate_hypotheses(gaps: list[dict], paper_store: dict, topic: str, backend: str) -> list[dict]:
    llm = LLMBackend(backend)
    hypotheses = []

    for gap in gaps:
        query = gap.get("method", "") + " " + gap.get("problem", "")
        query_emb = get_embeddings([query])[0].reshape(1, -1)

        # Collect top chunks from every paper and track which papers contributed
        supporting_chunks = []
        cited_ids = []
        for arxiv_id, data in paper_store.items():
            hits = search_chunks(data["index"], data["chunks"], query_emb, top_k=config.SYNTHESIS_TOP_K)
            if hits:
                supporting_chunks.extend(hits)
                cited_ids.append(arxiv_id)

        prompt = HYPOTHESIS_GENERATION_PROMPT.format(
            gap=json.dumps(gap),
            supporting_chunks="\n---\n".join(supporting_chunks),
        )

        try:
            raw = llm.call(prompt, use_strong=True)
            hypothesis = parse_json(raw)
            if hypothesis is None:
                raise ValueError(f"could not parse JSON from response: {raw[:300]}")
        except Exception as e:
            logger.warning("hypothesis generation failed for gap '%s / %s': %s", gap.get("method"), gap.get("problem"), e)
            continue

        hypothesis["citations"] = cited_ids
        hypotheses.append(hypothesis)

    hypotheses.sort(key=lambda h: h.get("feasibility_score", 0.0), reverse=True)
    top = hypotheses[:config.HYPOTHESIS_TOP_K]

    top_score = top[0]["feasibility_score"] if top else 0.0
    print(f"[hypothesis] {len(top)} hypotheses generated, top score: {top_score:.2f}", flush=True)
    return top


def score_is_sufficient(hypotheses: list[dict]) -> bool:
    return bool(hypotheses) and hypotheses[0]["feasibility_score"] >= config.MIN_CONFIDENCE_THRESHOLD

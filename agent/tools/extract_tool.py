import json
import logging

from agent.backend import LLMBackend
from agent.prompts import EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


def extract_methods_problems(arxiv_id: str, domain: str, chunks: list[str], backend: str) -> dict:
    llm = LLMBackend(backend)
    prompt = EXTRACTION_PROMPT.format(
        arxiv_id=arxiv_id,
        domain=domain,
        chunks="\n---\n".join(chunks),
    )

    try:
        raw = llm.call(prompt, use_strong=False)
        result = json.loads(raw)
    except (json.JSONDecodeError, Exception) as e:
        logger.warning("extract_methods_problems failed for %s: %s", arxiv_id, e)
        return {"methods": [], "problems": []}

    methods = result.get("methods", [])
    problems = result.get("problems", [])
    print(f"[extract] {arxiv_id}: {len(methods)} methods, {len(problems)} problems", flush=True)
    return {"methods": methods, "problems": problems}

import logging

from agent.backend import LLMBackend
from agent.prompts import GAP_DETECTION_PROMPT
from core.json_utils import parse_json

logger = logging.getLogger(__name__)


def detect_gaps(methods: list[str], problems: list[str], backend: str) -> list[dict]:
    # dedup while preserving some ordering predictability
    unique_methods = list(set(methods))
    unique_problems = list(set(problems))

    llm = LLMBackend(backend)
    prompt = GAP_DETECTION_PROMPT.format(
        methods_list="\n".join(f"- {m}" for m in unique_methods),
        problems_list="\n".join(f"- {p}" for p in unique_problems),
    )

    try:
        raw = llm.call(prompt, use_strong=True)
        result = parse_json(raw)
        if result is None:
            raise ValueError(f"could not parse JSON from response: {raw[:300]}")
    except Exception as e:
        logger.warning("detect_gaps failed: %s", e)
        return []

    gaps = result.get("gaps", [])
    print(f"[gap] found {len(gaps)} gaps", flush=True)
    return gaps

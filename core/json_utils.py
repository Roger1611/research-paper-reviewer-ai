import json
import re


def parse_json(text: str) -> dict | None:
    """Parse JSON from an LLM response, stripping markdown fences if present."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[^\n]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None

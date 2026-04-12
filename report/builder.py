from datetime import datetime, timezone
from typing import Any


_DEFAULTS: dict[str, Any] = {
    "papers": [],
    "consensus": [],
    "contested": [],
    "open_problems": [],
    "synthesis_narrative": "",
    "hypotheses": [],
}


def build_report(topic: str, synthesis_json: dict, paper_metadata: list[dict]) -> dict[str, Any]:
    """Merge synthesis output with ArXiv metadata and return a validated report dict."""
    report: dict[str, Any] = {"topic": topic}

    for key, default in _DEFAULTS.items():
        report[key] = synthesis_json.get(key, default)

    # Build a lookup so we can enrich papers[] with full metadata from arxiv_search
    meta_by_id = {p["arxiv_id"]: p for p in paper_metadata if "arxiv_id" in p}

    enriched = []
    for paper in report["papers"]:
        aid = paper.get("arxiv_id", "")
        if aid in meta_by_id:
            merged = {**paper, **meta_by_id[aid]}
        else:
            merged = paper
        merged.setdefault("url", f"https://arxiv.org/abs/{aid}" if aid else "")
        enriched.append(merged)

    # Add any papers from metadata that the LLM omitted from papers[]
    seen = {p.get("arxiv_id") for p in enriched}
    for meta in paper_metadata:
        if meta.get("arxiv_id") not in seen:
            meta.setdefault("url", f"https://arxiv.org/abs/{meta['arxiv_id']}")
            enriched.append(meta)

    report["papers"] = enriched

    # normalise consensus: model may return plain strings instead of dicts
    report["consensus"] = [
        {"finding": item, "confidence": 0.0, "citations": []} if isinstance(item, str) else item
        for item in report["consensus"]
    ]

    report["hypotheses"] = synthesis_json.get("hypotheses", [])
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    return report

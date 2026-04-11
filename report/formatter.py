_BAR_FILLED = "█"
_BAR_EMPTY = "░"
_BAR_LEN = 8


def _confidence_bar(score: float) -> str:
    filled = round(score * _BAR_LEN)
    bar = _BAR_FILLED * filled + _BAR_EMPTY * (_BAR_LEN - filled)
    return f"{bar} {round(score * 100)}%"


def _arxiv_link(arxiv_id: str) -> str:
    return f"[{arxiv_id}](https://arxiv.org/abs/{arxiv_id})"


def format_report_markdown(report: dict) -> str:
    topic = report.get("topic", "Unknown Topic")
    lines = [f"## Research Synthesis: {topic}", ""]

    # Papers Analyzed
    lines += ["### Papers Analyzed", ""]
    for p in report.get("papers", []):
        aid = p.get("arxiv_id", "")
        title = p.get("title", aid)
        authors = p.get("authors", "")
        url = p.get("url") or f"https://arxiv.org/abs/{aid}"
        lines.append(f"- **[{title}]({url})**" + (f"  \n  {authors}" if authors else ""))
    lines.append("")

    # Consensus Findings
    lines += ["### Consensus Findings", ""]
    for item in report.get("consensus", []):
        finding = item.get("finding", "")
        confidence = item.get("confidence", 0.0)
        citations = item.get("citations", [])
        bar = _confidence_bar(confidence)
        cite_str = "  \n  *" + " · ".join(_arxiv_link(c) for c in citations) + "*" if citations else ""
        lines.append(f"- {finding}  \n  `{bar}`{cite_str}")
    lines.append("")

    # Contested Claims
    lines += ["### Contested Claims", ""]
    for item in report.get("contested", []):
        claim = item.get("claim", "")
        positions = item.get("positions", [])
        citations = item.get("citations", [])
        lines.append(f"- **{claim}**")
        for pos in positions:
            lines.append(f"  - {pos}")
        if citations:
            lines.append("  *" + " · ".join(_arxiv_link(c) for c in citations) + "*")
    lines.append("")

    # Open Problems
    lines += ["### Open Problems", ""]
    for problem in report.get("open_problems", []):
        lines.append(f"- {problem}")
    lines.append("")

    # Synthesis Narrative
    lines += ["### Full Synthesis Narrative", ""]
    lines.append(report.get("synthesis_narrative", ""))

    return "\n".join(lines)

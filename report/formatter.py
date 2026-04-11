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


def format_report_html(report: dict) -> str:
    """Wrap the markdown report in minimal styled HTML for st.components.v1.html."""
    md = format_report_markdown(report)
    # Escape for embedding, then naive md -> html for the bits Streamlit won't render
    escaped = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    # Convert markdown links [text](url) -> <a> tags
    import re
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        r'<a href="\2" target="_blank">\1</a>',
        escaped,
    )
    # Bold
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)
    # Code spans (confidence bars)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    # Headings
    escaped = re.sub(r"^### (.+)$", r"<h3>\1</h3>", escaped, flags=re.MULTILINE)
    escaped = re.sub(r"^## (.+)$", r"<h2>\1</h2>", escaped, flags=re.MULTILINE)
    # List items
    escaped = re.sub(r"^- (.+)$", r"<li>\1</li>", escaped, flags=re.MULTILINE)
    escaped = re.sub(r"(<li>.*</li>\n?)+", r"<ul>\g<0></ul>", escaped, flags=re.DOTALL)
    # Paragraphs (double newline -> <p>)
    paragraphs = re.split(r"\n{2,}", escaped)
    body = "\n".join(
        p if re.match(r"^\s*<(h[23]|ul|li)", p) else f"<p>{p}</p>"
        for p in paragraphs if p.strip()
    )

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         line-height: 1.6; max-width: 860px; margin: 0 auto; padding: 1rem 1.5rem;
         color: #1a1a1a; }}
  h2 {{ border-bottom: 2px solid #e0e0e0; padding-bottom: .3rem; }}
  h3 {{ margin-top: 1.5rem; color: #333; }}
  code {{ font-family: monospace; background: #f4f4f4; padding: 1px 5px; border-radius: 3px; }}
  a {{ color: #0969da; text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  li {{ margin-bottom: .4rem; }}
  ul {{ padding-left: 1.4rem; }}
</style>
</head>
<body>
{body}
</body>
</html>"""

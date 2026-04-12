import sys

sys.stdout.reconfigure(encoding="utf-8")

from agent.orchestrator import run_research_agent
from report.builder import build_report
from report.formatter import format_report_markdown

if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "transformer efficiency in edge deployment"
    result = run_research_agent(topic)
    paper_meta = result.pop("_papers_meta", [])
    report = build_report(topic, result, paper_meta)
    print(format_report_markdown(report))

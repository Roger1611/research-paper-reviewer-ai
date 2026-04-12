import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

from agent.orchestrator import run_research_agent
from report.builder import build_report
from report.formatter import format_report_markdown

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous research synthesis agent")
    parser.add_argument("topic", help="Research topic to investigate")
    parser.add_argument("--backend", choices=["openrouter", "ollama"], default="openrouter")
    args = parser.parse_args()

    print(f"Backend: {args.backend}", flush=True)

    result = run_research_agent(args.topic, backend=args.backend)
    paper_meta = result.pop("_papers_meta", [])
    report = build_report(args.topic, result, paper_meta)
    print(format_report_markdown(report))

from typing import TypedDict


class AgentState(TypedDict):
    topic: str
    domain_a: str
    domain_b: str
    papers_a: list[dict]
    papers_b: list[dict]
    methods: list[str]       # extracted from domain_a papers
    problems: list[str]      # extracted from domain_b papers
    gaps: list[dict]         # {method, problem, reasoning}
    hypotheses: list[dict]   # {hypothesis, feasibility_score, evidence, citations}
    iteration: int
    final_report: dict
    error: str
    backend: str

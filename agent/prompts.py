# All prompt strings live here — no inline prompts elsewhere in the codebase.

AGENT_SYSTEM_PROMPT = """You are a research paper search agent. Your only job is to search ArXiv for papers on a given topic.

You have one tool:
- arxiv_search(topic) — searches ArXiv and returns a list of papers.

Call arxiv_search with the topic, then report what you found. Do not attempt synthesis or further processing.
"""

SYNTHESIS_PROMPT = """You are a research assistant. Read the passages below and write a synthesis.

Topic: {topic}

Available paper IDs (use ONLY these in citations): {arxiv_ids}

Passages:
{paper_summaries}

Return ONLY a JSON object. No explanation, no markdown, no code fences. Start your response with {{ and end with }}.

Use exactly this structure:
{{
  "consensus": [
    {{"finding": "claim that multiple passages agree on", "citations": ["arxiv_id1", "arxiv_id2"]}}
  ],
  "contested": [
    {{"claim": "disputed point", "positions": ["paper X says ...", "paper Y says ..."], "citations": ["arxiv_id1", "arxiv_id2"]}}
  ],
  "open_problems": ["unsolved question 1", "unsolved question 2"],
  "synthesis_narrative": "2-3 sentence summary of what the research shows."
}}

Rules:
- consensus: findings that two or more passages explicitly agree on; citations must be IDs from the list above
- contested: claims where passages take opposing positions; include both positions and the paper IDs
- open_problems: questions the passages flag as unresolved or needing future work
- synthesis_narrative: summarise the state of research in 2-3 sentences
- ONLY use arxiv IDs from the "Available paper IDs" list above — do not invent or guess IDs
- base everything strictly on the passages above
"""

CONSENSUS_EXTRACTION_PROMPT = """Paper A [{id_a}]:
{chunks_a}

Paper B [{id_b}]:
{chunks_b}

List claims that both papers explicitly agree on.
Return a JSON array of strings. If there are none, return [].
"""

CONTRADICTION_EXTRACTION_PROMPT = """Paper A [{id_a}]:
{chunks_a}

Paper B [{id_b}]:
{chunks_b}

List claims where Paper A and Paper B take directly opposing positions.
Return a JSON array: [{{"claim": "", "position_a": "", "position_b": ""}}]. If there are none, return [].
"""

TOPIC_DECOMPOSITION_PROMPT = """You are a research strategist. Given a research topic, identify the two distinct domains it spans.

Topic: {topic}

Domain A is the methods/techniques side — the approaches, algorithms, or tools that researchers develop or apply.
Domain B is the problems/applications side — the challenges, tasks, or application areas that need to be solved.

Think carefully about where the natural boundary lies. Do not split arbitrarily or pick domains that are too broad to be useful.

Return only valid JSON. No explanation, no markdown, no code fences.
{{"domain_a": "...", "domain_b": "...", "rationale": "..."}}
"""

EXTRACTION_PROMPT = """You are a research analyst. Read the following chunks from paper [{arxiv_id}] and extract structured information.

Domain context: {domain}

Chunks:
{chunks}

Extract:
- methods: techniques, algorithms, or approaches this paper introduces or applies
- problems: specific challenges or tasks this paper addresses

Be concrete. Prefer specific names over vague descriptions (e.g. "quantization-aware training" not "a training technique").

Return only valid JSON. No explanation, no markdown, no code fences.
{{"methods": ["...", "..."], "problems": ["...", "..."]}}
"""

GAP_DETECTION_PROMPT = """You are a research gap analyst.

Methods extracted from the literature:
{methods_list}

Problems identified in the literature:
{problems_list}

Find combinations where a method has clearly not been applied to a problem and doing so would be non-trivial but plausible. Ignore combinations that are already well-studied or where the connection is too obvious to be interesting. Return at most 6 gaps.

Return only valid JSON. No explanation, no markdown, no code fences.
{{"gaps": [{{"method": "...", "problem": "...", "reasoning": "..."}}]}}
"""

HYPOTHESIS_GENERATION_PROMPT = """You are a research hypothesis generator.

Gap to investigate:
{gap}

Supporting evidence from the literature:
{supporting_chunks}

Generate a concrete, testable hypothesis about what would happen if this method were applied to this problem. Include:
- hypothesis: a clear, falsifiable statement of the predicted outcome
- mechanism: why this method could plausibly work on this problem at a technical level
- challenge: the main obstacle that would need to be overcome
- feasibility_score: a float from 0.0 to 1.0 reflecting how achievable this is given current techniques
- score_rationale: one or two sentences justifying the score

Return only valid JSON. No explanation, no markdown, no code fences.
{{"hypothesis": "...", "mechanism": "...", "challenge": "...", "feasibility_score": 0.0, "score_rationale": "..."}}
"""

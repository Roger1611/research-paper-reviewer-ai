# Autonomous Research Synthesis Agent

Give it a research topic; it fetches papers from ArXiv, extracts methods and open problems, identifies research gaps, generates scored hypotheses, and produces a structured report.

## How it works

The pipeline is a 6-node LangGraph `StateGraph`. Each node updates a shared `AgentState` dict.

1. **decompose_topic** — splits the topic into two domains: a methods/techniques side (Domain A) and a problems/applications side (Domain B).

2. **search_domains** — runs `arxiv_search` separately for each domain, filters out off-topic results by embedding similarity (cosine threshold 0.25), then downloads and indexes each surviving paper's full text into a per-paper FAISS store.

3. **extract_knowledge** — for each paper, calls an LLM to extract the specific methods it introduces and the problems it addresses. Results are accumulated and deduplicated across all papers.

4. **detect_gaps** — sends the full methods list and problems list to an LLM and asks it to find non-obvious combinations where a method hasn't been applied to a problem.

5. **generate_hypotheses** — for each gap, retrieves supporting chunks from the paper store and asks an LLM to generate a concrete, mechanistically grounded hypothesis with a feasibility score.

6. **conditional loop** — if the top hypothesis score is below `MIN_CONFIDENCE_THRESHOLD` and the iteration limit hasn't been reached, loops back to step 5. Otherwise proceeds to `build_report`.

## Project layout

```
cli.py                  Entry point
agent/
  orchestrator.py       StateGraph definition and run_research_agent()
  state.py              AgentState TypedDict
  backend.py            LLMBackend — wraps OpenRouter and Ollama behind one interface
  prompts.py            All prompt strings
  tools/
    arxiv_tool.py       arxiv_search — fetches paper metadata
    pdf_tool.py         fetch_paper_text — downloads, chunks, indexes PDFs
    extract_tool.py     extract_methods_problems — per-paper LLM extraction
    gap_tool.py         detect_gaps — cross-domain gap analysis
    hypothesis_tool.py  generate_hypotheses — scored hypothesis generation
    synthesis_tool.py   synthesize_papers — legacy ollama pipeline only
core/
  chunker.py            Sentence-aware text chunking
  embedder.py           SentenceTransformer embeddings (all-MiniLM-L6-v2)
  retriever.py          FAISS index creation and search
  loader.py             PDF text extraction
  json_utils.py         parse_json — shared fence-stripping JSON parser
report/
  builder.py            Assembles and validates the final report dict
  formatter.py          Renders the report dict to markdown
config.py               All constants
```

## Backend comparison

| | OpenRouter (default) | Ollama |
|---|---|---|
| Quality | Higher | Lower |
| Cost | API credits | Free |
| Setup | API key in `.env` | Ollama running locally |
| Flag | _(default)_ | `--backend ollama` |

## Setup

```bash
pip install -r requirements.txt

# OpenRouter (default)
cp .env.example .env
# edit .env and add your OPENROUTER_API_KEY

# Ollama (optional)
# uncomment langchain-ollama in requirements.txt, then:
ollama pull llama3.1:8b-instruct-q4_K_M
```

## Run

```bash
python cli.py "sensor fusion for autonomous driving"
python cli.py "sensor fusion for autonomous driving" --backend ollama
```

## Output schema

```json
{
  "topic": "...",
  "papers": [{"title": "", "authors": "", "arxiv_id": "", "url": ""}],
  "hypotheses": [
    {
      "hypothesis": "...",
      "mechanism": "...",
      "challenge": "...",
      "feasibility_score": 0.72,
      "score_rationale": "...",
      "citations": ["2301.xxxxx"]
    }
  ],
  "consensus": [{"finding": "", "confidence": 0.85, "citations": ["2301.xxxxx"]}],
  "contested": [{"claim": "", "positions": ["stance A", "stance B"], "citations": []}],
  "open_problems": ["..."],
  "synthesis_narrative": "...",
  "generated_at": "2026-..."
}
```

## Config

| Constant | Default |
|---|---|
| `FAST_MODEL` | `google/gemini-2.0-flash-lite-001` |
| `STRONG_MODEL` | `deepseek/deepseek-chat` |
| `OLLAMA_FAST_MODEL` | `llama3.1:8b-instruct-q4_K_M` |
| `ARXIV_MAX_RESULTS` | `6` |
| `MAX_HYPOTHESIS_ITERATIONS` | `2` |
| `MIN_CONFIDENCE_THRESHOLD` | `0.35` |
| `HYPOTHESIS_TOP_K` | `5` |
| `CHUNK_SIZE` | `220` words |
| `SYNTHESIS_TOP_K` | `3` chunks per paper |

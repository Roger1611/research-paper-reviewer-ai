# Autonomous Research Synthesis Agent

Fetches papers from ArXiv, reads their full text, cross-references findings, and produces a structured synthesis report — from a single topic string.

## How it works

1. `arxiv_search` — fetches the top-N most relevant papers from ArXiv (no API key needed)
2. `fetch_paper_text` — downloads each PDF, chunks it, and builds a per-paper FAISS index
3. `synthesize_papers` — queries all indexes, calls a local Ollama LLM, returns structured JSON

A LangChain `AgentExecutor` with ReAct reasoning decides the order and repetition of tool calls. It loads at least 4 papers before synthesizing, identifies both consensus findings and contested claims, and scores confidence via cosine similarity between the topic embedding and supporting chunk embeddings.

## Project layout

```
cli.py                  Entry point — runs the agent and prints the report
agent/
  orchestrator.py       AgentExecutor setup and run_research_agent()
  memory.py             ConversationSummaryMemory backed by Ollama
  prompts.py            All prompt strings (ReAct, synthesis, consensus, contradiction)
  tools/
    arxiv_tool.py       arxiv_search — fetches paper metadata
    pdf_tool.py         fetch_paper_text — downloads, chunks, indexes PDFs
    synthesis_tool.py   synthesize_papers — cross-paper synthesis and confidence scoring
core/
  chunker.py            Sentence-aware text chunking
  embedder.py           SentenceTransformer embeddings (all-MiniLM-L6-v2)
  retriever.py          FAISS index creation and search
  loader.py             PDF and plain text extraction
report/
  builder.py            Validates and assembles the final report dict
  formatter.py          Renders the report dict to markdown
config.py               All constants
```

## Setup

```bash
pip install -r requirements.txt
ollama pull llama3.1:8b-instruct-q4
```

## Run

```bash
ollama serve   # if not already running
python cli.py "diffusion models for protein structure prediction"
```

## Output schema

```json
{
  "topic": "...",
  "papers": [{"title": "", "authors": "", "arxiv_id": "", "url": ""}],
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
| `OLLAMA_MODEL` | `llama3.1:8b-instruct-q4` |
| `ARXIV_MAX_RESULTS` | `6` |
| `CHUNK_SIZE` | `220` words |
| `CHUNK_OVERLAP` | `40` words |
| `SYNTHESIS_TOP_K` | `3` chunks per paper |

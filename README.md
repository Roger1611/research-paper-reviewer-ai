# Autonomous Research Synthesis Agent

Fetches papers from ArXiv, reads their full text, cross-references findings, and produces a structured synthesis report — from a single topic string.

## How it works

1. **Search** — a LangGraph ReAct agent calls `arxiv_search` and returns paper metadata. Papers whose title+abstract embed below 0.25 cosine similarity with the topic are dropped before any PDFs are downloaded.

2. **Fetch** — the orchestrator calls `fetch_paper_text` directly for each surviving paper. Each PDF is downloaded, chunked, and indexed in a per-paper FAISS store.

3. **Synthesize** — the orchestrator calls `synthesize_papers` directly (not via the LLM). The top-K chunks from each paper are retrieved and passed to a local Ollama LLM with a structured prompt asking for consensus findings, contested claims, and open problems in JSON form.

4. **Score** — each consensus finding is scored by cosine similarity between the finding text embedding and the chunks that cited it (or all retrieved chunks if no citations). Hallucinated citation IDs are stripped; version-less IDs (e.g. `2405.09820`) are resolved to their versioned form.

5. **Render** — the JSON synthesis is merged with ArXiv metadata and formatted to markdown.

## Project layout

```
cli.py                  Entry point — runs the agent and prints the report
agent/
  orchestrator.py       LangGraph agent for search; direct orchestration of fetch + synthesis
  memory.py             Stub — in-run state is handled by LangGraph message graph
  prompts.py            All prompt strings
  tools/
    arxiv_tool.py       arxiv_search — fetches paper metadata from ArXiv
    pdf_tool.py         fetch_paper_text — downloads, chunks, and indexes PDFs
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
ollama pull llama3.1:8b-instruct-q4_K_M
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
| `OLLAMA_MODEL` | `llama3.1:8b-instruct-q4_K_M` |
| `ARXIV_MAX_RESULTS` | `6` |
| `CHUNK_SIZE` | `220` words |
| `CHUNK_OVERLAP` | `40` words |
| `SYNTHESIS_TOP_K` | `3` chunks per paper |

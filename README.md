# Research Paper Reviewer

This is a simple student-level project for reviewing research papers using:

- Streamlit
- sentence-transformers
- FAISS
- Ollama

The app lets you:

- upload a PDF or text file
- extract the text
- split it into chunks
- create embeddings
- store embeddings in FAISS
- ask questions about the document
- get answers from a local Ollama model

## Project Files

- `app.py` - Streamlit UI and full pipeline
- `loader.py` - loads PDF and text files
- `chunker.py` - splits text into simple chunks
- `embedder.py` - creates embeddings
- `retriever.py` - stores and searches chunks with FAISS
- `requirements.txt` - Python dependencies
- `README.md` - project instructions

## Install

```bash
pip install -r requirements.txt
```

## Ollama Model

Make sure Ollama is running locally and the model is available:

```bash
ollama run llama3.1:8b-instruct-q4
```

The app sends requests to:

```text
http://localhost:11434/api/generate
```

## Run the App

```bash
streamlit run app.py
```

## How It Works

1. Upload a PDF or text file.
2. The app extracts the text.
3. The text is split into simple fixed-size chunks.
4. Each chunk is converted into an embedding.
5. The embeddings are stored in FAISS.
6. When you ask a question, the app retrieves the most relevant chunks.
7. The retrieved chunks are sent to Ollama.
8. The model answers using only the document context.

## Notes

- Chunking is simple and beginner-friendly.
- The app returns top matching chunks from FAISS.
- If the answer is not in the document, the prompt tells the model to say `Not found`.

import requests
import streamlit as st

from loader import load_file
from chunker import split_text
from embedder import get_embeddings
from retriever import create_faiss_index, search_chunks


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b-instruct-q4"


def ask_ollama(question, context):
    # Build a simple prompt using the retrieved chunks.
    prompt = f"""You are an AI research assistant.
Answer ONLY using the context below.
If not found, say 'Not found'.

Context:
{context}

Question:
{question}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    return data.get("response", "No response received.")


st.set_page_config(page_title="Research Paper Reviewer", layout="wide")
st.title("Research Paper Reviewer")

uploaded_file = st.file_uploader("Upload a PDF or text file", type=["pdf", "txt"])
question = st.text_input("Ask a question about the paper")
show_chunks = st.checkbox("Show retrieved chunks")

if uploaded_file:
    try:
        # Load the full document text.
        document_text = load_file(uploaded_file)

        if not document_text.strip():
            st.error("The file was loaded, but no text was found.")
        else:
            # Split the document into simple word-based chunks.
            chunks = split_text(document_text, chunk_size=400)
            st.success(f"File loaded successfully. Created {len(chunks)} chunks.")

            # Create embeddings and FAISS index once per upload.
            chunk_embeddings = get_embeddings(chunks)
            index = create_faiss_index(chunk_embeddings)

            if st.button("Ask"):
                if not question.strip():
                    st.warning("Please enter a question.")
                else:
                    question_embedding = get_embeddings([question])
                    top_chunks = search_chunks(
                        index=index,
                        chunks=chunks,
                        query_embedding=question_embedding,
                        top_k=4,
                    )

                    context = "\n\n".join(top_chunks)
                    answer = ask_ollama(question, context)

                    st.subheader("Answer")
                    st.write(answer)

                    if show_chunks:
                        st.subheader("Retrieved Chunks")
                        for i, chunk in enumerate(top_chunks, start=1):
                            st.markdown(f"**Chunk {i}:**")
                            st.write(chunk)

    except Exception as error:
        st.error(f"Error: {error}")

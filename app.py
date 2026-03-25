import requests
import streamlit as st

from loader import load_bytes
from chunker import split_text
from embedder import get_embeddings
from retriever import create_faiss_index, search_chunks


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b-instruct-q4"

CHUNK_SIZE = 220
CHUNK_OVERLAP = 40


@st.cache_data(show_spinner=False)
def load_document(file_bytes, file_name, file_type):
    return load_bytes(file_bytes, file_name=file_name, file_type=file_type)


@st.cache_data(show_spinner=False)
def chunk_document(document_text):
    return split_text(
        document_text,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


@st.cache_resource(show_spinner=False)
def build_index(chunks):
    chunk_list = list(chunks)
    chunk_embeddings = get_embeddings(chunk_list)
    return create_faiss_index(chunk_embeddings)


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
        file_bytes = uploaded_file.getvalue()

        # Load the full document text.
        document_text = load_document(file_bytes, uploaded_file.name, uploaded_file.type)

        if not document_text.strip():
            st.error("The file was loaded, but no text was found.")
        else:
            # Split the document into overlapped, structure-aware chunks.
            chunks = chunk_document(document_text)
            st.success(f"File loaded successfully. Created {len(chunks)} chunks.")

            # Create embeddings and FAISS index once per unique chunk set.
            index = build_index(tuple(chunks))

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

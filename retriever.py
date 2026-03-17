import faiss
import numpy as np


def create_faiss_index(embeddings):
    # Build a simple FAISS index using L2 distance.
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype="float32"))
    return index


def search_chunks(index, chunks, query_embedding, top_k=4):
    # Search the most relevant chunks for the question.
    distances, indices = index.search(np.array(query_embedding, dtype="float32"), top_k)

    results = []
    for chunk_index in indices[0]:
        if 0 <= chunk_index < len(chunks):
            results.append(chunks[chunk_index])

    return results

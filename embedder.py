from sentence_transformers import SentenceTransformer


model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embeddings(text_list):
    # Convert text chunks or questions into embedding vectors.
    return model.encode(text_list, convert_to_numpy=True)

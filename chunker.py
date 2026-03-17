def split_text(text, chunk_size=400):
    # Split text into simple fixed-size word chunks.
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        if chunk_text.strip():
            chunks.append(chunk_text)

    return chunks

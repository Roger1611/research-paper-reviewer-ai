import re


_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")


def _normalize_text(text):
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


def _split_paragraph(paragraph, chunk_size):
    words = paragraph.split()
    if len(words) <= chunk_size:
        return [paragraph]

    sentences = [sentence.strip() for sentence in _SENTENCE_BOUNDARY_RE.split(paragraph) if sentence.strip()]
    if len(sentences) <= 1:
        return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

    parts = []
    current_sentences = []
    current_size = 0

    for sentence in sentences:
        sentence_words = sentence.split()
        sentence_size = len(sentence_words)

        if sentence_size > chunk_size:
            if current_sentences:
                parts.append(" ".join(current_sentences))
                current_sentences = []
                current_size = 0

            parts.extend(
                " ".join(sentence_words[i:i + chunk_size])
                for i in range(0, sentence_size, chunk_size)
            )
            continue

        if current_size + sentence_size <= chunk_size:
            current_sentences.append(sentence)
            current_size += sentence_size
            continue

        parts.append(" ".join(current_sentences))
        current_sentences = [sentence]
        current_size = sentence_size

    if current_sentences:
        parts.append(" ".join(current_sentences))

    return parts


def split_text(text, chunk_size=220, chunk_overlap=40):
    # Build chunks from paragraphs and sentences so embeddings keep more context.
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be between 0 and chunk_size - 1.")

    normalized_text = _normalize_text(text)
    if not normalized_text:
        return []

    paragraphs = [paragraph for paragraph in re.split(r"\n\s*\n+", normalized_text) if paragraph]
    chunks = []
    current_words = []

    for paragraph in paragraphs:
        for part in _split_paragraph(paragraph, chunk_size):
            part_words = part.split()
            if not part_words:
                continue

            if current_words and len(current_words) + len(part_words) > chunk_size:
                chunks.append(" ".join(current_words))
                overlap_words = current_words[-chunk_overlap:] if chunk_overlap else []
                max_overlap = max(chunk_size - len(part_words), 0)
                current_words = overlap_words[-max_overlap:] if overlap_words else []

            current_words.extend(part_words)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks

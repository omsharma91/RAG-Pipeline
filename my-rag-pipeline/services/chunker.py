from typing import List


def sliding_window_chunking(text: str, chunk_size: int = 300, chunk_overlap: int = 10) -> List[str]:
    """Split text into word-based sliding window chunks.

    Args:
        text: The input text to chunk.
        chunk_size: Number of words per chunk.
        chunk_overlap: Number of words to overlap between chunks.

    Returns:
        A list of chunk strings.
    """
    if not text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")

    if chunk_size <= chunk_overlap:
        raise ValueError("chunk_size must be larger than chunk_overlap")

    words = text.split()
    step = chunk_size - chunk_overlap
    chunks: List[str] = []

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break

    return chunks

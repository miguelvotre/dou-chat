"""
Hierarchical chunking for DOU acts.

Strategy: parent document retrieval
- Full act is stored in the 'atos' table (parent document)
- ~512-token chunks with 50-word overlap are stored in the 'chunks' tables
- Each chunk carries all metadata from its parent act
"""

from ingestion.parser import Ato

CHUNK_SIZE = 512   # approximate tokens (words * 1.3)
OVERLAP = 50       # word overlap between chunks


def _split_words(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks


def chunk_ato(ato: Ato, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """
    Splits the full text of an act into chunks.
    Prefixes each chunk with title and issuing body to improve embedding recall.
    """
    prefix = f"[{ato.orgao}] {ato.titulo}"
    body = ato.texto_completo

    raw_chunks = _split_words(body, chunk_size, overlap)

    if not raw_chunks:
        return [prefix] if prefix.strip() else []

    # Full prefix on first chunk; subsequent chunks get a shorter context header
    result = []
    for i, chunk in enumerate(raw_chunks):
        if i == 0:
            result.append(f"{prefix}\n\n{chunk}")
        else:
            result.append(f"[{ato.orgao}]\n\n{chunk}")

    return result

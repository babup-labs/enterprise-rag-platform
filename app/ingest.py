import hashlib

from .chunking import chunk_text, load_text
from .embeddings import embed
from .vectorstore import get_collection


def ingest_document(filename: str, data: bytes) -> int:
    """Load, chunk, embed and store one document. Re-ingesting the same file replaces it."""
    chunks = chunk_text(load_text(filename, data))
    if not chunks:
        raise ValueError("No text could be extracted from this document.")
    doc_id = hashlib.sha1(filename.encode()).hexdigest()[:12]
    col = get_collection()
    col.delete(where={"doc_id": doc_id})
    col.add(
        ids=[f"{doc_id}-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
        metadatas=[{"doc_id": doc_id, "source": filename, "chunk": i} for i in range(len(chunks))],
    )
    return len(chunks)

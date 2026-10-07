import chromadb
from .config import settings

_client = None


def get_collection():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_dir)
    return _client.get_or_create_collection("documents", metadata={"hnsw:space": "cosine"})

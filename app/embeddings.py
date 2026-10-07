from openai import OpenAI
from .config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


def embed(texts: list[str]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), 96):
        resp = _get_client().embeddings.create(model=settings.embedding_model, input=texts[i:i + 96])
        vectors.extend(d.embedding for d in resp.data)
    return vectors

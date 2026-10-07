from openai import OpenAI

from .config import settings
from .embeddings import embed
from .vectorstore import get_collection

SYSTEM_PROMPT = (
    "You answer questions using ONLY the numbered context passages provided. "
    "Cite passages like [1] or [2]. If the context does not contain the answer, "
    "say you could not find it in the documents. Do not use outside knowledge."
)


def retrieve(question: str, k: int | None = None) -> list[dict]:
    col = get_collection()
    if col.count() == 0:
        return []
    res = col.query(query_embeddings=embed([question]), n_results=k or settings.top_k)
    hits = []
    for text, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        if dist <= settings.max_distance:  # drop weak matches
            hits.append({"text": text, "source": meta["source"], "chunk": meta["chunk"], "distance": round(dist, 4)})
    return hits


def answer(question: str, k: int | None = None) -> dict:
    hits = retrieve(question, k)
    if not hits:
        return {"answer": "I could not find relevant information in the ingested documents.", "sources": []}
    context = "\n\n".join(f"[{i + 1}] ({h['source']}) {h['text']}" for i, h in enumerate(hits))
    resp = OpenAI().chat.completions.create(
        model=settings.chat_model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
    )
    return {"answer": resp.choices[0].message.content, "sources": hits}

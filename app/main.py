from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .ingest import ingest_document
from .rag import answer
from .vectorstore import get_collection

app = FastAPI(title="Enterprise AI RAG Platform", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    top_k: int | None = Field(default=None, ge=1, le=10)


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": get_collection().count()}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    try:
        n = ingest_document(file.filename or "upload.txt", await file.read())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"filename": file.filename, "chunks_added": n}


@app.post("/ask")
def ask(req: AskRequest):
    return answer(req.question, req.top_k)

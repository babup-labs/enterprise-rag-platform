# Enterprise AI RAG Platform

An enterprise knowledge assistant that answers questions from your own documents using retrieval-augmented generation (RAG). Answers are grounded in retrieved passages and cite their sources, and the assistant says so when the documents don't contain the answer.

**Stack:** Python · FastAPI · OpenAI · LangChain text splitters · ChromaDB · Docker

## Architecture

```text
Documents (.pdf / .txt / .md)
        ↓
Document loading        app/chunking.py
        ↓
Chunking                RecursiveCharacterTextSplitter
        ↓
Embeddings              app/embeddings.py  (OpenAI)
        ↓
Vector database         app/vectorstore.py (ChromaDB, cosine)
        ↓
Semantic retrieval      app/rag.py  (top-k + distance threshold)
        ↓
Context assembly        numbered, source-labelled passages
        ↓
LLM                     answers only from the context, with citations
        ↓
Grounded response       { answer, sources[] }
```

## Quick start

```bash
git clone https://github.com/<your-username>/enterprise-rag-platform.git
cd enterprise-rag-platform
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                  # then add your OPENAI_API_KEY
uvicorn app.main:app --reload
```

Open http://localhost:8000/docs for the interactive API.

```bash
# 1. Ingest the sample document
curl -F "file=@data/sample/company_handbook.md" http://localhost:8000/ingest

# 2. Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How many remote days per week are allowed?"}'
```

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service status and number of indexed chunks |
| POST | `/ingest` | Upload a `.pdf`, `.txt` or `.md` file to index it |
| POST | `/ask` | Ask a question; returns an answer and the source passages |

## Docker

```bash
docker build -t enterprise-rag-platform .
docker run -p 8000:8000 --env-file .env enterprise-rag-platform
```

## Configuration

All settings live in `.env` (see `.env.example`): models, chunk size and overlap, `TOP_K`, and `MAX_DISTANCE` (lower is stricter about relevance).

## Tests

```bash
pytest
```

The tests cover loading and chunking and run without an API key.

## Roadmap

- Streaming responses
- Hybrid (keyword + vector) search and re-ranking
- Per-user access control on documents
- Retrieval evaluation set (hit rate / MRR)
- Swap ChromaDB for Pinecone or FAISS behind the same interface

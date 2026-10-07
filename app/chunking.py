"""Document loading and chunking (no network or API calls)."""
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import settings


def load_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith((".txt", ".md")):
        return data.decode("utf-8", errors="ignore")
    raise ValueError("Unsupported file type. Use .pdf, .txt or .md")


def chunk_text(text: str, size: int | None = None, overlap: int | None = None) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size or settings.chunk_size,
        chunk_overlap=overlap if overlap is not None else settings.chunk_overlap,
    )
    return [c.strip() for c in splitter.split_text(text) if c.strip()]

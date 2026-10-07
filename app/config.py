import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    chroma_dir: str = os.getenv("CHROMA_DIR", "./chroma_db")
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    max_distance: float = float(os.getenv("MAX_DISTANCE", "0.6"))


settings = Settings()

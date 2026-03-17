from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _split_csv(value: str | None, default: tuple[str, ...] = ()) -> tuple[str, ...]:
    if value is None:
        return default
    parts = tuple(part.strip() for part in value.split(",") if part.strip())
    return parts or default


@dataclass(slots=True)
class Settings:
    project_root: Path
    index_path: Path
    embeddings_path: Path
    top_k: int = 5
    google_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    query_rewrite_model: str = "gemini-2.5-flash"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    enable_query_rewrite: bool = True
    frontend_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )

    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        resolved_project_root = project_root or Path.cwd()
        load_dotenv(resolved_project_root / ".env", override=False)

        return cls(
            project_root=resolved_project_root,
            index_path=resolved_project_root / os.getenv("RAG_INDEX_PATH", "data/index/rag_index.json"),
            embeddings_path=resolved_project_root / os.getenv(
                "RAG_EMBEDDINGS_PATH",
                "data/index/rag_embeddings.npz",
            ),
            top_k=int(os.getenv("RAG_TOP_K", "5")),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            query_rewrite_model=os.getenv(
                "QUERY_REWRITE_MODEL",
                os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            ),
            embedding_model=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            ),
            enable_query_rewrite=_as_bool(os.getenv("ENABLE_QUERY_REWRITE"), default=True),
            frontend_origins=_split_csv(
                os.getenv("FRONTEND_ORIGINS"),
                default=("http://localhost:5173", "http://127.0.0.1:5173"),
            ),
        )

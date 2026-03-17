from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class Settings:
    project_root: Path
    index_path: Path
    embeddings_path: Path
    top_k: int = 5
    google_api_key: str | None = None

    @classmethod
    def from_env(cls) -> "Settings":
        project_root = Path.cwd()
        index_path = project_root / os.getenv("RAG_INDEX_PATH", "data/index/rag_index.json")
        embeddings_path = project_root / os.getenv(
            "RAG_EMBEDDINGS_PATH", "data/index/rag_embeddings.npz"
        )
        top_k = int(os.getenv("RAG_TOP_K", "5"))
        google_api_key = os.getenv("GOOGLE_API_KEY")
        return cls(
            project_root=project_root,
            index_path=index_path,
            embeddings_path=embeddings_path,
            top_k=top_k,
            google_api_key=google_api_key,
        )

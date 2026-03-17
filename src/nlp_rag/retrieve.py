from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass(slots=True)
class SearchResult:
    score: float
    source: str
    text: str
    metadata: dict[str, str]


class RagRetriever:
    def __init__(
        self,
        index_path: Path,
        embeddings_path: Path,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    ) -> None:
        self._records = json.loads(index_path.read_text(encoding="utf-8"))
        self._embeddings = np.load(embeddings_path)["embeddings"]
        self._embedder = SentenceTransformer(model_name)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        query_embedding = self._embedder.encode([query], normalize_embeddings=True)[0]
        scores = self._embeddings @ query_embedding
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            record = self._records[int(idx)]
            results.append(
                SearchResult(
                    score=float(scores[int(idx)]),
                    source=record["source"],
                    text=record["text"],
                    metadata=record["metadata"],
                )
            )
        return results

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


WORD_RE = re.compile(r"\w+", flags=re.UNICODE)


@dataclass(slots=True)
class SearchResult:
    score: float
    source: str
    text: str
    metadata: dict[str, str]


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value.lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^\w\s]", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _tokenize(value: str) -> set[str]:
    return set(WORD_RE.findall(_normalize_text(value)))


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
        self._lexical_docs = [
            _tokenize(f"{record['text']} {' '.join(record['metadata'].values())}") for record in self._records
        ]

    def search(self, query: str, top_k: int = 5, alpha: float = 0.78) -> list[SearchResult]:
        query_embedding = self._embedder.encode([query], normalize_embeddings=True)[0]
        semantic_scores = self._embeddings @ query_embedding
        semantic_scores = (semantic_scores + 1.0) / 2.0

        query_tokens = _tokenize(query)
        lexical_scores = np.zeros(len(self._records), dtype=np.float32)
        if query_tokens:
            for idx, doc_tokens in enumerate(self._lexical_docs):
                overlap = len(query_tokens & doc_tokens)
                if overlap:
                    lexical_scores[idx] = overlap / max(len(query_tokens), 1)

        blended = alpha * semantic_scores + (1.0 - alpha) * lexical_scores
        top_indices = np.argsort(blended)[::-1][:top_k]

        results: list[SearchResult] = []
        for idx in top_indices:
            record = self._records[int(idx)]
            metadata = dict(record["metadata"])
            metadata["semantic_score"] = f"{semantic_scores[int(idx)]:.4f}"
            metadata["lexical_score"] = f"{lexical_scores[int(idx)]:.4f}"
            results.append(
                SearchResult(
                    score=float(blended[int(idx)]),
                    source=record["source"],
                    text=record["text"],
                    metadata=metadata,
                )
            )
        return results

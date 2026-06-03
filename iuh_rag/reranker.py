"""Cross-encoder reranker module.

Optional component — if the cross-encoder / sentence-transformers package is
unavailable or the model cannot be downloaded, the reranker silently falls back
to the lexical overlap score that already exists in retrievers.py.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .config import DEFAULT_EMBEDDING_DEVICE
from .device import resolve_torch_device
from .retrievers import SearchResult, lexical_rerank
from .text import token_overlap_score


class CrossEncoderReranker:
    """Wraps a cross-encoder model for reranking retrieved evidence.

    Graceful fallback: if the model is not available, `rerank()` calls
    `lexical_rerank()` from `retrievers.py` instead of crashing.

    Supported model names (HuggingFace):
        - ``BAAI/bge-reranker-base``
        - ``BAAI/bge-reranker-large``
        - ``cross-encoder/ms-marco-MiniLM-L-6-v2``
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        batch_size: int = 16,
        max_length: int = 512,
        device: str = DEFAULT_EMBEDDING_DEVICE,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.requested_device = device
        self.device = resolve_torch_device(device)
        self.model = None
        self.error: Optional[str] = None
        self._load_model()

    def _load_model(self) -> bool:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            try:
                self.model = CrossEncoder(self.model_name, max_length=self.max_length, device=self.device)
            except TypeError:
                self.model = CrossEncoder(self.model_name, max_length=self.max_length)
                if hasattr(self.model, "model"):
                    self.model.model.to(self.device)
            self.error = None
            return True
        except ImportError:
            self.error = "sentence-transformers not installed."
            return False
        except Exception as exc:
            self.error = str(exc)
            return False

    def _score_pairs(self, query: str, results: Sequence[SearchResult]) -> List[float]:
        """Return cross-encoder scores for (query, chunk_text) pairs."""
        if self.model is None:
            return [token_overlap_score(query, r.chunk.get("text", "")) for r in results]
        pairs = [(query, r.chunk.get("text", "")[:self.max_length]) for r in results]
        raw_scores: Any = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        # CrossEncoder returns ndarray or list
        if hasattr(raw_scores, "tolist"):
            return raw_scores.tolist()
        return list(raw_scores)

    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        top_k: int = 5,
    ) -> List[SearchResult]:
        """Rerank results using cross-encoder scores.

        Falls back to lexical rerank if model is unavailable.
        """
        if not results:
            return []
        if self.model is None:
            return lexical_rerank(query, results, top_k=top_k)

        scores = self._score_pairs(query, results)
        reranked = []
        for result, score in zip(results, scores):
            result = SearchResult(
                chunk=result.chunk,
                score=float(score),
                rank=result.rank,
                source=result.source,
                sources=result.sources,
            )
            reranked.append(result)
        ordered = sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]
        for rank, item in enumerate(ordered, start=1):
            item.rank = rank
        return ordered

    def status(self) -> Dict[str, Any]:
        model_device = None
        if self.model is not None and hasattr(self.model, "model"):
            try:
                model_device = str(next(self.model.model.parameters()).device)
            except Exception:
                model_device = None
        return {
            "model_name": self.model_name,
            "requested_device": self.requested_device,
            "resolved_device": self.device,
            "model_device": model_device,
            "loaded": self.model is not None,
            "error": self.error,
        }

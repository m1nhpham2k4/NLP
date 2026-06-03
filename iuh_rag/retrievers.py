from __future__ import annotations

import atexit
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_QDRANT_COLLECTION_PREFIX,
    DEFAULT_QDRANT_MODE,
    DEFAULT_QDRANT_PATH,
    DEFAULT_QDRANT_URL,
    INDEX_DIR,
)
from .device import resolve_torch_device
from .text import normalize_text, tokenize, token_overlap_score

try:
    from langchain_core.embeddings import Embeddings as LangChainEmbeddings
except Exception:
    class LangChainEmbeddings:  # type: ignore[no-redef]
        pass


@dataclass
class SearchResult:
    chunk: Dict[str, Any]
    score: float
    rank: int
    source: str
    sources: List[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return self.chunk["chunk_id"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "rank": self.rank,
            "source": self.source,
            "sources": self.sources or [self.source],
            "file_name": self.chunk.get("file_name"),
            "relative_path": self.chunk.get("relative_path"),
            "type": self.chunk.get("type"),
            "text": self.chunk.get("text"),
        }


def chunk_matches_filters(chunk: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    if not filters:
        return True
    metadata = {**chunk, **chunk.get("metadata", {})}
    for key, expected in filters.items():
        if expected in (None, "", [], set()):
            continue
        actual = metadata.get(key)
        if key == "year" and expected in metadata.get("years", []):
            continue
        if isinstance(actual, (list, tuple, set)):
            if isinstance(expected, (list, tuple, set)):
                if not set(expected) & set(actual):
                    return False
            elif expected not in actual:
                return False
            continue
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


class BaseRetriever:
    name = "base"

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        raise NotImplementedError


class TfidfRetriever(BaseRetriever):
    name = "tfidf"

    def __init__(self, chunks: Sequence[Dict[str, Any]], analyzer: str = "char_wb") -> None:
        self.chunks = list(chunks)
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer,
            ngram_range=(2, 5) if analyzer.startswith("char") else (1, 2),
            lowercase=False,
            min_df=1,
        )
        corpus = [normalize_text(chunk["text"]) for chunk in self.chunks]
        self.matrix = self.vectorizer.fit_transform(corpus)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        query_vec = self.vectorizer.transform([normalize_text(query)])
        scores = cosine_similarity(query_vec, self.matrix)[0]
        order = np.argsort(scores)[::-1]
        results: List[SearchResult] = []
        for idx in order:
            chunk = self.chunks[int(idx)]
            if scores[idx] <= 0:
                break
            if not chunk_matches_filters(chunk, filters):
                continue
            results.append(SearchResult(chunk=chunk, score=float(scores[idx]), rank=len(results) + 1, source=self.name))
            if len(results) >= top_k:
                break
        return results


class BM25Retriever(BaseRetriever):
    name = "bm25"

    def __init__(self, chunks: Sequence[Dict[str, Any]]) -> None:
        self.chunks = list(chunks)
        self.tokenized_corpus = [tokenize(chunk["text"]) for chunk in self.chunks]
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.tokenized_corpus)
        except Exception:
            self.bm25 = None

    def _fallback_scores(self, query_tokens: List[str]) -> np.ndarray:
        query_set = set(query_tokens)
        scores = []
        for tokens in self.tokenized_corpus:
            if not tokens:
                scores.append(0.0)
                continue
            token_set = set(tokens)
            scores.append(len(query_set & token_set) / max(1, len(query_set)))
        return np.array(scores, dtype=float)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens) if self.bm25 is not None else self._fallback_scores(query_tokens)
        order = np.argsort(scores)[::-1]
        results: List[SearchResult] = []
        for idx in order:
            chunk = self.chunks[int(idx)]
            if scores[idx] <= 0:
                break
            if not chunk_matches_filters(chunk, filters):
                continue
            results.append(SearchResult(chunk=chunk, score=float(scores[idx]), rank=len(results) + 1, source=self.name))
            if len(results) >= top_k:
                break
        return results


class DenseEmbeddingRetriever(BaseRetriever):
    """Optional dense retriever. It is skipped cleanly if a model is unavailable."""

    def __init__(
        self,
        chunks: Sequence[Dict[str, Any]],
        model_name: str,
        name: Optional[str] = None,
        cache_dir: Path = INDEX_DIR,
        batch_size: int = 32,
        device: str = DEFAULT_EMBEDDING_DEVICE,
    ) -> None:
        self.chunks = list(chunks)
        self.model_name = model_name
        self.name = name or f"dense:{model_name}"
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.requested_device = device
        self.device = self._resolve_device(device)
        self.model = None
        self.embeddings: Optional[np.ndarray] = None
        self.error: Optional[str] = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        return resolve_torch_device(device)

    def _cache_path(self) -> Path:
        digest = hashlib.sha1(self.model_name.encode("utf-8")).hexdigest()[:12]
        return self.cache_dir / f"dense_{digest}.joblib"

    def status(self) -> Dict[str, Any]:
        cache_path = self._cache_path()
        model_device = None
        if self.model is not None:
            try:
                model_device = str(self.model.device)
            except Exception:
                model_device = None
        return {
            "name": self.name,
            "model_name": self.model_name,
            "requested_device": self.requested_device,
            "resolved_device": self.device,
            "model_device": model_device,
            "cache_path": str(cache_path),
            "cache_exists": cache_path.exists(),
            "embeddings_loaded": self.embeddings is not None,
            "error": self.error,
        }

    def _passage_texts(self) -> List[str]:
        if "e5" in self.model_name.lower():
            return [f"passage: {chunk['text']}" for chunk in self.chunks]
        return [chunk["text"] for chunk in self.chunks]

    def _query_text(self, query: str) -> str:
        if "e5" in self.model_name.lower():
            return f"query: {query}"
        return query

    def _load_model(self, local_files_only: bool = False) -> bool:
        try:
            from sentence_transformers import SentenceTransformer

            try:
                self.model = SentenceTransformer(self.model_name, local_files_only=local_files_only, device=self.device)
            except TypeError:
                self.model = SentenceTransformer(self.model_name)
                if hasattr(self.model, "to"):
                    self.model.to(self.device)
            self.error = None
            return True
        except Exception as exc:
            self.error = str(exc)
            return False

    def _encode(self, texts: Sequence[str], show_progress_bar: bool = False) -> np.ndarray:
        try:
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress_bar,
                device=self.device,
            ).astype("float32")
        except TypeError:
            return self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=show_progress_bar,
            ).astype("float32")

    def build_index(self, force: bool = False) -> bool:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self._cache_path()
        if cache_path.exists() and not force:
            payload = joblib.load(cache_path)
            cached_embeddings = payload.get("embeddings")
            if cached_embeddings is not None and len(cached_embeddings) == len(self.chunks):
                self.embeddings = cached_embeddings
                self.error = None
                return True

        if not self._load_model(local_files_only=True) and not self._load_model(local_files_only=False):
            return False
        self.embeddings = self._encode(self._passage_texts(), show_progress_bar=True)
        joblib.dump({"model_name": self.model_name, "embeddings": self.embeddings}, cache_path)
        return True

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        if self.embeddings is None and self.error:
            return []
        if self.embeddings is None and not self.build_index(force=False):
            return []
        if self.model is None:
            if self.error:
                return []
            if not self._load_model(local_files_only=self._cache_path().exists()):
                return []

        query_embedding = self._encode([self._query_text(query)], show_progress_bar=False)[0]
        scores = np.dot(self.embeddings, query_embedding)
        order = np.argsort(scores)[::-1]
        results: List[SearchResult] = []
        for idx in order:
            if int(idx) >= len(self.chunks):
                continue
            chunk = self.chunks[int(idx)]
            if not chunk_matches_filters(chunk, filters):
                continue
            results.append(SearchResult(chunk=chunk, score=float(scores[idx]), rank=len(results) + 1, source=self.name))
            if len(results) >= top_k:
                break
        return results


def _qdrant_collection_name(prefix: str, retriever_name: str, model_name: str) -> str:
    key = retriever_name.split(":", 1)[-1] if retriever_name else model_name
    safe_key = re.sub(r"[^a-zA-Z0-9_-]+", "_", key).strip("_").lower() or "dense"
    digest = hashlib.sha1(model_name.encode("utf-8")).hexdigest()[:8]
    safe_prefix = re.sub(r"[^a-zA-Z0-9_-]+", "_", prefix).strip("_").lower() or "iuh_rag"
    return f"{safe_prefix}_{safe_key}_{digest}"


def _clean_qdrant_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): cleaned
            for key, item in value.items()
            if (cleaned := _clean_qdrant_payload(item)) is not None
        }
    if isinstance(value, (list, tuple, set)):
        return [
            cleaned
            for item in value
            if (cleaned := _clean_qdrant_payload(item)) is not None
        ]
    return str(value)


_QDRANT_LOCAL_CLIENTS: Dict[str, Any] = {}


def _close_qdrant_local_clients() -> None:
    for client in list(_QDRANT_LOCAL_CLIENTS.values()):
        try:
            client.close()
        except Exception:
            pass
    _QDRANT_LOCAL_CLIENTS.clear()


atexit.register(_close_qdrant_local_clients)


def _qdrant_local_client_key(path: str) -> str:
    return str(Path(path).expanduser().resolve())


class SentenceTransformerLangChainEmbeddings(LangChainEmbeddings):
    """LangChain embeddings adapter around the local SentenceTransformer retriever model."""

    def __init__(self, retriever: DenseEmbeddingRetriever) -> None:
        self.retriever = retriever

    def _ensure_model(self) -> bool:
        if self.retriever.model is not None:
            return True
        if self.retriever._load_model(local_files_only=True):
            return True
        return self.retriever._load_model(local_files_only=False)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not self._ensure_model():
            raise RuntimeError(self.retriever.error or "Cannot load embedding model.")
        prepared = [f"passage: {text}" for text in texts] if "e5" in self.retriever.model_name.lower() else texts
        return self.retriever._encode(prepared, show_progress_bar=True).tolist()

    def embed_query(self, text: str) -> List[float]:
        if not self._ensure_model():
            raise RuntimeError(self.retriever.error or "Cannot load embedding model.")
        return self.retriever._encode([self.retriever._query_text(text)], show_progress_bar=False)[0].tolist()


class QdrantDenseEmbeddingRetriever(DenseEmbeddingRetriever):
    """Dense retriever backed by LangChain QdrantVectorStore."""

    def __init__(
        self,
        chunks: Sequence[Dict[str, Any]],
        model_name: str,
        name: Optional[str] = None,
        batch_size: int = 32,
        device: str = DEFAULT_EMBEDDING_DEVICE,
        qdrant_mode: str = DEFAULT_QDRANT_MODE,
        qdrant_path: str = DEFAULT_QDRANT_PATH,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        collection_prefix: str = DEFAULT_QDRANT_COLLECTION_PREFIX,
    ) -> None:
        super().__init__(
            chunks=chunks,
            model_name=model_name,
            name=name,
            batch_size=batch_size,
            device=device,
        )
        self.qdrant_mode = qdrant_mode
        self.qdrant_path = qdrant_path
        self.qdrant_url = qdrant_url
        self.collection_prefix = collection_prefix
        self.collection_name = _qdrant_collection_name(collection_prefix, self.name, self.model_name)
        self.client = None
        self.vectorstore = None
        self.langchain_embeddings = SentenceTransformerLangChainEmbeddings(self)
        self.chunk_by_id = {chunk.get("chunk_id"): chunk for chunk in self.chunks}

    def _qdrant_location(self) -> str:
        if self.qdrant_mode == "local":
            return f"local path {self.qdrant_path}"
        return f"HTTP URL {self.qdrant_url}"

    def _qdrant_error(self, exc: Exception) -> str:
        if self.qdrant_mode == "local":
            return (
                f"Cannot use local embedded Qdrant at {self.qdrant_path}. "
                "Close other Python processes using the same qdrant_storage path, then retry. "
                f"Original error: {exc}"
            )
        return f"Cannot connect/use Qdrant at {self.qdrant_url}. Original error: {exc}"

    def _client(self):
        if self.client is not None:
            return self.client
        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:
            self.error = "qdrant-client is not installed. Run: pip install qdrant-client"
            raise RuntimeError(self.error) from exc
        try:
            if self.qdrant_mode == "local":
                client_key = _qdrant_local_client_key(self.qdrant_path)
                cached_client = _QDRANT_LOCAL_CLIENTS.get(client_key)
                if cached_client is not None:
                    try:
                        cached_client.get_collections()
                        self.client = cached_client
                        return self.client
                    except Exception:
                        _QDRANT_LOCAL_CLIENTS.pop(client_key, None)
                self.client = QdrantClient(path=self.qdrant_path)
                _QDRANT_LOCAL_CLIENTS[client_key] = self.client
            elif self.qdrant_mode == "http":
                self.client = QdrantClient(url=self.qdrant_url, timeout=60)
            else:
                raise ValueError("qdrant_mode must be 'local' or 'http'")
            return self.client
        except Exception as exc:
            self.error = self._qdrant_error(exc)
            raise RuntimeError(self.error) from exc

    def _vectorstore(self):
        if self.vectorstore is not None:
            return self.vectorstore
        try:
            from langchain_qdrant import QdrantVectorStore
        except ImportError as exc:
            self.error = "langchain-qdrant is not installed. Run: pip install langchain-qdrant"
            raise RuntimeError(self.error) from exc
        self.vectorstore = QdrantVectorStore(
            client=self._client(),
            collection_name=self.collection_name,
            embedding=self.langchain_embeddings,
        )
        return self.vectorstore

    def _models(self):
        try:
            from qdrant_client import models
        except ImportError as exc:
            self.error = "qdrant-client is not installed. Run: pip install qdrant-client"
            raise RuntimeError(self.error) from exc
        return models

    def _collection_exists(self) -> bool:
        client = self._client()
        try:
            if hasattr(client, "collection_exists"):
                return bool(client.collection_exists(self.collection_name))
            client.get_collection(self.collection_name)
            return True
        except Exception:
            return False

    def _point_count(self) -> int:
        client = self._client()
        try:
            response = client.count(collection_name=self.collection_name, exact=True)
            return int(getattr(response, "count", 0))
        except Exception:
            return 0

    def status(self) -> Dict[str, Any]:
        status = super().status()
        status.update(
            {
                "vector_store": "qdrant",
                "langchain_vectorstore": "langchain_qdrant.QdrantVectorStore",
                "qdrant_mode": self.qdrant_mode,
                "qdrant_path": self.qdrant_path if self.qdrant_mode == "local" else None,
                "qdrant_url": self.qdrant_url if self.qdrant_mode == "http" else None,
                "collection_name": self.collection_name,
                "collection_prefix": self.collection_prefix,
                "dashboard_url": self.qdrant_url.rstrip("/") + "/dashboard" if self.qdrant_mode == "http" else None,
            }
        )
        try:
            status["collection_exists"] = self._collection_exists()
            status["point_count"] = self._point_count() if status["collection_exists"] else 0
        except Exception as exc:
            status["collection_exists"] = False
            status["point_count"] = 0
            status["error"] = self._qdrant_error(exc)
        return status

    def _point_id(self, chunk: Dict[str, Any], index: int) -> str:
        chunk_id = str(chunk.get("chunk_id") or index)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.collection_name}:{chunk_id}"))

    def _payload_for_chunk(self, chunk: Dict[str, Any], index: int) -> Dict[str, Any]:
        payload = _clean_qdrant_payload(chunk) or {}
        payload["chunk_index"] = index
        payload["chunk_id"] = chunk.get("chunk_id")
        payload["text"] = chunk.get("text", "")
        payload["file_name"] = chunk.get("file_name")
        payload["relative_path"] = chunk.get("relative_path")
        payload["type"] = chunk.get("type")
        payload["model_name"] = self.model_name
        payload["retriever_name"] = self.name
        return {key: value for key, value in payload.items() if value is not None}

    def _create_collection(self, vector_size: int) -> None:
        client = self._client()
        models = self._models()
        client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE),
        )

    def _delete_collection_if_exists(self) -> None:
        client = self._client()
        if self._collection_exists():
            client.delete_collection(collection_name=self.collection_name)

    def build_index(self, force: bool = False) -> bool:
        try:
            if not self.chunks:
                self.error = None
                return True
            exists = self._collection_exists()
            if exists and not force and self._point_count() >= len(self.chunks):
                self._vectorstore()
                self.error = None
                return True

            if not self._load_model(local_files_only=True) and not self._load_model(local_files_only=False):
                return False
            sample_embedding = self._encode([self._passage_texts()[0]], show_progress_bar=False)

            if exists:
                self._delete_collection_if_exists()
            self._create_collection(vector_size=int(sample_embedding.shape[1]))
            self.vectorstore = None
            texts = [chunk.get("text", "") for chunk in self.chunks]
            metadatas = [self._payload_for_chunk(chunk, idx) for idx, chunk in enumerate(self.chunks)]
            ids = [self._point_id(chunk, idx) for idx, chunk in enumerate(self.chunks)]
            embeddings = self._encode(texts, show_progress_bar=True)
            self.embeddings = embeddings
            client = self._client()
            models = self._models()
            for start in range(0, len(texts), self.batch_size):
                end = start + self.batch_size
                points = [
                    models.PointStruct(
                        id=ids[index],
                        vector=embeddings[index].tolist(),
                        payload=metadatas[index],
                    )
                    for index in range(start, min(end, len(texts)))
                ]
                client.upsert(collection_name=self.collection_name, points=points, wait=True)
            self._vectorstore()
            self.error = None
            return True
        except Exception as exc:
            message = str(exc)
            self.error = message if "installed" in message or "embedding model" in message else self._qdrant_error(exc)
            return False

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        try:
            if not self._collection_exists() or self._point_count() == 0:
                if not self.build_index(force=False):
                    return []
            vectorstore = self._vectorstore()
            limit = min(len(self.chunks), max(top_k * 20, top_k))
            docs_and_scores = vectorstore.similarity_search_with_score(query, k=limit)
        except Exception as exc:
            message = str(exc)
            self.error = message if "installed" in message or "embedding model" in message else self._qdrant_error(exc)
            return []

        results: List[SearchResult] = []
        for doc, score in docs_and_scores:
            payload = getattr(doc, "metadata", None) or {}
            idx = payload.get("chunk_index")
            chunk = None
            if isinstance(idx, int) and 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
            elif payload.get("chunk_id") in self.chunk_by_id:
                chunk = self.chunk_by_id[payload["chunk_id"]]
            if chunk is None or not chunk_matches_filters(chunk, filters):
                continue
            results.append(SearchResult(chunk=chunk, score=float(score), rank=len(results) + 1, source=self.name))
            if len(results) >= top_k:
                break
        return results


def rrf_fuse(
    result_lists: Iterable[Sequence[SearchResult]],
    top_k: int = 5,
    rrf_k: int = 60,
) -> List[SearchResult]:
    fused: Dict[str, SearchResult] = {}
    for result_list in result_lists:
        for result in result_list:
            score = 1.0 / (rrf_k + result.rank)
            result_sources = result.sources or [result.source]
            if result.chunk_id not in fused:
                fused[result.chunk_id] = SearchResult(
                    chunk=result.chunk,
                    score=score,
                    rank=0,
                    source="rrf",
                    sources=list(result_sources),
                )
            else:
                fused[result.chunk_id].score += score
                for source in result_sources:
                    if source not in fused[result.chunk_id].sources:
                        fused[result.chunk_id].sources.append(source)

    ordered = sorted(fused.values(), key=lambda item: item.score, reverse=True)[:top_k]
    for rank, result in enumerate(ordered, start=1):
        result.rank = rank
    return ordered


class MultiRetriever(BaseRetriever):
    name = "multi_rrf"

    def __init__(self, retrievers: Sequence[BaseRetriever], rrf_k: int = 60) -> None:
        self.retrievers = list(retrievers)
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        candidate_lists = [retriever.search(query, top_k=top_k, filters=filters) for retriever in self.retrievers]
        return rrf_fuse(candidate_lists, top_k=top_k, rrf_k=self.rrf_k)


def lexical_rerank(query: str, results: Sequence[SearchResult], top_k: int = 5) -> List[SearchResult]:
    reranked: List[SearchResult] = []
    for result in results:
        overlap = token_overlap_score(query, result.chunk.get("text", ""))
        result.score = float(result.score) + overlap
        reranked.append(result)
    ordered = sorted(reranked, key=lambda item: item.score, reverse=True)[:top_k]
    for rank, result in enumerate(ordered, start=1):
        result.rank = rank
    return ordered

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Sequence

from ..config import CHUNKS_PATH, EMBEDDING_MODELS
from ..data import infer_type, load_chunks


CATEGORY_LABELS = {
    "cac_khoa": "khoa",
    "cac_trung_tam": "trung_tam",
    "dao_tao": "dao_tao",
    "faq": "faq",
    "tong_quan_ve_truong": "tong_quan_ve_truong",
    "tuyen_sinh": "tuyen_sinh",
    "unknown": "unknown",
}

CATEGORY_ORDER = [
    "khoa",
    "trung_tam",
    "dao_tao",
    "faq",
    "tong_quan_ve_truong",
    "tuyen_sinh",
    "unknown",
]

EMBEDDING_MODEL_MAX_TOKENS = {
    "miniLM": 128,
    "e5": 512,
    "bge_m3": 8192,
    "vietnamese_bi": 256,
}


def _embedding_input_text(model_name: str, text: str) -> str:
    if "e5" in model_name.lower():
        return f"passage: {text}"
    return text


def _safe_model_max_length(value: Any) -> int | None:
    if not isinstance(value, int):
        return None
    if value <= 0 or value > 1_000_000:
        return None
    return value


class EmbeddingTokenCounter:
    """Token counter that mirrors the tokenizer used by embedding models."""

    def __init__(self, model_name: str, local_files_only: bool = False) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.tokenizer = None
        self.error: str | None = None
        self.tokenizer_model_max_length: int | None = None
        self._load_tokenizer()

    def _load_tokenizer(self) -> None:
        try:
            from transformers import AutoTokenizer
        except Exception as exc:
            self.error = f"transformers is not available: {exc}"
            return

        attempts = [True] if self.local_files_only else [False, True]
        errors: List[str] = []
        for local_only in attempts:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    local_files_only=local_only,
                )
                self.tokenizer_model_max_length = _safe_model_max_length(
                    getattr(self.tokenizer, "model_max_length", None)
                )
                self.error = None
                return
            except Exception as exc:
                mode = "local cache" if local_only else "online/cache"
                errors.append(f"{mode}: {exc}")
        self.error = "Cannot load tokenizer. " + " | ".join(errors)

    def count(self, text: str) -> int:
        if self.tokenizer is None:
            raise RuntimeError(self.error or "Tokenizer is not loaded.")
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            verbose=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids = encoded.get("input_ids", [])
        return len(input_ids)


def _category_for_chunk(chunk: Dict[str, Any]) -> str:
    relative_path = chunk.get("relative_path", "")
    data_type = chunk.get("type") or infer_type(relative_path)
    return CATEGORY_LABELS.get(data_type, "unknown")


def _unit_for_chunk(chunk: Dict[str, Any]) -> str:
    relative_path = chunk.get("relative_path", "")
    parts = Path(relative_path).parts
    if len(parts) >= 2 and parts[0] in {"cac_khoa", "cac_trung_tam"}:
        return parts[1]
    return _category_for_chunk(chunk)


def _new_summary_bucket() -> Dict[str, Any]:
    return {
        "chunks": 0,
        "documents": set(),
        "tokens": 0,
        "min_tokens_per_chunk": None,
        "max_tokens_per_chunk": 0,
        "over_tokenizer_limit_chunks": 0,
        "over_embedding_limit_chunks": 0,
    }


def _update_bucket(
    buckets: MutableMapping[str, Dict[str, Any]],
    key: str,
    chunk: Dict[str, Any],
    token_count: int,
    tokenizer_model_max_length: int | None,
    embedding_model_max_tokens: int | None,
) -> None:
    bucket = buckets.setdefault(key, _new_summary_bucket())
    bucket["chunks"] += 1
    bucket["documents"].add(chunk.get("relative_path"))
    bucket["tokens"] += token_count
    if bucket["min_tokens_per_chunk"] is None:
        bucket["min_tokens_per_chunk"] = token_count
    else:
        bucket["min_tokens_per_chunk"] = min(bucket["min_tokens_per_chunk"], token_count)
    bucket["max_tokens_per_chunk"] = max(bucket["max_tokens_per_chunk"], token_count)
    if tokenizer_model_max_length is not None and token_count > tokenizer_model_max_length:
        bucket["over_tokenizer_limit_chunks"] += 1
    if embedding_model_max_tokens is not None and token_count > embedding_model_max_tokens:
        bucket["over_embedding_limit_chunks"] += 1


def _finalize_bucket(bucket: Dict[str, Any]) -> Dict[str, Any]:
    chunks = int(bucket["chunks"])
    tokens = int(bucket["tokens"])
    documents = bucket.get("documents", set())
    return {
        "chunks": chunks,
        "documents": len(documents),
        "tokens": tokens,
        "avg_tokens_per_chunk": round(tokens / chunks, 2) if chunks else 0.0,
        "min_tokens_per_chunk": int(bucket["min_tokens_per_chunk"] or 0),
        "max_tokens_per_chunk": int(bucket["max_tokens_per_chunk"]),
        "over_tokenizer_limit_chunks": int(bucket["over_tokenizer_limit_chunks"]),
        "over_embedding_limit_chunks": int(bucket["over_embedding_limit_chunks"]),
    }


def _ordered_category_summary(buckets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ordered: "OrderedDict[str, Any]" = OrderedDict()
    for category in CATEGORY_ORDER:
        if category in buckets:
            ordered[category] = _finalize_bucket(buckets[category])
    for category in sorted(set(buckets) - set(ordered)):
        ordered[category] = _finalize_bucket(buckets[category])
    return dict(ordered)


def _ordered_summary(buckets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    return {key: _finalize_bucket(buckets[key]) for key in sorted(buckets)}


def _model_keys(model_keys: Sequence[str] | None) -> List[str]:
    if model_keys is None:
        return list(EMBEDDING_MODELS.keys())
    return [key for key in model_keys if key]


def build_embedding_token_report(
    chunks: Sequence[Dict[str, Any]],
    model_keys: Sequence[str] | None = None,
    include_chunks: bool = True,
    local_files_only: bool = False,
) -> Dict[str, Any]:
    keys = _model_keys(model_keys)
    report: Dict[str, Any] = {
        "num_chunks": len(chunks),
        "category_labels": CATEGORY_LABELS,
        "models": {},
    }

    for key in keys:
        model_name = EMBEDDING_MODELS.get(key, key)
        counter = EmbeddingTokenCounter(model_name, local_files_only=local_files_only)
        embedding_model_max_tokens = EMBEDDING_MODEL_MAX_TOKENS.get(key, counter.tokenizer_model_max_length)
        model_report: Dict[str, Any] = {
            "model_key": key,
            "model_name": model_name,
            "tokenizer_loaded": counter.tokenizer is not None,
            "tokenizer_error": counter.error,
            "tokenizer_model_max_length": counter.tokenizer_model_max_length,
            "embedding_model_max_tokens": embedding_model_max_tokens,
            "embedding_input_prefix": "passage: " if "e5" in model_name.lower() else "",
            "by_category": {},
            "by_unit": {},
            "total": {},
        }
        if include_chunks:
            model_report["chunks"] = []

        if counter.tokenizer is None:
            report["models"][key] = model_report
            continue

        total_bucket = _new_summary_bucket()
        category_buckets: Dict[str, Dict[str, Any]] = {}
        unit_buckets: Dict[str, Dict[str, Any]] = {}

        for chunk in chunks:
            text = _embedding_input_text(model_name, chunk.get("text", ""))
            token_count = counter.count(text)
            category = _category_for_chunk(chunk)
            unit = _unit_for_chunk(chunk)

            _update_bucket(
                {"total": total_bucket},
                "total",
                chunk,
                token_count,
                counter.tokenizer_model_max_length,
                embedding_model_max_tokens,
            )
            _update_bucket(
                category_buckets,
                category,
                chunk,
                token_count,
                counter.tokenizer_model_max_length,
                embedding_model_max_tokens,
            )
            _update_bucket(
                unit_buckets,
                unit,
                chunk,
                token_count,
                counter.tokenizer_model_max_length,
                embedding_model_max_tokens,
            )

            if include_chunks:
                model_report["chunks"].append(
                    {
                        "chunk_id": chunk.get("chunk_id"),
                        "relative_path": chunk.get("relative_path"),
                        "category": category,
                        "unit": unit,
                        "token_count": token_count,
                    }
                )

        model_report["total"] = _finalize_bucket(total_bucket)
        model_report["by_category"] = _ordered_category_summary(category_buckets)
        model_report["by_unit"] = _ordered_summary(unit_buckets)
        report["models"][key] = model_report

    return report


def build_embedding_token_report_from_path(
    chunks_path: Path = CHUNKS_PATH,
    model_keys: Sequence[str] | None = None,
    include_chunks: bool = True,
    local_files_only: bool = False,
) -> Dict[str, Any]:
    report = build_embedding_token_report(
        load_chunks(chunks_path),
        model_keys=model_keys,
        include_chunks=include_chunks,
        local_files_only=local_files_only,
    )
    report["chunks_path"] = str(chunks_path)
    return report

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .config import CHUNKS_PATH, DATA_DIR, DATA_TYPES
from .data import faq_source_relative, infer_type, iter_source_files, load_chunks, read_text_file
from .text import clean_text


SUSPICIOUS_PATH_PATTERNS = {
    "to_chuc_dao_dao": "Ten thu muc co ve sai chinh ta; thuong nen la 'to_chuc_dao_tao'.",
    "thi_tpht": "Ten file co ve sai chinh ta; thuong nen la 'thi_thpt'.",
    "tham_khaodocx": "Ten file co duoi 'docx' bi dinh vao phan stem.",
}


def _relative(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def _source_relative(path: str) -> str:
    return path.split("#", 1)[0]


def _duplicate_values(values: List[str]) -> List[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _path_issues(paths: List[str]) -> List[Dict[str, str]]:
    issues: List[Dict[str, str]] = []
    for path in sorted(set(paths)):
        lowered = path.lower()
        for pattern, message in SUSPICIOUS_PATH_PATTERNS.items():
            if pattern in lowered:
                issues.append({"path": path, "pattern": pattern, "message": message})
    return issues


def audit_data(data_dir: Path = DATA_DIR, chunks_path: Path = CHUNKS_PATH) -> Dict[str, Any]:
    source_files = list(iter_source_files(data_dir))
    source_relatives = [_relative(path, data_dir) for path in source_files]
    if faq_relative := faq_source_relative(data_dir):
        source_relatives.append(faq_relative)
    source_relative_set = set(source_relatives)

    chunks = load_chunks(chunks_path) if chunks_path.exists() else []
    chunk_relatives = [str(chunk.get("relative_path", "")) for chunk in chunks]
    chunk_source_relatives = [_source_relative(path) for path in chunk_relatives]
    chunk_source_relative_set = set(chunk_source_relatives)
    chunk_ids = [str(chunk.get("chunk_id", "")) for chunk in chunks]

    empty_source_files: List[str] = []
    for path in source_files:
        text = clean_text(read_text_file(path))
        if not text:
            empty_source_files.append(_relative(path, data_dir))

    unknown_chunk_types = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "relative_path": chunk.get("relative_path"),
            "type": chunk.get("type"),
        }
        for chunk in chunks
        if chunk.get("type") not in DATA_TYPES
    ]
    empty_chunks = [
        {"chunk_id": chunk.get("chunk_id"), "relative_path": chunk.get("relative_path")}
        for chunk in chunks
        if not str(chunk.get("text", "")).strip()
    ]
    mismatched_chunk_types = [
        {
            "chunk_id": chunk.get("chunk_id"),
            "relative_path": chunk.get("relative_path"),
            "stored_type": chunk.get("type"),
            "inferred_type": infer_type(str(chunk.get("relative_path", ""))),
        }
        for chunk in chunks
        if chunk.get("type") != infer_type(str(chunk.get("relative_path", "")))
    ]

    top_level_dirs = sorted({Path(path).parts[0] for path in source_relatives if Path(path).parts})
    unexpected_top_level_dirs = [name for name in top_level_dirs if name not in DATA_TYPES]
    type_distribution = Counter(chunk.get("type", "unknown") for chunk in chunks)

    recommendations: List[str] = []
    if _duplicate_values(chunk_ids):
        recommendations.append("Rebuild chunks: duplicate chunk_id values were found.")
    if missing := sorted(chunk_source_relative_set - source_relative_set):
        recommendations.append(f"Rebuild processed/chunks.json: {len(missing)} chunk source path(s) no longer exist.")
    if _path_issues(source_relatives + chunk_relatives):
        recommendations.append("Review suspicious path names before rebuilding embeddings.")
    if unknown_chunk_types or unexpected_top_level_dirs:
        recommendations.append("Keep top-level data folders aligned with DATA_TYPES in iuh_rag/config.py.")

    return {
        "data_dir": str(data_dir),
        "chunks_path": str(chunks_path),
        "source_files": len(source_relatives),
        "chunks": len(chunks),
        "top_level_dirs": top_level_dirs,
        "unexpected_top_level_dirs": unexpected_top_level_dirs,
        "type_distribution": dict(sorted(type_distribution.items())),
        "duplicate_chunk_ids": _duplicate_values(chunk_ids),
        "empty_source_files": empty_source_files,
        "empty_chunks": empty_chunks,
        "unknown_chunk_types": unknown_chunk_types,
        "mismatched_chunk_types": mismatched_chunk_types,
        "chunks_missing_source_files": sorted(chunk_source_relative_set - source_relative_set),
        "source_files_without_chunks": sorted(source_relative_set - chunk_source_relative_set),
        "suspicious_paths": _path_issues(source_relatives + chunk_relatives),
        "recommendations": recommendations,
    }

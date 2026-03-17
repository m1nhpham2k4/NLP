from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


TEXT_EXTENSIONS = {".txt"}
CSV_EXTENSIONS = {".csv"}
DEFAULT_DATA_SOURCES = [
    "Thong_Tin_Khoa.csv",
    "Thong_Tin_Trung_Tam.csv",
    "Thong_Tin_Truong.csv",
    "Thong_Tin_Tuyen_Sinh.csv",
    "Quy_Che_Quy_Dinh.csv",
    "khoa_txt",
    "trungtam_txt",
    "phongban_txt",
    "lanhdao_txt",
    "vien_txt",
]


@dataclass(slots=True)
class DocumentChunk:
    source: str
    chunk_id: str
    text: str
    metadata: dict[str, str]


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def _collect_paths(project_root: Path, requested_sources: list[str] | None) -> list[Path]:
    sources = requested_sources or DEFAULT_DATA_SOURCES
    paths: list[Path] = []
    for source in sources:
        path = project_root / source
        if path.exists():
            paths.append(path)
    return paths


def _csv_rows_to_chunks(path: Path) -> Iterable[DocumentChunk]:
    df = pd.read_csv(path)
    for row_idx, row in df.fillna("").astype(str).iterrows():
        parts = [f"{column}: {value}" for column, value in row.items() if value.strip()]
        text = "\n".join(parts)
        for chunk_idx, chunk in enumerate(chunk_text(text)):
            yield DocumentChunk(
                source=str(path),
                chunk_id=f"{path.stem}-{row_idx}-{chunk_idx}",
                text=chunk,
                metadata={"row": str(row_idx), "type": "csv"},
            )


def _txt_to_chunks(path: Path) -> Iterable[DocumentChunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    for chunk_idx, chunk in enumerate(chunk_text(text)):
        yield DocumentChunk(
            source=str(path),
            chunk_id=f"{path.stem}-{chunk_idx}",
            text=chunk,
            metadata={"type": "txt"},
        )


def load_chunks(project_root: Path, requested_sources: list[str] | None = None) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in _collect_paths(project_root, requested_sources):
        if path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if file_path.suffix.lower() in TEXT_EXTENSIONS:
                    chunks.extend(_txt_to_chunks(file_path))
            continue

        if path.suffix.lower() in CSV_EXTENSIONS:
            chunks.extend(_csv_rows_to_chunks(path))
        elif path.suffix.lower() in TEXT_EXTENSIONS:
            chunks.extend(_txt_to_chunks(path))
    return chunks


def build_index(
    project_root: Path,
    index_path: Path,
    embeddings_path: Path,
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    requested_sources: list[str] | None = None,
) -> int:
    chunks = load_chunks(project_root, requested_sources=requested_sources)
    if not chunks:
        raise ValueError("No supported data sources were found to index.")

    embedder = SentenceTransformer(model_name)
    embeddings = embedder.encode([chunk.text for chunk in chunks], normalize_embeddings=True)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    index_path.write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(embeddings_path, embeddings=np.asarray(embeddings, dtype=np.float32))
    return len(chunks)

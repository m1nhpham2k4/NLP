from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from docx import Document as WordDocument
from sentence_transformers import SentenceTransformer


TEXT_EXTENSIONS = {".txt"}
TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}
DOC_EXTENSIONS = {".docx"}
DEFAULT_DATA_SOURCES = [
    "data/sources/iuh_data",
    "data/sources/Thong_Tin_Khoa.csv",
    "data/sources/Thong_Tin_Trung_Tam.csv",
    "data/sources/Thong_Tin_Truong.csv",
    "data/sources/Thong_Tin_Tuyen_Sinh.csv",
    "data/sources/Quy_Che_Quy_Dinh.csv",
    "data/sources/khoa_txt",
    "data/sources/trungtam_txt",
    "data/sources/phongban_txt",
    "data/sources/lanhdao_txt",
    "data/sources/vien_txt",
]


@dataclass(slots=True)
class DocumentChunk:
    source: str
    chunk_id: str
    text: str
    metadata: dict[str, str]


def _slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")
    return value.lower() or "chunk"


def _pretty_name(value: str) -> str:
    return value.replace("_", " ").replace("-", " ").strip()


def _extract_path_metadata(path: Path, project_root: Path) -> dict[str, str]:
    relative_path = path.relative_to(project_root)
    parents = list(relative_path.parts[:-1])
    metadata = {
        "relative_path": str(relative_path),
        "title": _pretty_name(path.stem),
        "source_type": path.suffix.lower().lstrip(".") or "directory",
        "section_path": " > ".join(_pretty_name(part) for part in parents) or "root",
        "category": _pretty_name(parents[0]) if parents else "root",
    }
    for idx, part in enumerate(parents[:4], start=1):
        metadata[f"level_{idx}"] = _pretty_name(part)
    return metadata


def chunk_text(text: str, chunk_size: int = 1100, overlap: int = 180) -> list[str]:
    normalized = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not normalized:
        return []

    paragraphs = [part.strip() for part in normalized.split("\n\n") if part.strip()]
    if not paragraphs:
        paragraphs = [normalized]

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= chunk_size:
            current = paragraph
            continue

        start = 0
        while start < len(paragraph):
            end = min(len(paragraph), start + chunk_size)
            piece = paragraph[start:end].strip()
            if piece:
                chunks.append(piece)
            if end == len(paragraph):
                break
            start = max(0, end - overlap)
        current = ""

    if current:
        chunks.append(current)
    return chunks


def _resolve_source_path(project_root: Path, source: str) -> Path | None:
    direct_path = project_root / source
    if direct_path.exists():
        return direct_path

    nested_path = project_root / "data" / "sources" / source
    if nested_path.exists():
        return nested_path

    return None


def _collect_paths(project_root: Path, requested_sources: list[str] | None) -> list[Path]:
    sources = requested_sources or DEFAULT_DATA_SOURCES
    paths: list[Path] = []
    for source in sources:
        resolved = _resolve_source_path(project_root, source)
        if resolved is not None:
            paths.append(resolved)
    return paths


def _yield_chunks(path: Path, project_root: Path, text: str, kind: str) -> Iterable[DocumentChunk]:
    base_metadata = _extract_path_metadata(path, project_root)
    base_metadata["kind"] = kind
    for chunk_idx, chunk in enumerate(chunk_text(text)):
        metadata = dict(base_metadata)
        metadata["chunk_index"] = str(chunk_idx)
        yield DocumentChunk(
            source=str(path),
            chunk_id=f"{_slugify(str(path.relative_to(project_root)))}-{chunk_idx}",
            text=chunk,
            metadata=metadata,
        )


def _csv_rows_to_chunks(path: Path, project_root: Path) -> Iterable[DocumentChunk]:
    df = pd.read_csv(path)
    base_metadata = _extract_path_metadata(path, project_root)
    for row_idx, row in df.fillna("").astype(str).iterrows():
        parts = [f"{column}: {value}" for column, value in row.items() if value.strip()]
        row_text = "\n".join(parts)
        if not row_text.strip():
            continue
        for chunk_idx, chunk in enumerate(chunk_text(row_text)):
            metadata = dict(base_metadata)
            metadata.update({"kind": "csv", "row": str(row_idx), "chunk_index": str(chunk_idx)})
            yield DocumentChunk(
                source=str(path),
                chunk_id=f"{_slugify(path.stem)}-row-{row_idx}-{chunk_idx}",
                text=chunk,
                metadata=metadata,
            )


def _xlsx_to_chunks(path: Path, project_root: Path) -> Iterable[DocumentChunk]:
    workbook = pd.read_excel(path, sheet_name=None)
    base_metadata = _extract_path_metadata(path, project_root)
    for sheet_name, df in workbook.items():
        for row_idx, row in df.fillna("").astype(str).iterrows():
            parts = [f"{column}: {value}" for column, value in row.items() if value.strip()]
            row_text = f"Sheet: {sheet_name}\n" + "\n".join(parts)
            if not row_text.strip():
                continue
            for chunk_idx, chunk in enumerate(chunk_text(row_text)):
                metadata = dict(base_metadata)
                metadata.update(
                    {
                        "kind": "xlsx",
                        "sheet": str(sheet_name),
                        "row": str(row_idx),
                        "chunk_index": str(chunk_idx),
                    }
                )
                yield DocumentChunk(
                    source=str(path),
                    chunk_id=f"{_slugify(path.stem)}-{_slugify(str(sheet_name))}-{row_idx}-{chunk_idx}",
                    text=chunk,
                    metadata=metadata,
                )


def _docx_to_chunks(path: Path, project_root: Path) -> Iterable[DocumentChunk]:
    doc = WordDocument(path)
    paragraphs = [paragraph.text.strip() for paragraph in doc.paragraphs if paragraph.text.strip()]
    if not paragraphs:
        return []
    return list(_yield_chunks(path, project_root, "\n\n".join(paragraphs), kind="docx"))


def _txt_to_chunks(path: Path, project_root: Path) -> Iterable[DocumentChunk]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return list(_yield_chunks(path, project_root, text, kind="txt"))


def load_chunks(project_root: Path, requested_sources: list[str] | None = None) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    for path in _collect_paths(project_root, requested_sources):
        if path.is_dir():
            for file_path in sorted(path.rglob("*")):
                if not file_path.is_file() or file_path.name.startswith("~$"):
                    continue
                suffix = file_path.suffix.lower()
                if suffix in TEXT_EXTENSIONS:
                    chunks.extend(_txt_to_chunks(file_path, project_root))
                elif suffix == ".csv":
                    chunks.extend(_csv_rows_to_chunks(file_path, project_root))
                elif suffix in {".xlsx", ".xls"}:
                    chunks.extend(_xlsx_to_chunks(file_path, project_root))
                elif suffix in DOC_EXTENSIONS:
                    chunks.extend(_docx_to_chunks(file_path, project_root))
            continue

        suffix = path.suffix.lower()
        if suffix == ".csv":
            chunks.extend(_csv_rows_to_chunks(path, project_root))
        elif suffix in {".xlsx", ".xls"}:
            chunks.extend(_xlsx_to_chunks(path, project_root))
        elif suffix in TEXT_EXTENSIONS:
            chunks.extend(_txt_to_chunks(path, project_root))
        elif suffix in DOC_EXTENSIONS and not path.name.startswith("~$"):
            chunks.extend(_docx_to_chunks(path, project_root))
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
    embeddings = embedder.encode(
        [chunk.text for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    index_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    index_path.write_text(
        json.dumps([asdict(chunk) for chunk in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(embeddings_path, embeddings=np.asarray(embeddings, dtype=np.float32))
    return len(chunks)

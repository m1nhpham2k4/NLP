from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import networkx as nx

from .config import TRIPLES_PATH
from .data import pretty_label, save_json
from .retrievers import SearchResult
from .text import normalize_text, token_overlap_score


ADMISSION_CODE_RE = re.compile(r"\b[A-D]\d{2}\b")
MA_NGANH_RE = re.compile(r"\b7\d{6}\b|\b[0-9]{7}\b")


def _subject_from_department(chunk: Dict[str, Any]) -> str | None:
    metadata = chunk.get("metadata", {})
    department = metadata.get("department")
    if department:
        return department if department.lower().startswith("khoa") else f"Khoa {department}"
    return None


def _program_from_path(chunk: Dict[str, Any]) -> str | None:
    metadata = chunk.get("metadata", {})
    program = metadata.get("program")
    if not program:
        return None
    file_name = str(chunk.get("file_name", "")).lower()
    if file_name in {"gioi_thieu.txt", "ban_lanh_dao.txt", "co_so_vat_chat.txt"}:
        return None
    return f"Ngành {program}" if not program.lower().startswith("ngành") else program


def _is_generic_conflict_subject(subject: str) -> bool:
    subject_norm = normalize_text(subject)
    return subject.endswith(".txt") or subject_norm in {"tuyen sinh"}


def extract_triples(chunks: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    triples: List[Dict[str, Any]] = []

    for chunk in chunks:
        source_chunk_id = chunk["chunk_id"]
        relative_path = chunk.get("relative_path", "")
        parts = Path(relative_path).parts
        text = chunk.get("text", "")
        text_norm = normalize_text(text)

        department = _subject_from_department(chunk)
        program = _program_from_path(chunk)

        if department:
            triples.append(
                {
                    "subject": "Trường Đại học Công nghiệp TP.HCM",
                    "relation": "co_khoa",
                    "object": department,
                    "source_chunk_id": source_chunk_id,
                }
            )

        if department and program and "dao_tao" in parts:
            triples.append(
                {
                    "subject": department,
                    "relation": "dao_tao",
                    "object": program,
                    "source_chunk_id": source_chunk_id,
                }
            )

        center = chunk.get("metadata", {}).get("center")
        if center:
            triples.append(
                {
                    "subject": "Trường Đại học Công nghiệp TP.HCM",
                    "relation": "co_trung_tam",
                    "object": center,
                    "source_chunk_id": source_chunk_id,
                }
            )

        if "tuyen_sinh" in relative_path or "tuyen sinh" in text_norm:
            for code in sorted(set(ADMISSION_CODE_RE.findall(text))):
                triples.append(
                    {
                        "subject": program or chunk.get("file_name", "Tuyển sinh"),
                        "relation": "xet_tuyen",
                        "object": code,
                        "source_chunk_id": source_chunk_id,
                    }
                )
            for code in sorted(set(MA_NGANH_RE.findall(text))):
                triples.append(
                    {
                        "subject": program or chunk.get("file_name", "Tuyển sinh"),
                        "relation": "co_ma_nganh",
                        "object": code,
                        "source_chunk_id": source_chunk_id,
                    }
                )

        if "hoc_bong" in relative_path:
            triples.append(
                {
                    "subject": "Quy định học bổng",
                    "relation": "duoc_neu_trong",
                    "object": chunk.get("file_name", "hoc_bong"),
                    "source_chunk_id": source_chunk_id,
                }
            )

    deduped = []
    seen = set()
    for triple in triples:
        key = (triple["subject"], triple["relation"], triple["object"], triple["source_chunk_id"])
        if key not in seen:
            deduped.append(triple)
            seen.add(key)
    return deduped


class KnowledgeGraph:
    def __init__(self, chunks: Sequence[Dict[str, Any]], triples: Sequence[Dict[str, Any]] | None = None) -> None:
        self.chunks = {chunk["chunk_id"]: chunk for chunk in chunks}
        self.triples = list(triples) if triples is not None else extract_triples(chunks)
        self.graph = nx.MultiDiGraph()
        self._build_graph()

    def _build_graph(self) -> None:
        for triple in self.triples:
            self.graph.add_edge(
                triple["subject"],
                triple["object"],
                relation=triple["relation"],
                source_chunk_id=triple["source_chunk_id"],
            )

    def save(self, path=TRIPLES_PATH) -> None:
        save_json(path, self.triples)

    def search(self, query: str, top_k: int = 5) -> List[SearchResult]:
        scores: Dict[str, float] = defaultdict(float)
        sources: Dict[str, List[str]] = defaultdict(list)
        for triple in self.triples:
            triple_text = f"{triple['subject']} {pretty_label(triple['relation'])} {triple['object']}"
            score = token_overlap_score(query, triple_text)
            if score <= 0:
                continue
            chunk_id = triple["source_chunk_id"]
            scores[chunk_id] += score
            if "graph" not in sources[chunk_id]:
                sources[chunk_id].append("graph")

        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
        results = []
        for rank, (chunk_id, score) in enumerate(ordered, start=1):
            chunk = self.chunks.get(chunk_id)
            if not chunk:
                continue
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=float(score),
                    rank=rank,
                    source="graph",
                    sources=sources[chunk_id],
                )
            )
        return results

    def detect_metadata_conflicts(self) -> List[Dict[str, Any]]:
        grouped: Dict[tuple[str, str], Dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for triple in self.triples:
            if triple["relation"] not in {"co_ma_nganh"}:
                continue
            if _is_generic_conflict_subject(str(triple["subject"])):
                continue
            key = (triple["subject"], triple["relation"])
            grouped[key]["objects"].add(triple["object"])
            grouped[key]["sources"].add(triple["source_chunk_id"])

        conflicts = []
        for (subject, relation), payload in grouped.items():
            if len(payload["objects"]) > 1:
                conflicts.append(
                    {
                        "subject": subject,
                        "relation": relation,
                        "values": sorted(payload["objects"]),
                        "source_chunk_ids": sorted(payload["sources"]),
                    }
                )
        return conflicts

import json
import csv
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .config import (
    CANDIDATE_QUESTIONS_PATH,
    CHUNKS_PATH,
    DATA_DIR,
    DATA_TYPES,
    FINAL_QUESTIONS_PATH,
)
from .text import chunk_text, clean_text, normalize_text, slugify


TEXT_SUFFIXES = {".txt", ""}
SKIP_SUFFIXES = {".csv", ".xlsx", ".ipynb", ".docx", ".png", ".drawio", ".joblib"}
FAQ_FILE_NAMES = ("faq_data.csv", "faq_with_header.csv", "faq_with_analysis.csv")
FAQ_QUESTIONS_PER_CHUNK = 12


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def read_text_file(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1258"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(errors="ignore")


def _read_faq_csv(path: Path) -> List[Dict[str, str]]:
    for encoding in ("utf-8-sig", "utf-8", "cp1258"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                rows = list(csv.DictReader(file))
            break
        except UnicodeDecodeError:
            continue
    else:
        with path.open("r", errors="ignore", newline="") as file:
            rows = list(csv.DictReader(file))

    output = []
    for row in rows:
        question = clean_text(row.get("question", ""))
        answer = clean_text(row.get("answer", ""))
        if question and answer:
            output.append({"question": question, "answer": answer})
    return output


def _faq_source_path(data_dir: Path) -> Path | None:
    candidates = [
        data_dir.parent / "faq_model" / "faq_data.csv",
        data_dir / "faq_data.csv",
        data_dir / "faq_with_header.csv",
        data_dir / "faq_with_analysis.csv",
    ]
    seen = set()
    for path in candidates:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        if path.exists():
            return path
    return None


def _faq_type_and_topic(question: str, answer: str) -> tuple[str, str]:
    text_norm = normalize_text(f"{question} {answer}")

    def has_any(phrases: tuple[str, ...]) -> bool:
        return any(phrase in text_norm for phrase in phrases)

    if has_any(("tuyen sinh", "xet tuyen", "diem chuan", "diem trung tuyen", "thi thpt", "hoc ba", "danh gia nang luc", "chi tieu", "ma nganh")):
        return "tuyen_sinh", "tuyen_sinh"
    if has_any(("hoc bong", "mien giam")):
        return "dao_tao", "hoc_bong"
    if has_any(("hoc phi", "dong tien")):
        return "dao_tao", "hoc_phi"
    if has_any(("quy che", "quy dinh", "tin chi", "tot nghiep", "thoi hoc", "bao luu", "nghi hoc", "dang ky hoc", "hoc lai", "cai thien", "diem trung binh")):
        return "dao_tao", "quy_dinh"
    if has_any(("trung tam", "thu vien", "ktx", "ky tuc xa")):
        return "cac_trung_tam", "ho_tro"
    if has_any(("khoa ", "nganh ", "chuong trinh dao tao", "chuan dau ra", "viec lam", "nghe nghiep")):
        return "cac_khoa", "khoa_nganh"
    return "faq", "faq"


def _faq_relative_path(path: Path, data_dir: Path) -> str:
    if path.name in FAQ_FILE_NAMES:
        return f"faq/{path.name}"
    for base in (data_dir, data_dir.parent):
        try:
            return path.relative_to(base).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def faq_source_relative(data_dir: Path = DATA_DIR) -> str | None:
    faq_path = _faq_source_path(data_dir)
    if faq_path is None:
        return None
    return _faq_relative_path(faq_path, data_dir)


def _build_faq_chunks(data_dir: Path, existing_ids: set[str]) -> List[Dict[str, Any]]:
    faq_path = _faq_source_path(data_dir)
    if faq_path is None:
        return []

    rows = _read_faq_csv(faq_path)
    grouped: Dict[str, List[str]] = {}
    original_answer: Dict[str, str] = {}
    for row in rows:
        answer = row["answer"]
        answer_key = clean_text(answer).casefold()
        original_answer.setdefault(answer_key, answer)
        questions = grouped.setdefault(answer_key, [])
        if row["question"] not in questions:
            questions.append(row["question"])

    chunks: List[Dict[str, Any]] = []
    relative_path = _faq_relative_path(faq_path, data_dir)
    for answer_index, (answer_key, questions) in enumerate(grouped.items(), start=1):
        answer = original_answer[answer_key]
        chunk_type, topic = _faq_type_and_topic(" ".join(questions[:5]), answer)
        for batch_index, start in enumerate(range(0, len(questions), FAQ_QUESTIONS_PER_CHUNK)):
            batch = questions[start : start + FAQ_QUESTIONS_PER_CHUNK]
            question_lines = "\n".join(f"- {question}" for question in batch)
            text = clean_text(
                "\n".join(
                    [
                        "FAQ IUH",
                        f"Chủ đề: {topic}",
                        "Câu hỏi mẫu:",
                        question_lines,
                        "Trả lời:",
                        answer,
                    ]
                )
            )
            file_slug = f"faq_{answer_index:04d}_{slugify(batch[0], max_len=50)}"
            chunks.append(
                {
                    "chunk_id": _stable_chunk_id(file_slug, batch_index, existing_ids),
                    "file_name": faq_path.name,
                    "relative_path": f"{relative_path}#answer_{answer_index:04d}_batch_{batch_index:03d}",
                    "type": "faq",
                    "text": text,
                    "metadata": {
                        "source_kind": "faq",
                        "type": chunk_type,
                        "content_type": chunk_type,
                        "topic": topic,
                        "faq_answer_id": f"faq_{answer_index:04d}",
                        "faq_batch_index": batch_index,
                        "faq_question_count": len(batch),
                        "faq_total_questions_for_answer": len(questions),
                        "faq_source_path": relative_path,
                    },
                }
            )
    return chunks


def iter_source_files(data_dir: Path = DATA_DIR) -> Iterable[Path]:
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name.startswith(".") or path.name.startswith("~$"):
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            yield path


def infer_type(relative_path: str) -> str:
    first_part = Path(relative_path).parts[0] if Path(relative_path).parts else ""
    return first_part if first_part in DATA_TYPES else "unknown"


def _stable_chunk_id(file_slug: str, chunk_index: int, existing: set[str]) -> str:
    base = f"{file_slug}_chunk_{chunk_index:03d}"
    if base not in existing:
        existing.add(base)
        return base
    counter = 1
    while f"{base}_{counter}" in existing:
        counter += 1
    chunk_id = f"{base}_{counter}"
    existing.add(chunk_id)
    return chunk_id


def build_chunks(
    data_dir: Path = DATA_DIR,
    output_path: Path = CHUNKS_PATH,
    max_chars: int = 1400,
    overlap_chars: int = 220,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    existing_ids: set[str] = set()

    for file_path in iter_source_files(data_dir):
        relative_path = file_path.relative_to(data_dir).as_posix()
        raw_text = read_text_file(file_path)
        text = clean_text(raw_text)
        if not text:
            continue

        file_slug = slugify(file_path.stem or file_path.name)
        file_title = pretty_label(file_path.stem)
        
        for index, chunk in enumerate(chunk_text(text, max_chars=max_chars, overlap_chars=overlap_chars)):
            # Thêm tiêu đề file vào đầu mỗi chunk để tránh mất ngữ cảnh (Context Loss)
            # Giúp các chunk ở giữa file (như các dòng của bảng điểm chuẩn) vẫn chứa từ khóa "Điểm chuẩn 2025"
            contextual_chunk = f"[{file_title}]\n{chunk}" if index > 0 else chunk
            
            chunks.append(
                {
                    "chunk_id": _stable_chunk_id(file_slug, index, existing_ids),
                    "file_name": file_path.name,
                    "relative_path": relative_path,
                    "type": infer_type(relative_path),
                    "text": contextual_chunk,
                }
            )

    chunks.extend(_build_faq_chunks(data_dir, existing_ids))
    save_json(output_path, chunks)
    return chunks


def load_chunks(path: Path = CHUNKS_PATH) -> List[Dict[str, Any]]:
    chunks = load_json(path)
    if not isinstance(chunks, list):
        raise ValueError(f"{path} must contain a JSON array.")
    for chunk in chunks:
        for field in ("chunk_id", "file_name", "relative_path", "type", "text"):
            if field not in chunk:
                raise ValueError(f"Chunk is missing field {field}: {chunk}")
    return chunks


def load_questions(path: Path = FINAL_QUESTIONS_PATH) -> List[Dict[str, Any]]:
    questions = load_json(path)
    if not isinstance(questions, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return questions


def ensure_questions_final(
    candidate_path: Path = CANDIDATE_QUESTIONS_PATH,
    final_path: Path = FINAL_QUESTIONS_PATH,
    overwrite: bool = False,
) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    if final_path.exists() and not overwrite:
        return final_path
    shutil.copyfile(candidate_path, final_path)
    return final_path


def build_chunk_lookup(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {chunk["chunk_id"]: chunk for chunk in chunks}


def pretty_label(value: str) -> str:
    value = value.replace("_", " ").strip()
    words = []
    for word in value.split():
        if word.lower() in {"cntt", "iuh", "tp", "hcm"}:
            words.append(word.upper())
        else:
            words.append(word.capitalize())
    return " ".join(words)


def infer_metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
    relative_path = chunk.get("relative_path", "")
    parts = Path(relative_path).parts
    text = chunk.get("text", "")
    metadata: Dict[str, Any] = {
        "type": chunk.get("type", infer_type(relative_path)),
        "file_name": chunk.get("file_name"),
        "relative_path": relative_path,
        "chunk_id": chunk.get("chunk_id"),
    }

    years = sorted(set(re.findall(r"\b20\d{2}\b", f"{relative_path} {text}")))
    if years:
        metadata["year"] = years[-1]
        metadata["years"] = years

    if parts and parts[0] == "cac_khoa" and len(parts) >= 2:
        metadata["department"] = pretty_label(parts[1])
    if parts and parts[0] == "cac_trung_tam" and len(parts) >= 2:
        metadata["center"] = pretty_label(parts[1])

    stem = Path(relative_path).stem or Path(relative_path).name
    if "dao_tao" in parts and stem:
        metadata["program"] = pretty_label(stem)

    lowered = relative_path.lower()
    if "tuyen_sinh" in lowered:
        metadata["topic"] = "tuyen_sinh"
    elif "hoc_bong" in lowered:
        metadata["topic"] = "hoc_bong"
    elif "quy_che" in lowered or "quy_dinh" in lowered or "noi_quy" in lowered:
        metadata["topic"] = "quy_dinh"
    elif "gioi_thieu" in lowered:
        metadata["topic"] = "gioi_thieu"
    elif "faq" in lowered:
        metadata["topic"] = "faq"

    return metadata


def enrich_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    enriched = []
    for chunk in chunks:
        item = dict(chunk)
        item["metadata"] = {**infer_metadata(item), **item.get("metadata", {})}
        enriched.append(item)
    return enriched

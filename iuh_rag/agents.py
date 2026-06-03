from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from .data import pretty_label
from .query_context import (
    build_conversation_context,
    build_evidence_constraints,
    build_reasoning_summary,
    build_retrieval_query,
    build_retrieval_strategy,
    build_rewritten_followup,
    build_risk_flags,
    build_scope,
    detect_followup,
    detect_topic,
    missing_slots_for,
    resolve_topic,
    resolve_year,
    topic_intent,
    topic_label,
)
from .text import normalize_text

ADMISSION_TERMS = {
    "diem chuan",
    "diem tuyen sinh",
    "diem trung tuyen",
    "tuyen sinh",
    "xet tuyen",
    "ma nganh",
    "to hop",
    "chi tieu",
}

PROGRAM_TERMS = {
    "chuong trinh",
    "dai tra",
    "tang cuong tieng anh",
    "chat luong cao",
    "tich hop",
}

TRAINING_RULE_TERMS = {
    "chuan dau ra tieng anh",
    "toeic",
    "tieng anh 1",
    "tieng anh 2",
    "trung binh chung",
    "diem ren luyen",
    "thoi gian dao tao",
    "thoi gian hoc",
    "hoc chinh thuc",
    "keo dai toi da",
    "tin chi",
    "tot nghiep",
    "thoi hoc",
    "hoc cai thien",
    "dong tien",
}

OVERVIEW_TERMS = {
    "tong quan",
    "toan canh",
    "bao quat",
    "toan bo du lieu",
    "toan the du lieu",
    "full data",
    "dataset",
    "data co gi",
    "du lieu co gi",
    "du lieu gom",
    "co nhung thong tin gi",
    "chatbot biet gi",
    "he thong biet gi",
    "tat ca thong tin",
    "tong hop thong tin",
}

FOLLOW_UP_TERMS = {
    "vay",
    "the",
    "con",
    "thi sao",
    "the nao",
    "nganh do",
    "nganh nay",
    "khoa do",
    "khoa nay",
    "chuong trinh do",
    "chuong trinh nay",
    "tim giup",
    "giup toi",
    "giup em",
    "ho tro toi",
    "ho tro em",
    "hay giup",
    "hay ho tro",
    "vang",
    "duoc",
    "ok",
    "oke",
    "dong y",
    "tiep tuc",
    "noi ro hon",
    "cu the hon",
    "giai thich them",
    "chi tiet hon",
    "ro hon",
}

FOCUS_TERMS = {
    "thong tin lien he": "thông tin liên hệ",
    "lien he": "thông tin liên hệ",
    "so dien thoai": "số điện thoại",
    "email": "email",
    "de tai nghien cuu khoa hoc": "đề tài nghiên cứu khoa học",
    "nghien cuu khoa hoc": "nghiên cứu khoa học",
    "khoa luan": "khóa luận",
    "giang vien huong dan": "giảng viên hướng dẫn",
    "hoc cai thien": "học cải thiện điểm",
    "cai thien diem": "học cải thiện điểm",
    "hoc phi": "học phí",
    "dong tien": "đóng tiền",
    "diem chuan": "điểm chuẩn",
    "tuyen sinh": "tuyển sinh",
    "hoc bong": "học bổng",
    "tot nghiep": "tốt nghiệp",
    "tin chi": "tín chỉ",
}

ROLE_RULES = [
    ("phu_huynh", ("con toi", "phu huynh", "con em")),
    ("thi_sinh", ("em muon xet tuyen", "thi sinh", "dang ky xet tuyen")),
    ("sinh_vien", ("sinh vien", "dang hoc", "hoc lai", "tot nghiep")),
    ("can_bo_tu_van", ("can bo tu van", "tu van tuyen sinh", "trich dan")),
]

GREETINGS = {"hello", "hi", "chao", "xin chao", "chao ban", "alo", "ban oi"}
HELP_PHRASES = {"ban giup duoc gi", "ban co the giup gi", "bot giup duoc gi", "huong dan toi"}


@dataclass
class QueryPlan:
    original_question: str
    rewritten_question: str
    role: str = "general"
    intent: str = "fact"
    filters: Dict[str, Any] = field(default_factory=dict)
    sub_questions: List[str] = field(default_factory=list)
    hypothesis_queries: List[str] = field(default_factory=list)
    current_entities: List[str] = field(default_factory=list)
    history_entities: List[str] = field(default_factory=list)
    comparison_entities: List[str] = field(default_factory=list)
    use_hyde: bool = False
    use_history: bool = False
    use_metadata: bool = False
    use_graph: bool = False
    needs_verification: bool = True
    is_followup: bool = False
    topic: str | None = None
    year: str | None = None
    entities: List[Dict[str, Any]] = field(default_factory=list)
    missing_slots: List[str] = field(default_factory=list)
    retrieval_query: str | None = None
    evidence_constraints: Dict[str, Any] = field(default_factory=dict)
    needs_clarification: bool = False
    clarification_question: str | None = None
    conversation_context: Dict[str, Any] = field(default_factory=dict)
    followup_reason: str | None = None
    is_topic_shift: bool = False
    reasoning_summary: str = ""
    scope: Dict[str, Any] = field(default_factory=dict)
    risk_flags: List[str] = field(default_factory=list)
    retrieval_strategy: Dict[str, Any] = field(default_factory=dict)


def _has_any(text_norm: str, phrases: Sequence[str]) -> bool:
    return any(phrase in text_norm for phrase in phrases)


def _strip_sources(text: str) -> str:
    return re.split(r"\n\s*Nguồn\s*:", str(text), maxsplit=1, flags=re.IGNORECASE)[0].strip()


def _unique_norm(items: Sequence[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        cleaned = re.sub(r"\s+", " ", str(item)).strip(" .,:;?!")
        key = normalize_text(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def _department_word(text: str) -> bool:
    text_norm = normalize_text(text)
    if "khoa luan" in text_norm:
        return False
    return bool(re.search(r"\bkhoa\b", str(text).lower()))


def _extract_subjects(text: str) -> List[str]:
    subjects: List[str] = []
    pattern = (
        r"\b(ngành|nganh|khoa|chương trình|chuong trinh|chuyên ngành|chuyen nganh)\s+"
        r"([A-ZĐa-zÀ-ỹ0-9][\wÀ-ỹ0-9]*(?:\s+[A-ZĐa-zÀ-ỹ0-9][\wÀ-ỹ0-9]*){0,5})"
    )
    for prefix, raw in re.findall(pattern, text, flags=re.IGNORECASE):
        value = re.split(
            r"\s+(?:ạ|a|năm|nam|20\d{2}|tại|tai|ở|o|của|cua|có|co|gồm|gom|những|nhung|gì|gi|nào|nao|không|khong|và|va|hoặc|hoac|để|de|trên|tren|lấy|lay|điểm|diem|xét|xet|tuyển|tuyen|cần|can|iuh|là|la|bao)\b",
            raw.strip(),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip(" ,.;:?!")
        if not value or normalize_text(value) in {"do", "nay", "truoc", "hoc", "nao", "gi", "nhung"}:
            continue
        prefix_norm = normalize_text(prefix)
        label = {"nganh": "ngành", "khoa": "khoa", "chuong trinh": "chương trình", "chuyen nganh": "chuyên ngành"}[prefix_norm]
        subjects.append(f"{label} {value}")
    return _unique_norm(subjects)


class QueryUnderstandingAgent:
    def __init__(self, known_terms: Sequence[str] | None = None) -> None:
        best: Dict[str, str] = {}
        for term in known_terms or []:
            key = normalize_text(term)
            if key and (key not in best or self._label_quality(term) > self._label_quality(best[key])):
                best[key] = term
        self.known_terms = sorted(best.values(), key=lambda term: len(normalize_text(term)), reverse=True)
        self.known_terms_norm = [(term, normalize_text(term)) for term in self.known_terms]

    def conversation_context(self, history: Sequence[Dict[str, Any]] | None = None):
        return build_conversation_context(
            history,
            lambda text: self.entities_in_text(text),
        )

    @staticmethod
    def _label_quality(term: str) -> int:
        return sum(1 for char in term if ord(char) > 127) - term.count("_")

    def detect_role(self, question: str, explicit_role: str | None = None) -> str:
        if explicit_role:
            return explicit_role
        q = normalize_text(question)
        return next((role for role, terms in ROLE_RULES if _has_any(q, terms)), "general")

    def detect_intent(self, question: str, history: Sequence[Dict[str, str]] | None = None) -> str:
        q = normalize_text(question)
        tokens = set(q.split())

        if q in GREETINGS or _has_any(q, HELP_PHRASES) or (len(tokens) <= 4 and tokens & {"hello", "hi", "chao", "alo"}):
            return "greeting"
        if history and self.is_contextual_follow_up(question, history):
            return "history"
        if _has_any(q, OVERVIEW_TERMS) or ("co gi" in q and tokens & {"du", "lieu", "data", "iuh", "truong"}):
            return "overview"
        if _has_any(q, ("mau thuan", "khac nhau giua cac nguon", "xung dot", "khong nhat quan", "trai nguoc")):
            return "conflict_check"
        if re.search(r"\b20(2[6-9]|[3-9]\d)\b", question) or _has_any(q, ("khong co du lieu", "chua co thong tin", "co ton tai")):
            return "missing_evidence"
        if _has_any(q, ("nen chon", "phu hop", "tu van", "em thich", "con toi thich")):
            return "tu_van"
        if _has_any(q, ("so sanh", "khac nhau", "giong nhau")):
            return "so_sanh"
        if _has_any(q, TRAINING_RULE_TERMS) or re.search(r"\bkhoa\s+(?:dh|cd|ch|lt|vl)\d", q):
            return "dao_tao_quy_dinh"
        if _has_any(q, PROGRAM_TERMS) and not _has_any(q, ADMISSION_TERMS):
            return "khoa_nganh"
        if _has_any(q, ADMISSION_TERMS) or ("diem" in tokens and (tokens & {"iuh", "truong"} or _has_any(q, PROGRAM_TERMS))):
            return "tuyen_sinh"
        if _has_any(q, ("quy che", "quy dinh", "hoc bong", "tin chi", "tot nghiep", "thoi hoc", "hoc cai thien", "dong tien")):
            return "dao_tao_quy_dinh"
        if _department_word(question) or _has_any(q, ("nganh", "dao tao", "trung tam")):
            return "khoa_nganh"
        return "fact"

    def entities_in_text(
        self,
        text: str,
        *,
        prefer_program: bool = False,
        exclude: Sequence[str] | None = None,
    ) -> List[str]:
        text_norm = normalize_text(text)
        exclude_norms = {normalize_text(item) for item in exclude or []}
        matches = []
        for term, term_norm in self.known_terms_norm:
            if not term_norm or term_norm in exclude_norms or term_norm not in text_norm:
                continue
            if prefer_program and (term_norm.startswith("trung tam ") or (term_norm.startswith("khoa ") and not term_norm.startswith("khoa hoc "))):
                continue
            matches.append((text_norm.rfind(term_norm), len(term_norm), pretty_label(term) if "_" in term else term))
        known = [term for _, _, term in sorted(matches, key=lambda row: (row[0], row[1]), reverse=True)]
        return _unique_norm([*known, *_extract_subjects(text)])

    def _last_entity(self, history: Sequence[Dict[str, str]], *, prefer_program: bool = False, exclude: Sequence[str] | None = None) -> str | None:
        for item in reversed(history[-8:]):
            if item.get("role") != "user":
                continue
            entities = self.entities_in_text(item.get("content", ""), prefer_program=prefer_program, exclude=exclude)
            if entities:
                return entities[0]
        return None

    @staticmethod
    def _last_user_year(history: Sequence[Dict[str, str]]) -> str | None:
        for item in reversed(history[-12:]):
            if item.get("role") == "user":
                match = re.search(r"\b20[2-3]\d\b", item.get("content", ""))
                if match:
                    return match.group(0)
        return None

    @staticmethod
    def _last_admission_topic(history: Sequence[Dict[str, str]]) -> str | None:
        for item in reversed(history[-12:]):
            if item.get("role") != "user":
                continue
            q = normalize_text(item.get("content", ""))
            if "diem chuan" in q:
                return "điểm chuẩn"
            if "diem trung tuyen" in q:
                return "điểm trúng tuyển"
            if "diem tuyen sinh" in q or ("diem" in q.split() and "tuyen sinh" in q):
                return "điểm tuyển sinh"
            if "xet tuyen" in q:
                return "thông tin xét tuyển"
            if "tuyen sinh" in q:
                return "thông tin tuyển sinh"
        return None

    @staticmethod
    def _is_short_action(question: str) -> bool:
        q = normalize_text(question)
        tokens = set(q.split())
        return len(tokens) <= 8 and (
            _has_any(q, ("tim giup", "giup", "ho tro", "hay giup", "hay ho tro", "vang", "duoc", "ok", "oke", "dong y", "tiep tuc"))
            or bool(tokens & {"tim", "giup", "vang", "duoc", "ok", "oke"})
        )

    @staticmethod
    def _is_clarify(question: str) -> bool:
        q = normalize_text(question)
        return len(q.split()) <= 8 and _has_any(q, ("noi ro hon", "cu the hon", "giai thich them", "chi tiet hon", "ro hon", "tai sao", "vi sao"))

    def is_contextual_follow_up(self, question: str, history: Sequence[Dict[str, str]] | None = None) -> bool:
        if not history:
            return False
        context = self.conversation_context(history)
        current_entities = self.entities_in_text(question)
        if detect_followup(question, history, context, current_entities=current_entities).is_followup:
            return True
        q = normalize_text(question)
        return (
            q.startswith(("vay ", "the ", "con ", "ca ", "cung ", "va "))
            or _has_any(q, FOLLOW_UP_TERMS)
            or bool(set(q.split()) & {"do", "nay", "tren", "day", "kia"})
            or (len(q.split()) <= 7 and _has_any(q, PROGRAM_TERMS))
        )

    def rewrite_with_history(self, question: str, history: Sequence[Dict[str, str]] | None = None) -> str:
        if not history:
            return question

        context = self.conversation_context(history)
        prefer_program = "nganh" in normalize_text(question).split()
        current_entities = self.entities_in_text(question, prefer_program=prefer_program)
        followup = detect_followup(question, history, context, current_entities=current_entities)
        topic = resolve_topic(question, context, followup)
        year = resolve_year(question, context, followup)
        rewritten_followup = build_rewritten_followup(
            question,
            context,
            followup,
            current_entities=current_entities,
            topic=topic,
            year=year,
        )
        if rewritten_followup:
            return rewritten_followup

        inferred = self._rewrite_short_follow_up(question, history)
        if inferred:
            return inferred

        current = self.entities_in_text(question, prefer_program=prefer_program)
        previous = self._last_entity(history, prefer_program=prefer_program, exclude=current)
        history_year = self._last_user_year(history)
        admission_topic = self._last_admission_topic(history)

        if self.is_contextual_follow_up(question, history) and current and previous:
            rewritten = f"So sánh {current[0]} với {previous}. Câu hỏi gốc: {question}"
        elif admission_topic and not _has_any(normalize_text(question), ADMISSION_TERMS) and self.is_contextual_follow_up(question, history):
            detail = re.sub(r"^\s*(xem|coi|cả|ca|cùng|cung|và|va)\s+", "", question, flags=re.IGNORECASE).strip(" ,.;:")
            rewritten = f"{admission_topic} {detail} của IUH"
        elif previous:
            rewritten = question
            for needle in ("ngành đó", "ngành này", "khoa đó", "khoa này", "chương trình đó", "chương trình này", "nó"):
                rewritten = re.sub(needle, previous, rewritten, flags=re.IGNORECASE)
            if rewritten == question and bool(set(normalize_text(question).split()) & {"do", "nay", "tren"}):
                rewritten = f"{previous}: {question}"
        else:
            rewritten = question

        if history_year and not re.search(r"\b20[2-3]\d\b", rewritten):
            rewritten = f"{rewritten} (Năm {history_year})"
        return rewritten

    def _rewrite_short_follow_up(self, question: str, history: Sequence[Dict[str, str]]) -> str | None:
        if not (self._is_short_action(question) or self._is_clarify(question)):
            return None

        last_user = self._last_substantive_user_question(history)
        offer = self._last_assistant_offer(history)
        focus = self._history_focus(history)

        if self._is_clarify(question) and last_user:
            return f"Làm rõ hơn câu hỏi trước: {last_user}"
        if offer:
            action = self._offer_to_query(offer)
            if action:
                return f"{action}. Ngữ cảnh: {focus}" if focus and normalize_text(focus) not in normalize_text(action) else action
        if last_user:
            prefix = "Tìm thêm thông tin để trả lời câu hỏi trước" if "tim" in normalize_text(question).split() else "Hỗ trợ trả lời tiếp câu hỏi trước"
            return f"{prefix}: {last_user}"
        return None

    def _last_substantive_user_question(self, history: Sequence[Dict[str, str]]) -> str | None:
        for item in reversed(history[-10:]):
            if item.get("role") != "user":
                continue
            content = _strip_sources(item.get("content", ""))
            if content and not self._is_short_action(content) and not self._is_clarify(content):
                return content
        return None

    @staticmethod
    def _last_assistant_offer(history: Sequence[Dict[str, str]]) -> str | None:
        for item in reversed(history[-8:]):
            if item.get("role") != "assistant":
                continue
            content = _strip_sources(item.get("content", ""))
            for sentence in reversed([part.strip(" -•\t") for part in re.split(r"(?<=[.!?])\s+|\n+", content) if part.strip()]):
                sentence_norm = normalize_text(sentence)
                has_offer = _has_any(sentence_norm, ("ban co muon", "em co muon", "neu ban can", "neu em can", "toi co the", "minh co the"))
                has_action = bool(set(sentence_norm.split()) & {"tim", "giup", "huong", "dan", "ho", "tro", "lien", "he"})
                if has_offer and has_action:
                    return sentence
        return None

    @staticmethod
    def _offer_to_query(sentence: str) -> str:
        text = re.sub(r"\[[0-9]+\]", "", sentence).strip(" \t\n?.!")
        found_marker = False
        for marker in ("tìm", "tim", "hướng dẫn", "huong dan", "hỗ trợ", "ho tro", "liên hệ", "lien he"):
            match = re.search(marker, text, flags=re.IGNORECASE)
            if match:
                text = text[match.start() :]
                found_marker = True
                break
        if not found_marker:
            return ""
        text = re.sub(r"(?i)\bgiúp\s+(?:bạn|em|tôi|mình|toi|minh)\s+", "", text)
        text = re.sub(r"(?i)\bkhông\s*$", "", text).strip(" ,.;:?!")
        return text if len(normalize_text(text).split()) >= 4 else ""

    def _history_focus(self, history: Sequence[Dict[str, str]]) -> str:
        focus: List[str] = []
        for item in history[-6:]:
            content = _strip_sources(item.get("content", ""))
            content_norm = normalize_text(content)
            focus.extend(label for needle, label in FOCUS_TERMS.items() if needle in content_norm)
            focus.extend(_extract_subjects(content))
            focus.extend(self.entities_in_text(content)[:2])
        return ", ".join(_unique_norm(focus)[:6])


class QueryPlanner:
    def __init__(self, known_terms: Sequence[str] | None = None) -> None:
        self.understanding = QueryUnderstandingAgent(known_terms=known_terms)

    def _metadata_filters(self, question: str, intent: str) -> Dict[str, Any]:
        q = normalize_text(question)
        filters: Dict[str, Any] = {}
        if intent == "tuyen_sinh" and not _department_word(question) and "trung tam" not in q:
            filters["type"] = "tuyen_sinh"
        elif "trung tam" in q:
            filters["type"] = "cac_trung_tam"
        elif _department_word(question) and intent in {"khoa_nganh", "tu_van", "so_sanh"}:
            filters["type"] = "cac_khoa"
        if year := re.search(r"\b20\d{2}\b", question):
            filters["year"] = year.group(0)
        return filters


    def _sub_questions(self, question: str, intent: str, entities: Sequence[str]) -> List[str]:
        if intent == "overview":
            return _unique_norm(
                [
                    question,
                    "Tổng quan dữ liệu IUH gồm giới thiệu trường, khoa ngành, trung tâm, đào tạo, quy định và tuyển sinh",
                    "Dữ liệu IUH có những nhóm tài liệu nào và mỗi nhóm trả lời được loại câu hỏi gì",
                ]
            )
        if entities and intent in {"so_sanh", "tu_van", "khoa_nganh"}:
            return _unique_norm([question, *[f"Thông tin chương trình đào tạo, chuẩn đầu ra và tuyển sinh của {entity}" for entity in entities]])
        if intent == "tu_van":
            return _unique_norm([question, "Các ngành đào tạo phù hợp với sở thích, năng lực và mục tiêu nghề nghiệp"])
        if intent == "tuyen_sinh":
            year = re.search(r"\b20\d{2}\b", question)
            suffix = f" năm {year.group(0)}" if year else ""
            queries = [question, f"Mã ngành, tổ hợp xét tuyển và chỉ tiêu tuyển sinh IUH{suffix}"]
            if "diem" in normalize_text(question).split():
                queries.append(f"Điểm chuẩn điểm trúng tuyển IUH{suffix} theo ngành và chương trình đào tạo")
            return _unique_norm(queries)
        if intent == "so_sanh":
            pieces = [piece.strip(" ?.,") for piece in re.split(r"\b(?:và|voi|với)\b", question, flags=re.IGNORECASE) if len(piece.strip()) > 8]
            return _unique_norm([question, *pieces])
        return [question]

    @staticmethod
    def _structured_entities(current_entities: Sequence[str], history_entities: Sequence[str]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for source, entities in (("current_question", current_entities), ("history", history_entities)):
            for entity in entities:
                if not entity:
                    continue
                rows.append({"type": "major", "value": entity, "source": source, "confidence": 0.9 if source == "current_question" else 0.75})
        seen = set()
        output = []
        for row in rows:
            key = normalize_text(row["value"])
            canonical = re.sub(r"^(nganh|khoa|chuong trinh|chuyen nganh)\s+", "", key)
            if key in seen or canonical in seen:
                continue
            seen.add(key)
            seen.add(canonical)
            output.append(row)
        return output

    @staticmethod
    def _clarification_question(missing_slots: Sequence[str], topic: str | None) -> str | None:
        if "major" not in missing_slots:
            return None
        label = topic_label(topic) or "thông tin"
        return f"Bạn muốn hỏi {label} của ngành/chương trình nào?"

    def plan(
        self,
        question: str,
        history: Sequence[Dict[str, str]] | None = None,
        role: str | None = None,
    ) -> QueryPlan:
        context = self.understanding.conversation_context(history)
        prefer_program_original = "nganh" in normalize_text(question).split()
        current_from_original = self.understanding.entities_in_text(question, prefer_program=prefer_program_original)
        followup = detect_followup(question, history, context, current_entities=current_from_original)
        topic = resolve_topic(question, context, followup)
        year = resolve_year(question, context, followup)

        rewritten_followup = build_rewritten_followup(
            question,
            context,
            followup,
            current_entities=current_from_original,
            topic=topic,
            year=year,
        )
        legacy_contextual = followup.is_followup or self.understanding.is_contextual_follow_up(question, history)
        rewritten = rewritten_followup or (self.understanding.rewrite_with_history(question, history) if legacy_contextual else question)
        intent = self.understanding.detect_intent(question, history)
        rewritten_intent = self.understanding.detect_intent(rewritten)
        topic_based_intent = topic_intent(topic)
        if topic_based_intent and (intent in {"fact", "history"} or followup.is_followup):
            intent = topic_based_intent
        elif intent == "history" or (history and rewritten != question and rewritten_intent != "fact"):
            intent = rewritten_intent
        is_topic_shift = bool(
            history
            and context.has_anchor()
            and not followup.is_followup
            and (bool(current_from_original) or (topic is not None and topic != context.active_topic) or intent in {"overview", "dao_tao_quy_dinh", "conflict_check", "missing_evidence"})
        )

        prefer_program = "nganh" in normalize_text(rewritten).split()
        current_entities = self.understanding.entities_in_text(rewritten, prefer_program=prefer_program)
        history_entities = []
        active_entity = current_from_original[0] if current_from_original else (None if is_topic_shift else context.active_entity)
        if history and active_entity and normalize_text(active_entity) not in {normalize_text(item) for item in current_entities}:
            history_entities = [active_entity]
        elif history and (previous := self.understanding._last_entity(history, prefer_program=prefer_program, exclude=current_entities)):
            history_entities = [previous]

        contextual = legacy_contextual
        action_follow_up = self.understanding._is_short_action(question) or self.understanding._is_clarify(question)
        if contextual and topic and current_from_original and intent in {"tuyen_sinh", "khoa_nganh", "dao_tao_quy_dinh", "tu_van"}:
            comparison_entities = current_entities[:1] or current_from_original[:1]
        else:
            comparison_entities = current_entities[:2] if action_follow_up else (
                _unique_norm([*current_entities[:1], *history_entities[:1]]) if contextual else current_entities[:2]
            )
        if contextual and not action_follow_up and len(comparison_entities) >= 2 and intent not in {"tuyen_sinh", "dao_tao_quy_dinh", "missing_evidence"}:
            intent = "so_sanh"

        filters = self._metadata_filters(rewritten, intent)
        if year and "year" not in filters:
            filters["year"] = year

        resolved_entity = current_entities[0] if current_entities else active_entity
        retrieval_query = build_retrieval_query(
            rewritten,
            entity=resolved_entity,
            topic=topic,
            year=year,
            original_question=question,
        )
        missing_slots = missing_slots_for(
            question,
            topic=topic,
            entity=resolved_entity,
            is_followup=followup.is_followup,
        )
        clarification = self._clarification_question(missing_slots, topic)
        evidence_constraints = build_evidence_constraints(
            entity=resolved_entity,
            topic=topic,
            year=year,
            original_question=question,
        )
        plan_query = retrieval_query or rewritten
        sub_questions = self._sub_questions(plan_query, intent, comparison_entities)
        if retrieval_query and normalize_text(retrieval_query) != normalize_text(rewritten):
            sub_questions = _unique_norm([retrieval_query, rewritten, *sub_questions])
        use_hyde = intent in {"tu_van", "so_sanh"}
        hypothesis_queries = (
            HyDEGenerator().generate_queries(plan_query, intent=intent, entities=comparison_entities or current_entities, filters=filters)
            if use_hyde
            else []
        )
        resolved_role = self.understanding.detect_role(rewritten, explicit_role=role)
        use_metadata = bool(filters)
        use_graph = intent in {"overview", "tuyen_sinh", "khoa_nganh", "so_sanh", "tu_van", "conflict_check"}
        needs_verification = True
        scope = build_scope(
            question,
            context,
            entity=resolved_entity,
            comparison_entities=comparison_entities,
            year=year,
            role=resolved_role,
        )
        retrieval_strategy = build_retrieval_strategy(
            use_history=bool(history and followup.is_followup),
            use_metadata=use_metadata,
            use_graph=use_graph,
            use_hyde=use_hyde,
            needs_verification=needs_verification,
            evidence_constraints=evidence_constraints,
        )
        risk_flags = build_risk_flags(
            is_followup=followup.is_followup,
            is_topic_shift=is_topic_shift,
            needs_clarification=bool(missing_slots),
            evidence_constraints=evidence_constraints,
            retrieval_query=plan_query,
            original_question=question,
        )
        reasoning_summary = build_reasoning_summary(
            is_followup=followup.is_followup,
            is_topic_shift=is_topic_shift,
            topic=topic,
            entity=resolved_entity,
            year=year,
            needs_clarification=bool(missing_slots),
        )

        return QueryPlan(
            original_question=question,
            rewritten_question=rewritten,
            role=resolved_role,
            intent=intent,
            filters=filters,
            sub_questions=sub_questions,
            hypothesis_queries=hypothesis_queries,
            current_entities=current_entities,
            history_entities=history_entities,
            comparison_entities=comparison_entities,
            use_hyde=use_hyde,
            use_history=bool(history),
            use_metadata=use_metadata,
            use_graph=use_graph,
            needs_verification=needs_verification,
            is_followup=followup.is_followup,
            topic=topic,
            year=year,
            entities=self._structured_entities(current_entities, history_entities),
            missing_slots=missing_slots,
            retrieval_query=retrieval_query,
            evidence_constraints=evidence_constraints,
            needs_clarification=bool(missing_slots),
            clarification_question=clarification,
            conversation_context=context.to_dict(),
            followup_reason=followup.reason,
            is_topic_shift=is_topic_shift,
            reasoning_summary=reasoning_summary,
            scope=scope,
            risk_flags=risk_flags,
            retrieval_strategy=retrieval_strategy,
        )


class HyDEGenerator:
    DEFAULT_MAX_QUERIES = 8

    def generate(
        self,
        question: str,
        intent: str = "fact",
        entities: Sequence[str] | None = None,
        filters: Dict[str, Any] | None = None,
    ) -> str:
        return " ".join(self.generate_queries(question, intent=intent, entities=entities, filters=filters))

    def generate_queries(
        self,
        question: str,
        intent: str = "fact",
        entities: Sequence[str] | None = None,
        filters: Dict[str, Any] | None = None,
        max_queries: int = DEFAULT_MAX_QUERIES,
    ) -> List[str]:
        if intent == "greeting":
            return []

        q = normalize_text(question)
        subjects = self._subjects(question, entities)
        year = str((filters or {}).get("year") or (re.search(r"\b20\d{2}\b", question).group(0) if re.search(r"\b20\d{2}\b", question) else ""))
        suffix = f" năm {year}" if year else ""
        queries: List[str]

        if intent == "overview" or _has_any(q, OVERVIEW_TERMS):
            queries = [
                "Dữ liệu IUH hiện có bao gồm những nhóm thông tin lớn nào?",
                "Tổng quan về IUH gồm lịch sử, sứ mạng, cơ cấu khoa, trung tâm và hoạt động đào tạo",
                "Tổng quan các khoa, trung tâm, ngành học, chương trình đào tạo và chuẩn đầu ra tại IUH",
                "Tổng quan quy chế đào tạo IUH: tín chỉ, học vụ, tốt nghiệp, thôi học, học bổng và quyền lợi sinh viên",
                "Tổng quan tuyển sinh IUH: phương thức xét tuyển, mã ngành, chỉ tiêu, tổ hợp và điểm chuẩn",
            ]
        elif intent == "tuyen_sinh" or _has_any(q, ADMISSION_TERMS):
            score = "diem" in q.split() or _has_any(q, ("diem chuan", "diem trung tuyen", "diem tuyen sinh"))
            queries = []
            for subject in subjects[:3]:
                target = self._target(subject)
                queries.extend(
                    [
                        f"Tổng quan tuyển sinh của {target} tại IUH{suffix} gồm những nội dung chính nào?",
                        f"Mã ngành, chỉ tiêu và tổ hợp xét tuyển của {target} tại IUH{suffix} là gì?",
                        f"Phương thức xét tuyển áp dụng cho {target} tại IUH{suffix} gồm những gì?",
                    ]
                )
                if score:
                    queries.append(f"Điểm chuẩn hoặc điểm trúng tuyển của {target} tại IUH{suffix} là bao nhiêu?")
            queries.extend([f"Thông tin tuyển sinh IUH{suffix}: mã ngành, tổ hợp, chỉ tiêu, phương thức xét tuyển"])
        elif intent == "so_sanh":
            comparison = " và ".join(self._target(subject) for subject in subjects[:3])
            queries = [
                f"{comparison} khác nhau như thế nào về chương trình đào tạo, mục tiêu và chuẩn đầu ra?",
                f"{comparison} có điểm tuyển sinh, mã ngành và tổ hợp xét tuyển khác nhau ra sao?",
                f"Cơ hội nghề nghiệp, kỹ năng cần có và mức độ phù hợp của {comparison} là gì?",
            ]
        elif intent == "tu_van":
            queries = []
            for subject in (subjects if subjects != ["nội dung được hỏi"] else self._interest_subjects(q))[:4]:
                target = self._target(subject)
                queries.extend(
                    [
                        f"{target} phù hợp với người học có sở thích và năng lực nào?",
                        f"{target} học những nội dung chính nào và yêu cầu kỹ năng gì?",
                        f"Tuyển sinh, mã ngành, tổ hợp xét tuyển và cơ hội nghề nghiệp của {target} tại IUH là gì?",
                    ]
                )
        elif intent == "khoa_nganh":
            queries = [
                item
                for subject in subjects[:3]
                for target in [self._target(subject)]
                for item in [
                    f"{target} tại IUH đào tạo những ngành hoặc chuyên ngành nào?",
                    f"Chương trình đào tạo, mục tiêu, chuẩn đầu ra và cơ hội nghề nghiệp của {target} là gì?",
                    f"Thông tin tuyển sinh, mã ngành và tổ hợp xét tuyển liên quan đến {target} là gì?",
                ]
            ]
        elif intent == "dao_tao_quy_dinh":
            queries = [
                f"Quy định đào tạo của IUH liên quan đến câu hỏi: {question}",
                "Điều kiện, quy trình, thời hạn, biểu mẫu và đối tượng áp dụng trong quy chế đào tạo IUH là gì?",
                "Các trường hợp học bổng, tín chỉ, tốt nghiệp, thôi học hoặc xử lý học vụ tại IUH",
            ]
        elif intent == "missing_evidence":
            queries = [
                f"IUH có công bố thông tin chính thức về {question} hay chưa?",
                f"Dữ liệu IUH có chứa thông tin theo năm, ngành, chương trình hoặc học phí liên quan đến {question} không?",
                "Thông báo mới nhất, năm áp dụng và nguồn chính thức cần kiểm tra trước khi trả lời",
            ]
        else:
            queries = [
                f"Nội dung chính trong tài liệu IUH liên quan trực tiếp đến: {question}",
                f"Câu hỏi {question} cần những số liệu, điều kiện, năm áp dụng và nguồn nào?",
                f"Các văn bản hoặc trang thông tin IUH có thể trả lời câu hỏi: {question}",
            ]

        subject_text = ", ".join(subjects[:2])
        queries.extend(
            [
                f"Tóm tắt tài liệu IUH về {subject_text}{suffix} để trả lời: {question}",
                f"Các ý chính, số liệu quan trọng và nguồn chính thức liên quan đến {subject_text}{suffix}",
            ]
        )
        return _unique_norm(queries)[:max_queries]

    def _subjects(self, question: str, entities: Sequence[str] | None) -> List[str]:
        q = normalize_text(question)
        subjects = [self._clean_label(entity) for entity in entities or []]
        subjects.extend(_extract_subjects(question))
        for phrase, label in [
            ("tang cuong tieng anh", "chương trình tăng cường tiếng Anh"),
            ("dai tra", "chương trình đại trà"),
            ("chat luong cao", "chương trình chất lượng cao"),
            ("tich hop", "chương trình tích hợp"),
        ]:
            if phrase in q:
                subjects.append(label)
        if not subjects and _has_any(q, ("iuh", "truong", "dai hoc cong nghiep")):
            subjects.append("IUH")
        return _unique_norm(subjects or ["nội dung được hỏi"])

    @staticmethod
    def _target(subject: str) -> str:
        subject = HyDEGenerator._clean_label(subject)
        subject_norm = normalize_text(subject)
        if subject_norm in {"iuh", "truong iuh", "dai hoc cong nghiep tphcm", "truong dai hoc cong nghiep tphcm"}:
            return "IUH"
        if subject_norm.startswith(("nganh ", "chuong trinh ", "trung tam ")) or (
            subject_norm.startswith("khoa ") and not subject_norm.startswith("khoa hoc ")
        ):
            return subject
        return subject if subject_norm == "noi dung duoc hoi" else f"ngành/chương trình {subject}"

    @staticmethod
    def _clean_label(value: str) -> str:
        value = str(value).strip()
        return pretty_label(value) if "_" in value else re.sub(r"\s+", " ", value)

    @staticmethod
    def _interest_subjects(question_norm: str) -> List[str]:
        interests: List[str] = []
        if _has_any(question_norm, ("ai", "tri tue nhan tao", "lap trinh", "du lieu", "toan")):
            interests.extend(["Công nghệ thông tin", "Khoa học dữ liệu", "Khoa học máy tính", "Kỹ thuật phần mềm"])
        if _has_any(question_norm, ("dien", "tu dong", "nang luong")):
            interests.extend(["Công nghệ kỹ thuật điện", "Công nghệ kỹ thuật điều khiển và tự động hóa"])
        if _has_any(question_norm, ("kinh doanh", "marketing", "ban hang", "quan tri")):
            interests.extend(["Quản trị kinh doanh", "Marketing", "Thương mại điện tử"])
        return interests or ["các ngành đào tạo phù hợp"]

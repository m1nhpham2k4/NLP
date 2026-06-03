from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Sequence

from .text import normalize_text


EntityExtractor = Callable[[str], Sequence[str]]


TOPIC_SPECS: Dict[str, Dict[str, Any]] = {
    "diem_chuan": {
        "label": "điểm chuẩn",
        "intent": "tuyen_sinh",
        "entity_required": True,
        "keywords": (
            "diem chuan",
            "diem trung tuyen",
            "diem tuyen sinh",
            "bao nhieu diem",
            "diem bao nhieu",
            "lay bao nhieu",
            "muc diem",
        ),
    },
    "to_hop_mon": {
        "label": "tổ hợp môn xét tuyển",
        "intent": "tuyen_sinh",
        "entity_required": True,
        "keywords": (
            "to hop",
            "to hop mon",
            "mon xet tuyen",
            "hoc mon gi",
            "mon gi",
            "mon nao",
            "khoi nao",
            "de dau",
        ),
    },
    "dieu_kien_xet_tuyen": {
        "label": "điều kiện xét tuyển",
        "intent": "tuyen_sinh",
        "entity_required": False,
        "keywords": (
            "dieu kien xet tuyen",
            "dieu kien tuyen sinh",
            "can gi",
            "yeu cau gi",
            "lam sao de dau",
            "lam sao de xet tuyen",
        ),
    },
    "chuong_trinh_hoc": {
        "label": "chương trình đào tạo",
        "intent": "khoa_nganh",
        "entity_required": True,
        "keywords": (
            "chuong trinh dao tao",
            "hoc nhung gi",
            "noi dung dao tao",
            "mon hoc",
            "chuan dau ra",
        ),
    },
    "hoc_phi": {
        "label": "học phí",
        "intent": "dao_tao_quy_dinh",
        "entity_required": False,
        "keywords": ("hoc phi", "dong tien", "muc phi", "phi dao tao"),
    },
    "hoc_bong": {
        "label": "học bổng",
        "intent": "dao_tao_quy_dinh",
        "entity_required": False,
        "keywords": ("hoc bong", "mien giam", "chinh sach ho tro"),
    },
    "viec_lam": {
        "label": "cơ hội việc làm",
        "intent": "tu_van",
        "entity_required": True,
        "keywords": (
            "co hoi viec lam",
            "viec lam",
            "ra truong lam gi",
            "nghe nghiep",
            "co hoi nghe nghiep",
        ),
    },
    "tuyen_sinh": {
        "label": "thông tin tuyển sinh",
        "intent": "tuyen_sinh",
        "entity_required": False,
        "keywords": (
            "tuyen sinh",
            "xet tuyen",
            "phuong thuc",
            "chi tieu",
            "ma nganh",
        ),
    },
}


FOLLOWUP_PATTERN_GROUPS: Dict[str, Sequence[str]] = {
    "transition": (
        r"^(vay|the|con|ca|cung|va|voi|o)\b",
        r"\bthi sao\b",
    ),
    "reference": (
        r"\b(nganh do|nganh nay|khoa do|khoa nay|chuong trinh do|chuong trinh nay|nam do|nam nay|o do|cai do|noi tren|tren do|do|nay)\b",
    ),
    "short_topic": (
        r"^(diem|diem bao nhieu|bao nhieu diem|lay bao nhieu|lay bao nhieu diem)$",
        r"^(nam gan nhat|moi nhat|gan nhat)$",
        r"^(hoc mon gi|mon gi|mon nao|to hop nao)$",
        r"^(co hoi viec lam|ra truong lam gi|viec lam)$",
        r"^(dieu kien xet tuyen|can gi|yeu cau gi)$",
    ),
    "year_only": (
        r"^20[2-3]\d$",
        r"^nam 20[2-3]\d$",
    ),
}


RECENT_YEAR_TERMS = ("nam gan nhat", "moi nhat", "gan nhat", "nam moi nhat")
BROAD_SCOPE_TERMS = ("cac nganh", "tat ca", "toan bo", "iuh", "truong", "dai hoc cong nghiep")

EDUCATION_LEVEL_KEYWORDS: Dict[str, Sequence[str]] = {
    "dai_hoc": ("dai hoc", "cu nhan", "ky su"),
    "cao_dang": ("cao dang",),
    "sau_dai_hoc": ("sau dai hoc",),
    "thac_si": ("thac si", "cao hoc"),
    "tien_si": ("tien si",),
    "van_bang_2": ("van bang 2", "bang hai"),
}

ADMISSION_TYPE_KEYWORDS: Dict[str, Sequence[str]] = {
    "xet_hoc_ba": ("hoc ba", "xet hoc ba"),
    "thi_thpt": ("thi thpt", "tot nghiep thpt", "diem thi"),
    "danh_gia_nang_luc": ("danh gia nang luc", "dgnl"),
    "tuyen_thang": ("tuyen thang", "uu tien xet tuyen"),
    "xet_tuyen": ("xet tuyen",),
    "tuyen_sinh_dai_hoc": ("tuyen sinh dai hoc",),
    "tuyen_sinh_thac_si": ("tuyen sinh thac si", "cao hoc"),
}

AUDIENCE_KEYWORDS: Dict[str, Sequence[str]] = {
    "thi_sinh": ("thi sinh", "em muon xet tuyen", "dang ky xet tuyen"),
    "phu_huynh": ("phu huynh", "con toi", "con em"),
    "sinh_vien": ("sinh vien", "dang hoc", "hoc lai", "tot nghiep"),
    "can_bo_tu_van": ("can bo tu van", "tu van tuyen sinh"),
}


@dataclass
class ConversationContext:
    active_entity: str | None = None
    active_major: str | None = None
    active_topic: str | None = None
    active_year: str | None = None
    last_intent: str | None = None
    last_question: str | None = None
    last_rewritten_question: str | None = None
    last_answer_summary: str | None = None
    source_messages: int = 0

    def has_anchor(self) -> bool:
        return bool(self.active_entity or self.active_topic or self.active_year or self.last_question)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "active_entity": self.active_entity,
            "active_major": self.active_major,
            "active_topic": self.active_topic,
            "active_year": self.active_year,
            "last_intent": self.last_intent,
            "last_question": self.last_question,
            "last_rewritten_question": self.last_rewritten_question,
            "source_messages": self.source_messages,
        }


@dataclass
class FollowupDecision:
    is_followup: bool = False
    reason: str | None = None
    confidence: float = 0.0
    topic_hint: str | None = None


def strip_sources(text: str) -> str:
    return re.split(r"\n\s*Nguồn\s*:", str(text), maxsplit=1, flags=re.IGNORECASE)[0].strip()


def _has_phrase(text_norm: str, phrases: Sequence[str]) -> bool:
    tokens = set(text_norm.split())
    return any((phrase in text_norm if " " in phrase else phrase in tokens) for phrase in phrases)


def extract_year(text: str) -> str | None:
    match = re.search(r"\b20[2-3]\d\b", str(text))
    return match.group(0) if match else None


def has_recent_year_request(text: str) -> bool:
    return _has_phrase(normalize_text(text), RECENT_YEAR_TERMS)


def detect_topic(text: str) -> str | None:
    text_norm = normalize_text(text)
    if not text_norm:
        return None
    for topic, spec in TOPIC_SPECS.items():
        if _has_phrase(text_norm, spec["keywords"]):
            return topic
    return None


def topic_label(topic: str | None) -> str | None:
    if not topic:
        return None
    spec = TOPIC_SPECS.get(topic)
    return str(spec["label"]) if spec else topic


def topic_intent(topic: str | None) -> str | None:
    if not topic:
        return None
    spec = TOPIC_SPECS.get(topic)
    return str(spec["intent"]) if spec else None


def topic_requires_entity(topic: str | None) -> bool:
    if not topic:
        return False
    spec = TOPIC_SPECS.get(topic)
    return bool(spec and spec.get("entity_required"))


def _first_keyword_label(text_norm: str, mapping: Dict[str, Sequence[str]]) -> str:
    for label, keywords in mapping.items():
        if _has_phrase(text_norm, keywords):
            return label
    return "unknown"


def detect_education_level(text: str) -> str:
    return _first_keyword_label(normalize_text(text), EDUCATION_LEVEL_KEYWORDS)


def detect_admission_type(text: str) -> str:
    return _first_keyword_label(normalize_text(text), ADMISSION_TYPE_KEYWORDS)


def detect_target_audience(text: str, role: str | None = None) -> str:
    if role and role != "general":
        return role
    return _first_keyword_label(normalize_text(text), AUDIENCE_KEYWORDS)


def infer_entity_type(entity: str | None) -> str:
    if not entity:
        return "unknown"
    entity_norm = normalize_text(entity)
    if entity_norm.startswith("khoa "):
        return "department"
    if entity_norm.startswith("trung tam "):
        return "center"
    if entity_norm.startswith(("chuong trinh ", "nganh ", "chuyen nganh ")):
        return "program"
    return "program"


def _metadata_list(item: Dict[str, Any], key: str) -> List[Any]:
    values: List[Any] = []
    if item.get(key):
        values.append(item.get(key))
    metadata = item.get("metadata")
    if isinstance(metadata, dict) and metadata.get(key):
        values.append(metadata.get(key))
    return values


def _first_metadata_value(item: Dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        for value in _metadata_list(item, key):
            if isinstance(value, list) and value:
                return str(value[0])
            if value:
                return str(value)
    return None


def build_conversation_context(
    history: Sequence[Dict[str, Any]] | None,
    entity_extractor: EntityExtractor,
    *,
    max_messages: int = 12,
) -> ConversationContext:
    if not history:
        return ConversationContext()

    recent = list(history[-max_messages:])
    context = ConversationContext(source_messages=len(recent))

    for item in reversed(recent):
        role = item.get("role")
        content = strip_sources(str(item.get("content", "")))
        if role == "user" and content and context.last_question is None:
            context.last_question = content
        if role == "assistant" and content and context.last_answer_summary is None:
            context.last_answer_summary = content[:400]
        if context.last_rewritten_question is None:
            context.last_rewritten_question = _first_metadata_value(item, "rewritten_question")

    for item in reversed(recent):
        if item.get("role") != "user":
            continue
        content = strip_sources(str(item.get("content", "")))
        entities = list(entity_extractor(content))
        if entities:
            context.active_entity = str(entities[0])
            context.active_major = context.active_entity
            break

    if context.active_entity is None:
        for item in reversed(recent):
            content = strip_sources(str(item.get("content", "")))
            metadata_entity = _first_metadata_value(item, "active_entity", "active_major")
            entities = [metadata_entity] if metadata_entity else list(entity_extractor(content))
            if entities:
                context.active_entity = str(entities[0])
                context.active_major = context.active_entity
                break

    for item in reversed(recent):
        content = strip_sources(str(item.get("content", "")))
        context.active_topic = _first_metadata_value(item, "topic") or detect_topic(content)
        if context.active_topic:
            break

    for item in reversed(recent):
        if item.get("role") != "user":
            continue
        year = _first_metadata_value(item, "year") or extract_year(str(item.get("content", "")))
        if year:
            context.active_year = year
            break

    if context.active_year is None:
        for item in reversed(recent):
            year = _first_metadata_value(item, "year") or extract_year(str(item.get("content", "")))
            if year:
                context.active_year = year
                break

    if context.active_topic:
        context.last_intent = topic_intent(context.active_topic)
    return context


def _matches_followup_pattern(text_norm: str) -> tuple[str | None, float]:
    for group, patterns in FOLLOWUP_PATTERN_GROUPS.items():
        for pattern in patterns:
            if re.search(pattern, text_norm):
                confidence = 0.9 if group in {"reference", "year_only"} else 0.82
                return group, confidence
    return None, 0.0


def detect_followup(
    question: str,
    history: Sequence[Dict[str, Any]] | None,
    context: ConversationContext,
    *,
    current_entities: Sequence[str] | None = None,
) -> FollowupDecision:
    if not history or not context.has_anchor():
        return FollowupDecision()

    q = normalize_text(question)
    if not q:
        return FollowupDecision()

    topic_hint = detect_topic(question)
    group, confidence = _matches_followup_pattern(q)
    if group:
        return FollowupDecision(True, group, confidence, topic_hint=topic_hint)

    tokens = q.split()
    if len(tokens) <= 8 and topic_hint and topic_hint not in {"hoc_bong"}:
        return FollowupDecision(True, "short_topic_with_context", 0.78, topic_hint=topic_hint)

    if len(tokens) <= 8 and current_entities and (context.active_topic or context.active_year):
        return FollowupDecision(True, "entity_switch", 0.76, topic_hint=topic_hint)

    if len(tokens) <= 5 and has_recent_year_request(question):
        return FollowupDecision(True, "recent_year", 0.84, topic_hint=topic_hint)

    return FollowupDecision(False, None, 0.0, topic_hint=topic_hint)


def _clean_entity(entity: str | None) -> str | None:
    if not entity:
        return None
    return re.sub(r"\s+", " ", str(entity)).strip(" .,:;?!")


def _target_phrase(entity: str | None) -> str:
    cleaned = _clean_entity(entity)
    if not cleaned:
        return "IUH"
    entity_norm = normalize_text(cleaned)
    if entity_norm.startswith(("nganh ", "khoa ", "chuong trinh ", "trung tam ")):
        return cleaned
    return f"ngành {cleaned}"


def _canonical_entity(entity: str | None) -> str | None:
    cleaned = _clean_entity(entity)
    if not cleaned:
        return None
    return re.sub(
        r"^(ngành|nganh|chương trình|chuong trinh|chuyên ngành|chuyen nganh)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()


def resolve_topic(question: str, context: ConversationContext, followup: FollowupDecision) -> str | None:
    return detect_topic(question) or (context.active_topic if followup.is_followup else None)


def resolve_year(question: str, context: ConversationContext, followup: FollowupDecision) -> str | None:
    explicit = extract_year(question)
    if explicit:
        return explicit
    if followup.is_followup and not has_recent_year_request(question):
        return context.active_year
    return None


def build_rewritten_followup(
    question: str,
    context: ConversationContext,
    followup: FollowupDecision,
    *,
    current_entities: Sequence[str] | None = None,
    topic: str | None = None,
    year: str | None = None,
) -> str | None:
    if not followup.is_followup:
        return None

    entity = _clean_entity((current_entities or [None])[0]) or context.active_entity
    topic = topic or context.active_topic
    recent = has_recent_year_request(question)
    year_text = " năm gần nhất" if recent else (f" năm {year}" if year else "")
    target = _target_phrase(entity)

    if topic == "diem_chuan":
        return f"Điểm chuẩn{year_text} của {target} IUH là bao nhiêu?"
    if topic == "to_hop_mon":
        return f"Tổ hợp môn xét tuyển{year_text} của {target} IUH là gì?"
    if topic == "dieu_kien_xet_tuyen":
        return f"Điều kiện xét tuyển{year_text} của {target} IUH là gì?"
    if topic == "chuong_trinh_hoc":
        return f"Chương trình đào tạo của {target} IUH học những môn gì?"
    if topic == "hoc_phi":
        return f"Học phí{year_text} của {target} IUH là bao nhiêu?"
    if topic == "hoc_bong":
        return f"Học bổng{year_text} của {target} IUH như thế nào?"
    if topic == "viec_lam":
        return f"Cơ hội việc làm của {target} IUH là gì?"
    if topic == "tuyen_sinh":
        return f"Thông tin tuyển sinh{year_text} của {target} là gì?"
    if entity:
        return f"Thông tin về {target} IUH liên quan đến câu hỏi: {question}"
    return None


def build_retrieval_query(
    rewritten_question: str,
    *,
    entity: str | None = None,
    topic: str | None = None,
    year: str | None = None,
    original_question: str | None = None,
) -> str:
    recent = has_recent_year_request(original_question or "")
    # Do not inject the literal phrase "năm gần nhất" into retrieval: it over-matches
    # documents whose title says "2 năm gần nhất" even when the entity/topic is more important.
    year_text = "" if recent else (f"năm {year}" if year else "")
    target = _target_phrase(entity) if entity else ""

    if topic == "diem_chuan":
        pieces = ["điểm chuẩn điểm trúng tuyển", target, year_text, "IUH"]
    elif topic == "to_hop_mon":
        pieces = ["tổ hợp môn xét tuyển", target, year_text, "IUH"]
    elif topic == "dieu_kien_xet_tuyen":
        pieces = ["điều kiện phương thức xét tuyển", target, year_text, "IUH"]
    elif topic == "chuong_trinh_hoc":
        pieces = ["chương trình đào tạo môn học chuẩn đầu ra", target, "IUH"]
    elif topic == "hoc_phi":
        pieces = ["học phí", target, year_text, "IUH"]
    elif topic == "hoc_bong":
        pieces = ["học bổng miễn giảm chính sách hỗ trợ", target, year_text, "IUH"]
    elif topic == "viec_lam":
        pieces = ["cơ hội việc làm nghề nghiệp", target, "IUH"]
    elif topic == "tuyen_sinh":
        pieces = ["thông tin tuyển sinh mã ngành chỉ tiêu tổ hợp", target, year_text, "IUH"]
    else:
        return rewritten_question

    query = " ".join(piece for piece in pieces if piece).strip()
    return re.sub(r"\s+", " ", query) or rewritten_question


def build_evidence_constraints(
    *,
    entity: str | None = None,
    topic: str | None = None,
    year: str | None = None,
    original_question: str | None = None,
) -> Dict[str, Any]:
    constraints: Dict[str, Any] = {}
    if entity:
        constraints["major"] = _canonical_entity(entity) or entity
        constraints["entity"] = entity
    if topic:
        constraints["topic"] = topic
    if year:
        constraints["year"] = year
    elif has_recent_year_request(original_question or ""):
        constraints["year_scope"] = "latest"
    return constraints


def missing_slots_for(
    question: str,
    *,
    topic: str | None,
    entity: str | None,
    is_followup: bool,
) -> List[str]:
    if not topic_requires_entity(topic) or entity:
        return []
    q = normalize_text(question)
    if _has_phrase(q, BROAD_SCOPE_TERMS):
        return []
    if is_followup or len(q.split()) <= 7:
        return ["major"]
    return []


def build_scope(
    question: str,
    context: ConversationContext,
    *,
    entity: str | None,
    comparison_entities: Sequence[str],
    year: str | None,
    role: str | None = None,
) -> Dict[str, Any]:
    context_text = " ".join(
        item
        for item in (
            question,
            context.last_question or "",
            context.last_rewritten_question or "",
        )
        if item
    )
    return {
        "education_level": detect_education_level(context_text),
        "admission_type": detect_admission_type(context_text),
        "year": year,
        "main_entity": _canonical_entity(entity) if entity else None,
        "main_entity_type": infer_entity_type(entity),
        "comparison_entities": list(comparison_entities),
        "target_audience": detect_target_audience(context_text, role=role),
    }


def build_retrieval_strategy(
    *,
    use_history: bool,
    use_metadata: bool,
    use_graph: bool,
    use_hyde: bool,
    needs_verification: bool,
    evidence_constraints: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "use_history": use_history,
        "use_metadata": use_metadata,
        "use_graph": use_graph,
        "use_hyde": use_hyde,
        "needs_verification": needs_verification,
        "should_abstain_if_scope_mismatch": bool(evidence_constraints),
    }


def build_risk_flags(
    *,
    is_followup: bool,
    is_topic_shift: bool,
    needs_clarification: bool,
    evidence_constraints: Dict[str, Any],
    retrieval_query: str,
    original_question: str,
) -> List[str]:
    flags: List[str] = []
    if needs_clarification:
        flags.append("missing_required_scope")
    if is_followup:
        flags.append("followup_rewritten_from_history")
    if is_topic_shift:
        flags.append("topic_shift_do_not_force_previous_context")
    if evidence_constraints.get("year_scope") == "latest":
        flags.append("latest_year_requires_newest_matching_evidence")
    if evidence_constraints.get("major"):
        flags.append("entity_scope_requires_matching_evidence")
    if normalize_text(retrieval_query) == normalize_text(original_question) and len(normalize_text(original_question).split()) <= 5:
        flags.append("short_query_without_expansion")
    return flags


def build_reasoning_summary(
    *,
    is_followup: bool,
    is_topic_shift: bool,
    topic: str | None,
    entity: str | None,
    year: str | None,
    needs_clarification: bool,
) -> str:
    if needs_clarification:
        return "Câu hỏi thiếu scope bắt buộc nên cần hỏi lại trước khi retrieval."
    if is_topic_shift:
        return "Câu hỏi hiện tại có chủ đề hoặc thực thể mới rõ ràng nên không ép dùng ngữ cảnh cũ."
    if is_followup:
        pieces = ["Câu hỏi là follow-up; đã khôi phục ngữ cảnh từ history"]
        if topic:
            pieces.append(f"topic={topic}")
        if entity:
            pieces.append(f"entity={_canonical_entity(entity) or entity}")
        if year:
            pieces.append(f"year={year}")
        return "; ".join(pieces) + "."
    return "Câu hỏi đủ độc lập; planner dùng câu hỏi hiện tại làm scope chính."

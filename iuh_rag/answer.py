from __future__ import annotations

import os
import re
from typing import Dict, List, Sequence

from .config import DEFAULT_OPENAI_MODEL
from .retrievers import SearchResult
from .text import sentence_split, token_overlap_score
from .verification import detect_conflicts, evidence_is_weak, self_correct_answer, verify_claims


ROLE_PREFIX = {
    "thi_sinh": "Dành cho thí sinh: ",
    "phu_huynh": "Dành cho phụ huynh: ",
    "sinh_vien": "Dành cho sinh viên: ",
    "can_bo_tu_van": "Dành cho cán bộ tư vấn: ",
    "general": "",
}

# Số câu tối đa theo role — thi_sinh/phu_huynh nhận câu trả lời ngắn hơn
ROLE_MAX_SENTENCES = {
    "thi_sinh": 4,
    "phu_huynh": 4,
    "sinh_vien": 6,
    "can_bo_tu_van": 8,
    "general": 5,
}

# Ngưỡng overlap tối thiểu theo role
ROLE_MIN_SCORE = {
    "thi_sinh": 0.07,
    "phu_huynh": 0.07,
    "sinh_vien": 0.06,
    "can_bo_tu_van": 0.05,
    "general": 0.08,
}

OVERVIEW_TYPE_SUMMARIES = {
    "tong_quan_ve_truong": "giới thiệu chung về trường, tầm nhìn, sứ mạng, mục tiêu và bối cảnh IUH",
    "cac_khoa": "các khoa, ngành/chương trình đào tạo, mục tiêu đào tạo, chuẩn đầu ra và cơ hội nghề nghiệp",
    "cac_trung_tam": "các trung tâm, đơn vị hỗ trợ, thư viện, cơ sở vật chất và dịch vụ phục vụ người học",
    "dao_tao": "quy chế đào tạo, tín chỉ, học vụ, tốt nghiệp, học bổng và các quy định dành cho sinh viên",
    "tuyen_sinh": "tuyển sinh, phương thức xét tuyển, mã ngành, tổ hợp, chỉ tiêu và điểm chuẩn theo năm",
}

OVERVIEW_TYPE_ORDER = [
    "tong_quan_ve_truong",
    "cac_khoa",
    "cac_trung_tam",
    "dao_tao",
    "tuyen_sinh",
]

ROLE_INTRO: Dict[str, Dict[str, str]] = {
    "thi_sinh": {
        "tu_van": "Thông tin sau đây sẽ giúp em định hướng lựa chọn ngành phù hợp: ",
        "so_sanh": "Dưới đây là phần so sánh để em cân nhắc lựa chọn ngành phù hợp: ",
        "tuyen_sinh": "Thông tin tuyển sinh em cần biết: ",
        "default": "Theo thông tin tuyển sinh hiện có: ",
    },
    "phu_huynh": {
        "tu_van": "Dưới đây là thông tin để phụ huynh tham khảo khi định hướng cho con: ",
        "so_sanh": "Dưới đây là phần so sánh để phụ huynh tham khảo khi định hướng cho con: ",
        "tuyen_sinh": "Thông tin tuyển sinh hiện có: ",
        "default": "Theo dữ liệu hiện có: ",
    },
    "sinh_vien": {
        "dao_tao_quy_dinh": "Theo quy định đào tạo hiện hành: ",
        "default": "Theo dữ liệu hiện có: ",
    },
    "can_bo_tu_van": {
        "default": "Theo các tài liệu được truy xuất (có kèm trích dẫn nguồn): ",
        "conflict_check": "Hệ thống phát hiện có sự khác biệt giữa các tài liệu được truy xuất: ",
    },
    "general": {
        "overview": "Tổng quan dữ liệu hiện có: ",
        "tu_van": "Dựa trên dữ liệu hiện có, các thông tin liên quan nhất để tư vấn là: ",
        "so_sanh": "Dựa trên các đoạn tài liệu được truy xuất, có thể đối chiếu như sau: ",
        "default": "Theo dữ liệu hiện có: ",
    },
}


def _get_intro(role: str, intent: str) -> str:
    """Get role+intent-specific intro text."""
    role_map = ROLE_INTRO.get(role, ROLE_INTRO["general"])
    return role_map.get(intent, role_map.get("default", "Theo dữ liệu hiện có: "))


class AnswerGenerator:
    def __init__(self, min_sentence_score: float = 0.08) -> None:
        self.min_sentence_score = min_sentence_score

    @staticmethod
    def _candidate_sentences(item: SearchResult) -> List[str]:
        text = item.chunk.get("text", "")
        metadata = item.chunk.get("metadata", {})
        if item.chunk.get("type") == "faq" or metadata.get("source_kind") == "faq":
            answer_text = re.split(r"\bTrả lời\s*:", text, maxsplit=1, flags=re.IGNORECASE)
            if len(answer_text) == 2:
                cleaned_answer = answer_text[1].strip()
                if len(cleaned_answer) <= 1000:
                    return [cleaned_answer]
                return sentence_split(cleaned_answer)
        return sentence_split(text)

    def _select_sentences(
        self,
        question: str,
        evidence: Sequence[SearchResult],
        max_sentences: int = 5,
        min_score: float | None = None,
    ) -> List[str]:
        threshold = min_score if min_score is not None else self.min_sentence_score
        scored = []
        for item in evidence:
            for sentence in self._candidate_sentences(item):
                if sentence.startswith("***"):
                    continue
                score = token_overlap_score(question, sentence)
                if score >= threshold:
                    scored.append((score, sentence.strip(), item.rank))
        scored = sorted(scored, key=lambda row: (row[0], -row[2]), reverse=True)

        selected = []
        seen = set()
        for _, sentence, _ in scored:
            normalized = sentence.lower()
            if normalized in seen:
                continue
            selected.append(sentence)
            seen.add(normalized)
            if len(selected) >= max_sentences:
                break

        if not selected and evidence:
            fallback = [sentence for sentence in self._candidate_sentences(evidence[0]) if not sentence.startswith("***")]
            selected = fallback[:1]
        return selected

    def _format_sources(self, evidence: Sequence[SearchResult], role: str = "general") -> str:
        lines = ["Nguồn:"]
        for idx, item in enumerate(evidence, start=1):
            if role == "can_bo_tu_van":
                # Detailed citation for advisors
                lines.append(
                    f"[{idx}] {item.chunk.get('file_name')} | {item.chunk_id} | {item.chunk.get('relative_path')} | score={item.score:.4f}"
                )
            else:
                lines.append(
                    f"[{idx}] {item.chunk.get('file_name')} | {item.chunk_id} | {item.chunk.get('relative_path')}"
                )
        return "\n".join(lines)

    def _overview_answer(self, evidence: Sequence[SearchResult], role: str = "general") -> str:
        grouped: Dict[str, List[int]] = {}
        for idx, item in enumerate(evidence, start=1):
            chunk_type = str(item.chunk.get("type") or "khac")
            grouped.setdefault(chunk_type, []).append(idx)

        lines = []
        for chunk_type in OVERVIEW_TYPE_ORDER:
            refs = grouped.get(chunk_type)
            if not refs:
                continue
            ref_text = ", ".join(f"[{idx}]" for idx in refs[:2])
            lines.append(f"- {OVERVIEW_TYPE_SUMMARIES[chunk_type]} {ref_text}.")

        for chunk_type, refs in grouped.items():
            if chunk_type in OVERVIEW_TYPE_SUMMARIES:
                continue
            ref_text = ", ".join(f"[{idx}]" for idx in refs[:2])
            lines.append(f"- nhóm dữ liệu khác liên quan đến câu hỏi {ref_text}.")

        prefix = ROLE_PREFIX.get(role, "") + _get_intro(role, "overview")
        if not lines:
            lines = ["- chưa đủ bằng chứng đa nhóm để mô tả toàn cảnh dữ liệu."]
        answer = prefix + "\n" + "\n".join(lines)
        return answer.strip() + "\n\n" + self._format_sources(evidence, role=role)

    def generate(
        self,
        question: str,
        evidence: Sequence[SearchResult],
        role: str = "general",
        intent: str = "fact",
        verify: bool = True,
        history: Sequence[Dict[str, str]] | None = None,
        evidence_constraints: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        if evidence_is_weak(evidence):
            if intent == "missing_evidence":
                answer = (
                    "Tôi chưa tìm thấy thông tin này trong dữ liệu hiện có của hệ thống. "
                    "Vui lòng kiểm tra tài liệu chính thức hoặc liên hệ trực tiếp với nhà trường."
                )
            else:
                answer = (
                    "Tôi chưa tìm thấy đủ bằng chứng trong dữ liệu hiện có để trả lời chắc chắn. "
                    "Bạn nên kiểm tra lại tài liệu chính thức hoặc bổ sung dữ liệu liên quan."
                )
            return {
                "answer": answer,
                "claim_verification": [],
                "conflicts": [],
                "confidence": "low",
                "abstained": True,
            }

        conflicts = detect_conflicts(question, evidence)
        if intent == "overview":
            type_count = len({item.chunk.get("type") for item in evidence if item.chunk.get("type")})
            return {
                "answer": self._overview_answer(evidence, role=role),
                "claim_verification": [],
                "conflicts": conflicts,
                "confidence": "high" if type_count >= 4 else "medium",
                "abstained": False,
            }

        if conflicts and intent == "conflict_check":
            answer = "Tôi phát hiện có dấu hiệu khác nhau giữa các nguồn được truy xuất. "
            answer += "Cần ưu tiên tài liệu chính thức hoặc nguồn mới hơn trước khi kết luận.\n\n"
            answer += self._format_sources(evidence, role=role)
            return {
                "answer": answer,
                "claim_verification": [],
                "conflicts": conflicts,
                "confidence": "medium",
                "abstained": False,
            }

        max_sentences = ROLE_MAX_SENTENCES.get(role, 5)
        min_score = ROLE_MIN_SCORE.get(role, self.min_sentence_score)
        sentences = self._select_sentences(question, evidence, max_sentences=max_sentences, min_score=min_score)
        prefix = ROLE_PREFIX.get(role, "")
        intro = prefix + _get_intro(role, intent)

        answer = intro + " ".join(sentences)
        answer = answer.strip() + "\n\n" + self._format_sources(evidence, role=role)

        verdicts = verify_claims(answer, evidence) if verify else []
        if verify:
            answer = self_correct_answer(answer, verdicts)

        confidence = "high"
        if evidence[0].score < 0.12 or any(item["status"] == "unsupported" for item in verdicts):
            confidence = "medium"

        return {
            "answer": answer,
            "claim_verification": verdicts,
            "conflicts": conflicts,
            "confidence": confidence,
            "abstained": False,
        }


class OpenAIAnswerGenerator(AnswerGenerator):
    def __init__(self, model: str = DEFAULT_OPENAI_MODEL, max_evidence_chars: int = 1400) -> None:
        super().__init__()
        self.model = model
        self.max_evidence_chars = max_evidence_chars
        self.client = None

    def _client(self):
        if self.client is not None:
            return self.client
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set. Set it before using --openai.")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("The openai package is not installed. Run: pip install openai") from exc
        self.client = OpenAI()
        return self.client

    def _evidence_block(self, evidence: Sequence[SearchResult]) -> str:
        blocks = []
        for idx, item in enumerate(evidence, start=1):
            text = item.chunk.get("text", "").strip()
            if len(text) > self.max_evidence_chars:
                text = text[: self.max_evidence_chars].rstrip() + "..."
            blocks.append(
                "\n".join(
                    [
                        f"[{idx}] chunk_id: {item.chunk_id}",
                        f"file_name: {item.chunk.get('file_name')}",
                        f"relative_path: {item.chunk.get('relative_path')}",
                        f"text: {text}",
                    ]
                )
            )
        return "\n\n".join(blocks)

    def _role_instruction(self, role: str) -> str:
        instructions = {
            "thi_sinh": (
                "Người hỏi là thí sinh. Hãy trả lời ngắn gọn, dễ hiểu, tránh thuật ngữ phức tạp. "
                "Tập trung vào thông tin tuyển sinh, ngành học và cơ hội."
            ),
            "phu_huynh": (
                "Người hỏi là phụ huynh. Hãy trả lời thân thiện, dễ hiểu, không dùng thuật ngữ chuyên ngành. "
                "Tập trung vào thông tin thiết thực cho gia đình."
            ),
            "sinh_vien": (
                "Người hỏi là sinh viên đang học. Hãy trả lời chi tiết, bao gồm điều khoản, quy định, số liệu cụ thể. "
                "Có thể dùng thuật ngữ chuyên ngành đào tạo."
            ),
            "can_bo_tu_van": (
                "Người hỏi là cán bộ tư vấn. Hãy trả lời đầy đủ, chính xác, kèm trích dẫn [1], [2] rõ ràng. "
                "Đề cập nguồn tài liệu, điều khoản cụ thể để có thể tra cứu lại."
            ),
            "general": "Trả lời bằng tiếng Việt tự nhiên, rõ ràng.",
        }
        return instructions.get(role, instructions["general"])

    def _intent_instruction(self, intent: str) -> str:
        instructions = {
            "so_sanh": (
                "Câu hỏi có thể là follow-up trong cuộc hội thoại. Hãy trả lời theo hướng so sánh các lựa chọn, "
                "không chỉ giới thiệu riêng một ngành. Cần nêu điểm giống nhau, điểm khác nhau, ai phù hợp hơn với từng lựa chọn, "
                "và đưa ra khuyến nghị có điều kiện dựa trên sở thích/mục tiêu của người hỏi. Nếu EVIDENCE thiếu dữ liệu cho một vế, hãy nói rõ."
            ),
            "tu_van": (
                "Hãy đóng vai trò tư vấn tuyển sinh/hướng nghiệp: xác định nhu cầu của người hỏi, đối chiếu các lựa chọn trong EVIDENCE, "
                "đưa ra lời khuyên thực tế và giải thích vì sao. Không quyết định thay người hỏi khi thiếu thông tin cá nhân."
            ),
            "history": (
                "Câu hỏi phụ thuộc lịch sử hội thoại. Hãy dùng câu hỏi đã được viết lại và EVIDENCE để trả lời đúng ngữ cảnh, "
                "không xem câu hỏi như một câu độc lập nếu nó có ý nối tiếp."
            ),
            "overview": (
                "Hãy trả lời ở mức tổng quan, chia theo các nhóm dữ liệu lớn nếu EVIDENCE có đủ: giới thiệu trường, khoa/ngành, "
                "trung tâm, tuyển sinh, quy chế đào tạo, học bổng/học vụ. Không sa vào riêng điểm tuyển sinh nếu người hỏi chỉ cần toàn cảnh."
            ),
        }
        return instructions.get(intent, "")

    def _format_history(self, history: Sequence[Dict[str, str]] | None) -> str:
        if not history:
            return ""
        lines = []
        for item in history:
            role_label = "Người dùng" if item.get("role") == "user" else "Trợ lý"
            content = re.split(r"\n\s*Nguồn\s*:", item.get("content", ""), maxsplit=1, flags=re.IGNORECASE)[0].strip()
            lines.append(f"- {role_label}: {content}")
        return "\n".join(lines)

    def _constraints_block(self, evidence_constraints: Dict[str, object] | None) -> str:
        if not evidence_constraints:
            return ""
        lines = ["RÀNG BUỘC NGỮ CẢNH CẦN KIỂM TRA:"]
        if evidence_constraints.get("major"):
            lines.append(f"- Ngành/chương trình: {evidence_constraints['major']}")
        if evidence_constraints.get("year"):
            lines.append(f"- Năm: {evidence_constraints['year']}")
        if evidence_constraints.get("year_scope"):
            lines.append(f"- Phạm vi năm: {evidence_constraints['year_scope']}")
        if evidence_constraints.get("topic"):
            lines.append(f"- Chủ đề: {evidence_constraints['topic']}")
        lines.append(
            "- Nếu EVIDENCE không khớp ngành/chương trình, năm hoặc chủ đề trên, hãy nói là chưa tìm thấy đủ bằng chứng phù hợp."
        )
        return "\n".join(lines)

    def _prompt(
        self,
        question: str,
        evidence: Sequence[SearchResult],
        role: str,
        intent: str,
        history: Sequence[Dict[str, str]] | None = None,
        evidence_constraints: Dict[str, object] | None = None,
    ) -> str:
        role_inst = self._role_instruction(role)
        intent_inst = self._intent_instruction(intent)
        intent_line = f"- {intent_inst}" if intent_inst else ""
        history_block = self._format_history(history)
        history_section = f"\nLỊCH SỬ HỘI THOẠI (context):\n{history_block}\n" if history_block else ""
        constraints_block = self._constraints_block(evidence_constraints)
        constraints_section = f"\n{constraints_block}\n" if constraints_block else ""
        return f"""
Bạn là hệ thống hỏi đáp RAG cho Trường Đại học Công nghiệp TP.HCM.

Nhiệm vụ:
- Trả lời câu hỏi bằng tiếng Việt tự nhiên, rõ ràng.
- Chỉ sử dụng thông tin trong EVIDENCE.
- Nếu có LỊCH SỬ HỘI THOẠI, hãy dùng nó để hiểu đúng ngữ cảnh câu hỏi và trả lời liền mạch.
- Mỗi ý quan trọng phải có citation dạng [1], [2] tương ứng với evidence.
- Không bịa thông tin ngoài EVIDENCE.
- Nếu EVIDENCE không đủ để trả lời chắc chắn, hãy nói rõ là chưa tìm thấy đủ bằng chứng.
- Nếu có nhiều nguồn khác nhau hoặc có dấu hiệu mâu thuẫn, hãy nêu rõ cần kiểm tra lại nguồn chính thức.
- {role_inst}
{intent_line}

Role người hỏi: {role}
Intent: {intent}
{history_section}
{constraints_section}
Câu hỏi:
{question}

EVIDENCE:
{self._evidence_block(evidence)}

Trả lời:
""".strip()

    def generate(
        self,
        question: str,
        evidence: Sequence[SearchResult],
        role: str = "general",
        intent: str = "fact",
        verify: bool = True,
        history: Sequence[Dict[str, str]] | None = None,
        evidence_constraints: Dict[str, object] | None = None,
    ) -> Dict[str, object]:
        if evidence_is_weak(evidence):
            if intent == "missing_evidence":
                answer = (
                    "Tôi chưa tìm thấy thông tin này trong dữ liệu hiện có của hệ thống. "
                    "Vui lòng kiểm tra tài liệu chính thức hoặc liên hệ trực tiếp với nhà trường."
                )
            else:
                answer = (
                    "Tôi chưa tìm thấy đủ bằng chứng trong dữ liệu hiện có để trả lời chắc chắn. "
                    "Bạn nên kiểm tra lại tài liệu chính thức hoặc bổ sung dữ liệu liên quan."
                )
            return {
                "answer": answer,
                "claim_verification": [],
                "conflicts": [],
                "confidence": "low",
                "llm_model": self.model,
                "llm_used": False,
                "llm_skip_reason": "weak_evidence",
                "abstained": True,
            }

        conflicts = detect_conflicts(question, evidence)
        try:
            response = self._client().responses.create(
                model=self.model,
                input=self._prompt(question, evidence, role, intent, history=history, evidence_constraints=evidence_constraints),
                temperature=0.1,
            )
            answer = response.output_text.strip()
        except Exception as exc:
            answer = (
                "Không gọi được OpenAI LLM để sinh câu trả lời. "
                f"Lỗi: {exc}\n\n"
                "Tạm thời trả về câu trả lời extractive từ evidence.\n\n"
                + super().generate(question, evidence, role=role, intent=intent, verify=False, history=history)["answer"]
            )
            return {
                "answer": answer,
                "claim_verification": [],
                "conflicts": conflicts,
                "confidence": "low",
                "llm_model": self.model,
                "llm_used": False,
                "llm_error": str(exc),
                "abstained": False,
            }

        if "Nguồn:" not in answer:
            answer = answer.strip() + "\n\n" + self._format_sources(evidence, role=role)

        verdicts = verify_claims(answer, evidence) if verify else []
        if verify:
            answer = self_correct_answer(answer, verdicts)

        confidence = "high"
        if any(item["status"] == "unsupported" for item in verdicts):
            confidence = "medium"
        if conflicts:
            confidence = "medium"

        return {
            "answer": answer,
            "claim_verification": verdicts,
            "conflicts": conflicts,
            "confidence": confidence,
            "llm_model": self.model,
            "llm_used": True,
            "llm_error": None,
            "abstained": False,
        }

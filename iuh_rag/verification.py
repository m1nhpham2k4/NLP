from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from .retrievers import SearchResult
from .text import normalize_text, token_overlap_score


CLAIM_SPLIT_RE = re.compile(r"(?<!TP\.)(?<=[.!?])\s+|\n+-\s+|\n+\d+\.\s+")


def extract_claims(answer: str) -> List[str]:
    body = answer.split("Nguồn:")[0]
    parts = CLAIM_SPLIT_RE.split(body)
    claims = []
    for part in parts:
        claim = part.strip(" \n-")
        if not claim:
            continue
        claim = re.sub(r"^(Dành cho [^:]+:\s*)?Theo dữ liệu hiện có:\s*", "", claim, flags=re.IGNORECASE)
        claim = re.sub(r"^Dựa trên [^:]+:\s*", "", claim, flags=re.IGNORECASE)
        if claim.lower().startswith(("toi chua", "tôi chưa", "nguồn")):
            continue
        if len(normalize_text(claim).split()) < 4:
            continue
        claims.append(claim)
    return claims


def verify_claims(
    answer: str,
    evidence: Sequence[SearchResult],
    support_threshold: float = 0.22,
) -> List[Dict[str, Any]]:
    claims = extract_claims(answer)
    verdicts: List[Dict[str, Any]] = []
    for claim in claims:
        best_score = 0.0
        best_evidence = None
        for item in evidence:
            score = token_overlap_score(claim, item.chunk.get("text", ""))
            if score > best_score:
                best_score = score
                best_evidence = item
        verdicts.append(
            {
                "claim": claim,
                "status": "supported" if best_score >= support_threshold else "unsupported",
                "score": round(best_score, 4),
                "evidence_chunk_id": best_evidence.chunk_id if best_evidence else None,
            }
        )
    return verdicts


def unsupported_claim_rate(verdicts: Sequence[Dict[str, Any]]) -> float:
    if not verdicts:
        return 0.0
    unsupported = sum(1 for item in verdicts if item["status"] != "supported")
    return unsupported / len(verdicts)


def self_correct_answer(answer: str, verdicts: Sequence[Dict[str, Any]]) -> str:
    if not verdicts:
        return answer
    unsupported = {item["claim"] for item in verdicts if item["status"] != "supported"}
    if not unsupported:
        return answer

    sources = ""
    if "Nguồn:" in answer:
        answer, sources = answer.split("Nguồn:", 1)
        sources = "Nguồn:" + sources

    kept = []
    for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
        sentence_clean = sentence.strip()
        if not sentence_clean:
            continue
        if any(claim in sentence_clean for claim in unsupported):
            continue
        kept.append(sentence_clean)

    removed_text = " Tôi đã lược bỏ các ý chưa đủ bằng chứng trong tài liệu được truy xuất."
    corrected = " ".join(kept).strip()
    if not corrected:
        corrected = "Tôi chưa tìm thấy đủ bằng chứng trong dữ liệu hiện có để trả lời chắc chắn."
    corrected = corrected + removed_text
    if sources:
        corrected = corrected + "\n\n" + sources.strip()
    return corrected


def evidence_is_weak(evidence: Sequence[SearchResult], min_top_score: float = 0.04) -> bool:
    if not evidence:
        return True
    return max(item.score for item in evidence) < min_top_score


def detect_conflicts(question: str, evidence: Sequence[SearchResult]) -> List[Dict[str, Any]]:
    q = normalize_text(question)
    conflicts: List[Dict[str, Any]] = []
    if not evidence:
        return conflicts

    if any(term in q for term in ["ma nganh", "ma tuyen sinh"]):
        code_by_source: Dict[str, set[str]] = {}
        for item in evidence:
            codes = set(re.findall(r"\b7\d{6}\b|\b[0-9]{7}\b", item.chunk.get("text", "")))
            if codes:
                code_by_source[item.chunk_id] = codes
        distinct = {tuple(sorted(values)) for values in code_by_source.values()}
        if len(distinct) > 1:
            conflicts.append({"type": "ma_nganh", "values_by_chunk": {k: sorted(v) for k, v in code_by_source.items()}})

    if any(term in q for term in ["to hop", "xet tuyen"]):
        combos_by_source: Dict[str, set[str]] = {}
        for item in evidence:
            combos = set(re.findall(r"\b[A-D]\d{2}\b", item.chunk.get("text", "")))
            if combos:
                combos_by_source[item.chunk_id] = combos
        distinct = {tuple(sorted(values)) for values in combos_by_source.values()}
        if len(distinct) > 1 and len(combos_by_source) > 1:
            conflicts.append({"type": "to_hop_xet_tuyen", "values_by_chunk": {k: sorted(v) for k, v in combos_by_source.items()}})

    return conflicts

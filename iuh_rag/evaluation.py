from __future__ import annotations

import math
from typing import Any, Callable, Dict, Iterable, List, Sequence

from .text import keyword_hit_ratio


def _relevant_ids(question: Dict[str, Any]) -> set[str]:
    return set(question.get("relevant_chunk_ids", []))


def recall_at_k(predicted: Sequence[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return 1.0 if relevant & set(predicted[:k]) else 0.0


def precision_at_k(predicted: Sequence[str], relevant: set[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(predicted[:k]) & relevant) / k


def mrr(predicted: Sequence[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(predicted, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(predicted: Sequence[str], relevant: set[str], k: int) -> float:
    dcg = 0.0
    for rank, chunk_id in enumerate(predicted[:k], start=1):
        if chunk_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    if ideal_hits == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg


def evaluate_retrieval(
    questions: Sequence[Dict[str, Any]],
    retrieve_fn: Callable[[Dict[str, Any]], Sequence[str]],
) -> Dict[str, float]:
    rows = []
    for question in questions:
        relevant = _relevant_ids(question)
        predicted = list(retrieve_fn(question))
        rows.append(
            {
                "recall@1": recall_at_k(predicted, relevant, 1),
                "recall@3": recall_at_k(predicted, relevant, 3),
                "recall@5": recall_at_k(predicted, relevant, 5),
                "precision@5": precision_at_k(predicted, relevant, 5),
                "mrr": mrr(predicted, relevant),
                "ndcg@5": ndcg_at_k(predicted, relevant, 5),
            }
        )

    if not rows:
        return {}
    metrics = rows[0].keys()
    summary = {metric: sum(row[metric] for row in rows) / len(rows) for metric in metrics}
    summary["n_questions"] = len(rows)
    summary["hits@1"] = int(sum(row["recall@1"] for row in rows))
    summary["hits@3"] = int(sum(row["recall@3"] for row in rows))
    summary["hits@5"] = int(sum(row["recall@5"] for row in rows))
    return summary


def evaluate_answers(
    questions: Sequence[Dict[str, Any]],
    answer_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, float]:
    rows = []
    for question in questions:
        result = answer_fn(question)
        answer = str(result.get("answer", ""))
        cited_ids = {item.get("chunk_id") for item in result.get("evidence", [])}
        relevant = _relevant_ids(question)
        verdicts = result.get("claim_verification", [])
        unsupported = sum(1 for item in verdicts if item.get("status") != "supported")
        abstained = bool(result.get("abstained", False))
        # Abstention is correct if the question is missing_evidence type
        expected_abstain = question.get("type") == "missing_evidence" or question.get("intent") == "missing_evidence"
        rows.append(
            {
                "answer_correctness": keyword_hit_ratio(answer, question.get("expected_keywords", [])),
                "citation_accuracy": 1.0 if cited_ids & relevant else 0.0,
                "unsupported_claim_rate": unsupported / len(verdicts) if verdicts else 0.0,
                "hallucination_rate": 1.0 if unsupported else 0.0,
                "abstained": 1.0 if abstained else 0.0,
                "abstention_correct": 1.0 if (abstained and expected_abstain) or (not abstained and not expected_abstain) else 0.0,
            }
        )
    if not rows:
        return {}
    metrics = rows[0].keys()
    summary = {metric: sum(row[metric] for row in rows) / len(rows) for metric in metrics}
    summary["n_questions"] = len(rows)
    return summary


def evaluate_abstention(
    questions: Sequence[Dict[str, Any]],
    answer_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, float]:
    """Evaluate abstention accuracy on questions that are missing_evidence type.

    Returns:
        abstention_accuracy: fraction of missing_evidence questions where system correctly abstained.
        false_abstention_rate: fraction of non-missing questions where system incorrectly abstained.
    """
    abstention_correct = []
    false_abstentions = []

    for question in questions:
        result = answer_fn(question)
        abstained = bool(result.get("abstained", False))
        is_missing = question.get("type") == "missing_evidence" or question.get("intent") == "missing_evidence"

        if is_missing:
            abstention_correct.append(1.0 if abstained else 0.0)
        else:
            false_abstentions.append(1.0 if abstained else 0.0)

    return {
        "abstention_accuracy": sum(abstention_correct) / max(1, len(abstention_correct)),
        "false_abstention_rate": sum(false_abstentions) / max(1, len(false_abstentions)),
        "n_missing_evidence": len(abstention_correct),
        "n_non_missing": len(false_abstentions),
    }


def evaluate_conflict_detection(
    questions: Sequence[Dict[str, Any]],
    answer_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, float]:
    """Evaluate conflict detection accuracy on conflict_check questions.

    A conflict is detected correctly if:
    - question type == 'conflict_check' AND result['conflicts'] is non-empty
    - OR question type != 'conflict_check' AND result['conflicts'] is empty
    """
    correct = []
    for question in questions:
        result = answer_fn(question)
        conflicts = result.get("conflicts", [])
        detected = len(conflicts) > 0
        is_conflict_q = question.get("type") == "conflict_check" or question.get("intent") == "conflict_check"
        correct.append(1.0 if detected == is_conflict_q else 0.0)

    return {
        "conflict_detection_accuracy": sum(correct) / max(1, len(correct)),
        "n_questions": len(correct),
    }


def evaluate_self_correction(
    questions: Sequence[Dict[str, Any]],
    answer_fn: Callable[[Dict[str, Any]], Dict[str, Any]],
) -> Dict[str, float]:
    """Evaluate self-correction success rate.

    Self-correction success = answer was modified (unsupported claims removed)
    and final answer still has content and no unsupported claims.
    """
    corrected = []
    for question in questions:
        result = answer_fn(question)
        verdicts = result.get("claim_verification", [])
        if not verdicts:
            continue
        unsupported = [v for v in verdicts if v.get("status") != "supported"]
        if not unsupported:
            corrected.append(1.0)  # Nothing to correct — perfect
            continue
        # Check if answer contains correction note
        answer = str(result.get("answer", ""))
        has_correction = "lược bỏ" in answer or "chưa đủ bằng chứng" in answer
        corrected.append(1.0 if has_correction else 0.0)

    return {
        "self_correction_success_rate": sum(corrected) / max(1, len(corrected)),
        "n_evaluated": len(corrected),
    }


def by_type_summary(
    questions: Sequence[Dict[str, Any]],
    retrieve_fn: Callable[[Dict[str, Any]], Sequence[str]],
) -> Dict[str, Dict[str, float]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for question in questions:
        groups.setdefault(question.get("type", "unknown"), []).append(question)
    return {group: evaluate_retrieval(items, retrieve_fn) for group, items in groups.items()}


def by_difficulty_summary(
    questions: Sequence[Dict[str, Any]],
    retrieve_fn: Callable[[Dict[str, Any]], Sequence[str]],
) -> Dict[str, Dict[str, float]]:
    """Evaluate retrieval grouped by difficulty: easy, medium, hard."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for question in questions:
        diff = question.get("difficulty", "unknown")
        groups.setdefault(diff, []).append(question)
    return {group: evaluate_retrieval(items, retrieve_fn) for group, items in groups.items()}

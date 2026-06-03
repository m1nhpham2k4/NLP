from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, TypedDict

try:
    from langchain_core._api import LangChainPendingDeprecationWarning
except Exception:  # pragma: no cover - depends on installed langchain-core version
    LangChainPendingDeprecationWarning = Warning  # type: ignore[assignment]

warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
    message=r"The default value of `allowed_objects`.*",
)

from langgraph.graph import END, START, StateGraph

from .agents import QueryPlan, QueryPlanner
from .answer import AnswerGenerator
from .retrievers import SearchResult


class AgentState(TypedDict, total=False):
    question: str
    history: Optional[List[Dict[str, str]]]
    role: Optional[str]
    top_k: int
    plan: QueryPlan
    evidence: List[SearchResult]
    answer_mode: str
    answer_data: Dict[str, Any]
    direct_payload: Optional[Dict[str, Any]]
    final_payload: Dict[str, Any]
    force_hyde: bool
    force_metadata: bool
    force_graph: bool
    force_verification: Optional[bool]


DirectAnswerFn = Callable[[str, QueryPlan], Optional[Dict[str, Any]]]


def create_rag_workflow(
    planner: QueryPlanner,
    retriever_fn: Callable[..., List[SearchResult]],
    answer_gen: AnswerGenerator,
    direct_answer_fn: DirectAnswerFn | None = None,
):
    graph = StateGraph(AgentState)

    def apply_runtime_flags(plan: QueryPlan, state: AgentState) -> QueryPlan:
        if state.get("force_hyde"):
            plan.use_hyde = True
        if state.get("force_metadata"):
            plan.use_metadata = True
        if state.get("force_graph"):
            plan.use_graph = True
        if state.get("force_verification") is not None:
            plan.needs_verification = bool(state.get("force_verification"))
        return plan

    def plan_query(state: AgentState) -> Dict[str, Any]:
        plan = planner.plan(
            state["question"],
            history=state.get("history"),
            role=state.get("role"),
        )
        plan = apply_runtime_flags(plan, state)
        direct_payload = direct_answer_fn(state["question"], plan) if direct_answer_fn else None
        return {"plan": plan, "direct_payload": direct_payload}

    def route_after_plan(state: AgentState) -> str:
        return "finalize_direct" if state.get("direct_payload") else "retrieve"

    def retrieve(state: AgentState) -> Dict[str, Any]:
        evidence = retriever_fn(
            state["plan"],
            top_k=state.get("top_k", 5),
            force_hyde=bool(state.get("force_hyde", False)),
            force_metadata=bool(state.get("force_metadata", False)),
            force_graph=bool(state.get("force_graph", False)),
        )
        return {"evidence": evidence}

    def generate_answer(state: AgentState) -> Dict[str, Any]:
        plan = state["plan"]
        generated = answer_gen.generate(
            plan.rewritten_question,
            state.get("evidence", []),
            role=plan.role,
            intent=plan.intent,
            verify=plan.needs_verification,
            history=state.get("history"),
            evidence_constraints=plan.evidence_constraints,
        )
        return {"answer_data": generated}

    def finalize_direct(state: AgentState) -> Dict[str, Any]:
        return {"final_payload": state["direct_payload"] or {}}

    def finalize_rag(state: AgentState) -> Dict[str, Any]:
        plan = state["plan"]
        generated = state.get("answer_data", {})
        evidence = state.get("evidence", [])
        payload = {
            "question": state["question"],
            "rewritten_question": plan.rewritten_question,
            "role": plan.role,
            "intent": plan.intent,
            "filters": plan.filters,
            "sub_questions": plan.sub_questions,
            "hypothesis_queries": plan.hypothesis_queries,
            "current_entities": plan.current_entities,
            "history_entities": plan.history_entities,
            "comparison_entities": plan.comparison_entities,
            "is_followup": plan.is_followup,
            "topic": plan.topic,
            "year": plan.year,
            "entities": plan.entities,
            "missing_slots": plan.missing_slots,
            "retrieval_query": plan.retrieval_query,
            "evidence_constraints": plan.evidence_constraints,
            "needs_clarification": plan.needs_clarification,
            "clarification_question": plan.clarification_question,
            "conversation_context": plan.conversation_context,
            "followup_reason": plan.followup_reason,
            "is_topic_shift": plan.is_topic_shift,
            "reasoning_summary": plan.reasoning_summary,
            "scope": plan.scope,
            "risk_flags": plan.risk_flags,
            "retrieval_strategy": plan.retrieval_strategy,
            "answer": generated.get("answer", ""),
            "answer_mode": state.get("answer_mode", "extractive"),
            "confidence": generated.get("confidence", "low"),
            "llm_model": generated.get("llm_model"),
            "llm_used": generated.get("llm_used"),
            "llm_error": generated.get("llm_error"),
            "llm_skip_reason": generated.get("llm_skip_reason"),
            "evidence": [item.to_dict() for item in evidence],
            "claim_verification": generated.get("claim_verification", []),
            "conflicts": generated.get("conflicts", []),
            "abstained": generated.get("abstained", False),
        }
        return {"final_payload": payload}

    graph.add_node("plan_query", plan_query)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)
    graph.add_node("finalize_direct", finalize_direct)
    graph.add_node("finalize_rag", finalize_rag)

    graph.add_edge(START, "plan_query")
    graph.add_conditional_edges(
        "plan_query",
        route_after_plan,
        {
            "retrieve": "retrieve",
            "finalize_direct": "finalize_direct",
        },
    )
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", "finalize_rag")
    graph.add_edge("finalize_direct", END)
    graph.add_edge("finalize_rag", END)

    return graph.compile()

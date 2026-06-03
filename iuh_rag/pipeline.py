from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .agents import HyDEGenerator, QueryPlan, QueryPlanner
from .answer import AnswerGenerator, OpenAIAnswerGenerator
from .config import (
    CHUNKS_PATH,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_QDRANT_COLLECTION_PREFIX,
    DEFAULT_QDRANT_MODE,
    DEFAULT_QDRANT_PATH,
    DEFAULT_QDRANT_URL,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_STORE,
    EMBEDDING_MODELS,
)
from .data import enrich_chunks, load_chunks
from .graph import KnowledgeGraph
from .workflow import create_rag_workflow
from .reranker import CrossEncoderReranker
from .retrievers import (
    BM25Retriever,
    DenseEmbeddingRetriever,
    MultiRetriever,
    QdrantDenseEmbeddingRetriever,
    SearchResult,
    chunk_matches_filters,
    TfidfRetriever,
    lexical_rerank,
    rrf_fuse,
)
from .text import normalize_text


GREETING_ANSWER = (
    "Xin chào! Tôi là chatbot hỗ trợ tra cứu thông tin về Trường Đại học Công nghiệp TP.HCM. "
    "Bạn có thể hỏi tôi về tuyển sinh, ngành đào tạo, khoa/trung tâm, quy chế đào tạo, học bổng, "
    "điểm chuẩn, chỉ tiêu hoặc nhờ tôi tư vấn chọn ngành. Ví dụ: "
    "\"Ngành Công nghệ thông tin xét tuyển tổ hợp nào?\" hoặc \"Khoa Công nghệ thông tin đào tạo những ngành nào?\""
)


def _unique_queries(items: Sequence[str | None]) -> List[str]:
    seen = set()
    output: List[str] = []
    for item in items:
        cleaned = re.sub(r"\s+", " ", str(item or "")).strip()
        key = normalize_text(cleaned)
        if cleaned and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


class AgenticGraphRAG:
    def __init__(
        self,
        chunks: Sequence[Dict[str, Any]],
        enable_dense: bool = False,
        dense_model_keys: Sequence[str] | None = None,
        answer_mode: str = "extractive",
        openai_model: str = DEFAULT_OPENAI_MODEL,
        reranker_model: Optional[str] = None,
        embedding_device: str = DEFAULT_EMBEDDING_DEVICE,
        vector_store: str = DEFAULT_VECTOR_STORE,
        qdrant_mode: str = DEFAULT_QDRANT_MODE,
        qdrant_path: str = DEFAULT_QDRANT_PATH,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        qdrant_collection_prefix: str = DEFAULT_QDRANT_COLLECTION_PREFIX,
    ) -> None:
        self.chunks = enrich_chunks(list(chunks))
        self.vector_store = vector_store
        self.qdrant_mode = qdrant_mode
        self.qdrant_path = qdrant_path
        self.qdrant_url = qdrant_url
        self.qdrant_collection_prefix = qdrant_collection_prefix
        self.known_terms = self._collect_known_terms()
        self.planner = QueryPlanner(known_terms=self.known_terms)
        self.hyde = HyDEGenerator()
        self.answer_mode = answer_mode
        if answer_mode == "openai":
            self.answer_generator = OpenAIAnswerGenerator(model=openai_model)
        else:
            self.answer_generator = AnswerGenerator()
        self.graph = KnowledgeGraph(self.chunks)

        self.bm25 = BM25Retriever(self.chunks)
        self.tfidf = TfidfRetriever(self.chunks)
        self.dense_retrievers: List[DenseEmbeddingRetriever] = []
        self.dense_by_key: Dict[str, DenseEmbeddingRetriever] = {}
        if enable_dense:
            if vector_store not in {"joblib", "qdrant"}:
                raise ValueError("vector_store must be 'joblib' or 'qdrant'")
            keys = list(dense_model_keys or EMBEDDING_MODELS.keys())
            for key in keys:
                model_name = EMBEDDING_MODELS.get(key, key)
                if vector_store == "qdrant":
                    retriever = QdrantDenseEmbeddingRetriever(
                        self.chunks,
                        model_name=model_name,
                        name=f"dense:{key}",
                        device=embedding_device,
                        qdrant_mode=qdrant_mode,
                        qdrant_path=qdrant_path,
                        qdrant_url=qdrant_url,
                        collection_prefix=qdrant_collection_prefix,
                    )
                else:
                    retriever = DenseEmbeddingRetriever(
                        self.chunks,
                        model_name=model_name,
                        name=f"dense:{key}",
                        device=embedding_device,
                    )
                self.dense_retrievers.append(retriever)
                self.dense_by_key[key] = retriever

        self.multi_embedding = MultiRetriever(self.dense_retrievers)
        self.hybrid = MultiRetriever([self.bm25, self.tfidf, *self.dense_retrievers])

        # Optional cross-encoder reranker
        self.reranker: Optional[CrossEncoderReranker] = None
        if reranker_model:
            self.reranker = CrossEncoderReranker(model_name=reranker_model, device=embedding_device)

        # Initialize LangGraph workflow
        self.workflow = create_rag_workflow(
            planner=self.planner,
            retriever_fn=self.retrieve_with_plan,
            answer_gen=self.answer_generator,
            direct_answer_fn=lambda question, plan: self._direct_answer_for_plan(
                question,
                plan,
                include_plan_fields=True,
            ),
        )

    @classmethod
    def from_path(
        cls,
        chunks_path: Path = CHUNKS_PATH,
        enable_dense: bool = False,
        dense_model_keys: Sequence[str] | None = None,
        answer_mode: str = "extractive",
        openai_model: str = DEFAULT_OPENAI_MODEL,
        reranker_model: Optional[str] = None,
        embedding_device: str = DEFAULT_EMBEDDING_DEVICE,
        vector_store: str = DEFAULT_VECTOR_STORE,
        qdrant_mode: str = DEFAULT_QDRANT_MODE,
        qdrant_path: str = DEFAULT_QDRANT_PATH,
        qdrant_url: str = DEFAULT_QDRANT_URL,
        qdrant_collection_prefix: str = DEFAULT_QDRANT_COLLECTION_PREFIX,
    ) -> "AgenticGraphRAG":
        return cls(
            load_chunks(chunks_path),
            enable_dense=enable_dense,
            dense_model_keys=dense_model_keys,
            answer_mode=answer_mode,
            openai_model=openai_model,
            reranker_model=reranker_model,
            embedding_device=embedding_device,
            vector_store=vector_store,
            qdrant_mode=qdrant_mode,
            qdrant_path=qdrant_path,
            qdrant_url=qdrant_url,
            qdrant_collection_prefix=qdrant_collection_prefix,
        )

    def _collect_known_terms(self) -> List[str]:
        terms = set()
        for chunk in self.chunks:
            metadata = chunk.get("metadata", {})
            for key in ("department", "center", "program"):
                value = metadata.get(key) or chunk.get(key)
                if value:
                    terms.add(str(value).replace(".txt", ""))
            for match in re.findall(
                r"(?:\*\*\*\s*)?(?:Ngành|Thạc sĩ|Tiến sĩ)\s+([A-ZĐ][\wÀ-ỹ]+(?:\s+[A-ZĐa-zà-ỹ0-9]+){1,6})",
                chunk.get("text", "")[:500],
            ):
                cleaned = re.split(r"\s+(?:Mục tiêu|Giới thiệu|Chuẩn đầu ra|Tuyển sinh)\b", match.strip())[0].strip()
                if cleaned:
                    terms.add(cleaned)
        return sorted(terms, key=len, reverse=True)

    def build_dense_indexes(self, force: bool = False) -> Dict[str, str]:
        statuses = {}
        for retriever in self.dense_retrievers:
            ok = retriever.build_index(force=force)
            statuses[retriever.name] = "ok" if ok else f"failed: {retriever.error}"
        return statuses

    def dense_statuses(self) -> Dict[str, Dict[str, Any]]:
        return {key: retriever.status() for key, retriever in self.dense_by_key.items()}

    def reranker_status(self) -> Dict[str, Any] | None:
        return self.reranker.status() if self.reranker is not None else None

    def _direct_answer_payload(
        self,
        question: str,
        answer: str,
        plan: QueryPlan,
        *,
        include_plan_fields: bool = False,
        skip_reason: str = "direct_conversation",
        abstained: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "question": question,
            "answer": answer,
            "answer_mode": self.answer_mode,
            "llm_model": None,
            "llm_used": False,
            "llm_error": None,
            "llm_skip_reason": skip_reason,
            "evidence": [],
            "claim_verification": [],
            "conflicts": [],
            "confidence": "high",
            "abstained": abstained,
        }
        if include_plan_fields:
            payload.update(
                {
                    "rewritten_question": plan.rewritten_question,
                    "role": plan.role,
                    "intent": plan.intent,
                    "filters": plan.filters,
                    "sub_questions": plan.sub_questions,
                    "hypothesis_queries": plan.hypothesis_queries,
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
                }
            )
        return payload

    def _direct_answer_for_plan(self, question: str, plan: QueryPlan, *, include_plan_fields: bool = False) -> Dict[str, Any] | None:
        if plan.needs_clarification:
            return self._direct_answer_payload(
                question,
                plan.clarification_question or "Bạn cần bổ sung thêm ngành/chương trình muốn hỏi.",
                plan,
                include_plan_fields=True,
                skip_reason="clarification_needed",
                abstained=True,
            )
        if plan.intent == "greeting":
            return self._direct_answer_payload(
                question,
                GREETING_ANSWER,
                plan,
                include_plan_fields=include_plan_fields,
            )
        return None
    def plan(self, question: str, history: Sequence[Dict[str, str]] | None = None, role: str | None = None) -> QueryPlan:
        return self.planner.plan(question, history=history, role=role)

    def retrieve_with_plan(
        self,
        plan: QueryPlan,
        top_k: int = DEFAULT_TOP_K,
        base_retriever: MultiRetriever | None = None,
        force_hyde: bool = False,
        force_graph: bool = False,
        force_metadata: bool = False,
    ) -> List[SearchResult]:
        candidate_lists: List[List[SearchResult]] = []
        retriever = base_retriever or self.hybrid
        metadata_filters = plan.filters if plan.use_metadata or force_metadata else None
        latest_year = self._resolve_latest_year_filter(plan, plan.filters)
        retrieval_query = plan.retrieval_query or plan.rewritten_question or plan.original_question
        if latest_year and latest_year not in retrieval_query:
            retrieval_query = f"{retrieval_query} năm {latest_year}"
        search_queries = _unique_queries([plan.original_question, retrieval_query, *plan.sub_questions])

        def append_search(query: str, query_top_k: int) -> None:
            # Soft Metadata Filtering: retrieve everything unfiltered to prevent omission
            candidate_lists.append(retriever.search(query, top_k=query_top_k, filters=None))

        for sub_question in search_queries:
            append_search(sub_question, top_k * 2)

        for entity in plan.comparison_entities:
            append_search(f"Thông tin chương trình đào tạo, chuẩn đầu ra và tuyển sinh của ngành {entity}", top_k * 2)

        if plan.use_hyde or force_hyde:
            hypothesis_queries = plan.hypothesis_queries or self.hyde.generate_queries(
                retrieval_query,
                intent=plan.intent,
                entities=plan.comparison_entities or plan.current_entities,
                filters=plan.filters,
            )
            plan.hypothesis_queries = hypothesis_queries
            hyde_lists = []
            for hypothesis in hypothesis_queries:
                hyde_lists.append(retriever.search(hypothesis, top_k=top_k, filters=None))
            if hyde_lists:
                fused_hyde = rrf_fuse(hyde_lists, top_k=top_k)
                candidate_lists.append(fused_hyde)


        if plan.intent == "overview":
            overview_type_queries = {
                "tong_quan_ve_truong": "Tổng quan về trường IUH lịch sử sứ mạng tầm nhìn mục tiêu chiến lược",
                "cac_khoa": "Tổng quan các khoa ngành chương trình đào tạo chuẩn đầu ra IUH",
                "cac_trung_tam": "Tổng quan các trung tâm đơn vị hỗ trợ cơ sở vật chất thư viện IUH",
                "dao_tao": "Tổng quan quy chế đào tạo tín chỉ học vụ tốt nghiệp học bổng IUH",
                "tuyen_sinh": "Tổng quan tuyển sinh phương thức xét tuyển mã ngành chỉ tiêu điểm chuẩn IUH",
            }
            for chunk_type, overview_query in overview_type_queries.items():
                candidate_lists.append(
                    retriever.search(
                        overview_query,
                        top_k=max(1, top_k),
                        filters={"type": chunk_type},
                    )
                )

        if plan.use_graph or force_graph:
            graph_results = self.graph.search(retrieval_query, top_k=top_k * 2)
            candidate_lists.append(graph_results)

        fusion_multiplier = 8 if plan.intent == "overview" else (6 if metadata_filters or plan.use_hyde or force_hyde or plan.use_graph or force_graph else 3)
        fusion_k = top_k * fusion_multiplier
        fused = rrf_fuse(candidate_lists, top_k=fusion_k)

        # Soft Metadata Boosting
        if metadata_filters:
            for item in fused:
                if chunk_matches_filters(item.chunk, metadata_filters):
                    item.score *= 1.5
            fused = sorted(fused, key=lambda item: item.score, reverse=True)
            for rank, item in enumerate(fused, start=1):
                item.rank = rank

        if self.reranker is not None:
            ranked = self.reranker.rerank(retrieval_query, fused, top_k=top_k)
        else:
            ranked = lexical_rerank(retrieval_query, fused, top_k=top_k)
        ranked = self._boost_evidence_constraints(plan, ranked)
        ranked = self._ensure_overview_coverage(plan, ranked, fused, top_k)
        return self._ensure_comparison_entity_coverage(plan, ranked, fused, top_k)

    @staticmethod
    def _chunk_years(chunk: Dict[str, Any]) -> List[int]:
        metadata = chunk.get("metadata", {})
        values: List[Any] = []
        if isinstance(metadata, dict):
            values.extend(metadata.get("years") or [])
            if metadata.get("year"):
                values.append(metadata.get("year"))
        if chunk.get("year"):
            values.append(chunk.get("year"))
        years = set()
        for value in values:
            for match in re.findall(r"\b20[2-3]\d\b", str(value)):
                years.add(int(match))
        return sorted(years)

    @staticmethod
    def _chunk_constraint_text(chunk: Dict[str, Any]) -> str:
        metadata = chunk.get("metadata", {})
        values = [
            chunk.get("text", ""),
            chunk.get("file_name", ""),
            chunk.get("relative_path", ""),
            chunk.get("type", ""),
        ]
        if isinstance(metadata, dict):
            values.extend(str(value) for value in metadata.values() if value is not None)
        return " ".join(str(value) for value in values if value is not None)

    @staticmethod
    def _chunk_mentions_constraint_entity(chunk: Dict[str, Any], entity: str | None) -> bool:
        if not entity:
            return True
        haystack_norm = normalize_text(AgenticGraphRAG._chunk_constraint_text(chunk))
        entity_norm = normalize_text(str(entity))
        canonical_entity = re.sub(r"^(nganh|chuong trinh|chuyen nganh)\s+", "", entity_norm)
        return entity_norm in haystack_norm or bool(canonical_entity and canonical_entity in haystack_norm)

    @staticmethod
    def _chunk_mentions_constraint_topic(chunk: Dict[str, Any], topic: str | None) -> bool:
        if not topic:
            return True
        haystack_norm = normalize_text(AgenticGraphRAG._chunk_constraint_text(chunk))
        topic_terms = {
            "diem_chuan": ("diem chuan", "diem trung tuyen", "diem tuyen sinh"),
            "to_hop_mon": ("to hop", "mon xet tuyen"),
            "dieu_kien_xet_tuyen": ("dieu kien", "xet tuyen", "phuong thuc"),
            "chuong_trinh_hoc": ("chuong trinh dao tao", "mon hoc", "chuan dau ra"),
            "hoc_phi": ("hoc phi", "dong tien"),
            "hoc_bong": ("hoc bong", "mien giam", "chinh sach ho tro"),
            "viec_lam": ("viec lam", "nghe nghiep"),
            "tuyen_sinh": ("tuyen sinh", "xet tuyen", "chi tieu", "ma nganh"),
        }.get(str(topic), ())
        return bool(topic_terms and any(term in haystack_norm for term in topic_terms))

    def _resolve_latest_year_filter(self, plan: QueryPlan, filters: Dict[str, Any] | None) -> str | None:
        constraints = plan.evidence_constraints or {}
        if constraints.get("year_scope") != "latest" or constraints.get("year"):
            return None
        if filters is None:
            return None

        base_filters = {key: value for key, value in filters.items() if key != "year"}
        entity = constraints.get("major") or constraints.get("entity")
        topic = constraints.get("topic")

        def matching_years(*, require_entity: bool, require_topic: bool) -> List[int]:
            years: List[int] = []
            for chunk in self.chunks:
                if not chunk_matches_filters(chunk, base_filters):
                    continue
                if require_entity and not self._chunk_mentions_constraint_entity(chunk, str(entity) if entity else None):
                    continue
                if require_topic and not self._chunk_mentions_constraint_topic(chunk, str(topic) if topic else None):
                    continue
                years.extend(self._chunk_years(chunk))
            return years

        years = (
            matching_years(require_entity=bool(entity), require_topic=bool(topic))
            or matching_years(require_entity=bool(entity), require_topic=False)
            or matching_years(require_entity=False, require_topic=bool(topic))
            or matching_years(require_entity=False, require_topic=False)
        )
        if not years:
            return None

        latest_year = str(max(years))
        filters["year"] = latest_year
        plan.filters = {**plan.filters, "year": latest_year}
        plan.year = latest_year
        plan.evidence_constraints = {**constraints, "year": latest_year}
        self._materialize_latest_year_in_plan(plan, latest_year)
        return latest_year

    @staticmethod
    def _materialize_latest_year_in_plan(plan: QueryPlan, latest_year: str) -> None:
        latest_pattern = re.compile(r"\bnăm\s+(?:gần nhất|mới nhất)\b|\bmới nhất\b|\bgần nhất\b", re.IGNORECASE)

        def replace_latest(text: str | None) -> str | None:
            if not text:
                return text
            return latest_pattern.sub(f"năm {latest_year}", text)

        plan.rewritten_question = replace_latest(plan.rewritten_question) or plan.rewritten_question
        plan.retrieval_query = replace_latest(plan.retrieval_query) or plan.retrieval_query
        if plan.retrieval_query and latest_year not in plan.retrieval_query:
            plan.retrieval_query = f"{plan.retrieval_query} năm {latest_year}"
        plan.sub_questions = [replace_latest(item) or item for item in plan.sub_questions]
        plan.hypothesis_queries = [replace_latest(item) or item for item in plan.hypothesis_queries]
        if isinstance(plan.scope, dict):
            plan.scope = {**plan.scope, "year": latest_year}
        if isinstance(plan.retrieval_strategy, dict):
            plan.retrieval_strategy = {
                **plan.retrieval_strategy,
                "resolved_latest_year": latest_year,
            }

    @staticmethod
    def _constraint_hit_score(plan: QueryPlan, result: SearchResult) -> float:
        constraints = plan.evidence_constraints or {}
        if not constraints:
            return 0.0
        chunk = result.chunk
        metadata = chunk.get("metadata", {})
        haystack = " ".join(
            str(value)
            for value in (
                chunk.get("text", ""),
                chunk.get("file_name", ""),
                chunk.get("relative_path", ""),
                chunk.get("type", ""),
                *([*metadata.values()] if isinstance(metadata, dict) else []),
            )
            if value is not None
        )
        haystack_norm = normalize_text(haystack)
        score = 0.0
        entity = constraints.get("major") or constraints.get("entity")
        if entity:
            entity_norm = normalize_text(str(entity))
            canonical_entity = re.sub(r"^(nganh|chuong trinh|chuyen nganh)\s+", "", entity_norm)
            if entity_norm in haystack_norm or (canonical_entity and canonical_entity in haystack_norm):
                score += 1.0
        year = constraints.get("year")
        if year and str(year) in haystack:
            score += 0.6
        topic = constraints.get("topic")
        topic_terms = {
            "diem_chuan": ("diem chuan", "diem trung tuyen", "diem tuyen sinh"),
            "to_hop_mon": ("to hop", "mon xet tuyen"),
            "dieu_kien_xet_tuyen": ("dieu kien", "xet tuyen", "phuong thuc"),
            "chuong_trinh_hoc": ("chuong trinh dao tao", "mon hoc", "chuan dau ra"),
            "hoc_phi": ("hoc phi", "dong tien"),
            "hoc_bong": ("hoc bong", "mien giam", "chinh sach ho tro"),
            "viec_lam": ("viec lam", "nghe nghiep"),
            "tuyen_sinh": ("tuyen sinh", "xet tuyen", "chi tieu", "ma nganh"),
        }.get(str(topic), ())
        if topic_terms and any(term in haystack_norm for term in topic_terms):
            score += 0.4
        return score

    def _boost_evidence_constraints(self, plan: QueryPlan, ranked: List[SearchResult]) -> List[SearchResult]:
        if not plan.evidence_constraints or len(ranked) <= 1:
            return ranked
        indexed = list(enumerate(ranked))
        indexed.sort(key=lambda row: (self._constraint_hit_score(plan, row[1]), -row[0]), reverse=True)
        return [item for _, item in indexed]

    @staticmethod
    def _result_mentions_entity(result: SearchResult, entity: str) -> bool:
        entity_norm = normalize_text(entity)
        if not entity_norm:
            return False
        chunk = result.chunk
        haystack = " ".join(
            str(chunk.get(key, ""))
            for key in ("text", "file_name", "relative_path", "type")
        )
        metadata = chunk.get("metadata", {})
        if isinstance(metadata, dict):
            haystack += " " + " ".join(str(value) for value in metadata.values() if value is not None)
        return entity_norm in normalize_text(haystack)

    def _ensure_comparison_entity_coverage(
        self,
        plan: QueryPlan,
        ranked: List[SearchResult],
        candidates: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        if len(plan.comparison_entities) < 2:
            return ranked
        entity_ranked = [
            item
            for item in ranked
            if any(self._result_mentions_entity(item, entity) for entity in plan.comparison_entities)
        ]
        selected = entity_ranked or list(ranked)
        selected_ids = {item.chunk_id for item in selected}
        coverage: List[SearchResult] = []
        for entity in plan.comparison_entities:
            if any(self._result_mentions_entity(item, entity) for item in selected):
                continue
            for candidate in candidates:
                if candidate.chunk_id in selected_ids:
                    continue
                if self._result_mentions_entity(candidate, entity):
                    coverage.append(candidate)
                    selected_ids.add(candidate.chunk_id)
                    break
        if not coverage:
            return selected[:top_k]
        output = coverage + [item for item in selected if item.chunk_id not in {extra.chunk_id for extra in coverage}]
        return output[:top_k]

    @staticmethod
    def _ensure_overview_coverage(
        plan: QueryPlan,
        ranked: List[SearchResult],
        candidates: List[SearchResult],
        top_k: int,
    ) -> List[SearchResult]:
        if plan.intent != "overview" or top_k <= 1:
            return ranked

        priority_types = ["tong_quan_ve_truong", "cac_khoa", "cac_trung_tam", "dao_tao", "tuyen_sinh"]
        selected: List[SearchResult] = []
        selected_ids = set()

        for chunk_type in priority_types:
            for candidate in candidates:
                if candidate.chunk_id in selected_ids:
                    continue
                if candidate.chunk.get("type") != chunk_type:
                    continue
                selected.append(candidate)
                selected_ids.add(candidate.chunk_id)
                break
            if len(selected) >= top_k:
                return selected[:top_k]

        for item in ranked:
            if item.chunk_id in selected_ids:
                continue
            selected.append(item)
            selected_ids.add(item.chunk_id)
            if len(selected) >= top_k:
                break
        return selected[:top_k] or ranked

    def retrieve(self, question: str, top_k: int = DEFAULT_TOP_K, **kwargs: Any) -> List[SearchResult]:
        plan = self.plan(question, history=kwargs.get("history"), role=kwargs.get("role"))
        return self.retrieve_with_plan(plan, top_k=top_k)

    def answer(
        self,
        question: str,
        history: Sequence[Dict[str, str]] | None = None,
        role: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        force_hyde: bool = False,
        force_metadata: bool = False,
        force_graph: bool = False,
        force_verification: bool | None = None,
    ) -> Dict[str, Any]:
        initial_state = {
            "question": question,
            "history": list(history) if history else None,
            "role": role,
            "top_k": top_k,
            "answer_mode": self.answer_mode,
            "evidence": [],
            "force_hyde": force_hyde,
            "force_metadata": force_metadata,
            "force_graph": force_graph,
            "force_verification": force_verification,
        }
        result = self.workflow.invoke(initial_state)
        return result["final_payload"]

    def retrieve_for_system(self, system_name: str, question: Dict[str, Any], top_k: int = DEFAULT_TOP_K) -> List[str]:
        if system_name != "S8_FULL":
            raise ValueError(f"System {system_name} is disabled. Only S8_FULL is supported.")
        q = question["question"]
        history = question.get("history", [])
        plan = self.plan(q, history=history)
        return [item.chunk_id for item in self.retrieve_with_plan(plan, top_k=top_k, base_retriever=self.hybrid)]

    def answer_for_system(self, system_name: str, question: Dict[str, Any], top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
        if system_name != "S8_FULL":
            raise ValueError(f"System {system_name} is disabled. Only S8_FULL is supported.")
        return self.answer(
            question["question"],
            history=question.get("history"),
            role=question.get("role"),
            top_k=top_k,
            force_hyde=False,
            force_metadata=False,
            force_graph=False,
            force_verification=True,
        )

    def system_descriptions(self) -> Dict[str, str]:
        return {
            "S8_FULL": "Proposed full system: BM25 + TF-IDF + 4 embeddings + RRF + agent planner + HyDE + history rewrite + metadata + graph + claim verification + self-correction.",
        }

    def system_names(self) -> List[str]:
        return [
            "S8_FULL",
        ]

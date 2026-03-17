from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.concurrency import run_in_threadpool

from ..runtime import answer_question_cached, clear_runtime_cache, index_exists, load_settings
from ..schemas import (
    BuildIndexRequest,
    BuildIndexResponse,
    ChatRequest,
    ChatResponse,
    MessageResponse,
    SearchResultResponse,
    StatusResponse,
)
from nlp_rag.ingest import build_index
from nlp_rag.service import ChatTurn, citations_markdown


router = APIRouter(tags=["rag"])


def _history_from_request(payload: ChatRequest) -> list[ChatTurn]:
    return [ChatTurn(role=item.role, content=item.content) for item in payload.history]


def _serialize_result(item) -> SearchResultResponse:
    return SearchResultResponse(
        score=item.score,
        source=item.source,
        text=item.text,
        metadata=item.metadata,
    )


@router.get("/health", response_model=MessageResponse)
async def healthcheck() -> MessageResponse:
    return MessageResponse(message="ok")


@router.get("/status", response_model=StatusResponse)
async def get_status() -> StatusResponse:
    settings = load_settings()
    return StatusResponse(
        app_name="IUH Assistant API",
        index_ready=index_exists(settings),
        gemini_configured=bool(settings.google_api_key),
        query_rewrite_enabled=settings.enable_query_rewrite,
        top_k=settings.top_k,
        gemini_model=settings.gemini_model,
        query_rewrite_model=settings.query_rewrite_model,
        embedding_model=settings.embedding_model,
        index_path=str(settings.index_path),
        embeddings_path=str(settings.embeddings_path),
        frontend_origins=list(settings.frontend_origins),
    )


@router.post("/index/build", response_model=BuildIndexResponse)
async def rebuild_index(payload: BuildIndexRequest) -> BuildIndexResponse:
    settings = load_settings()
    chunk_count = await run_in_threadpool(
        build_index,
        project_root=settings.project_root,
        index_path=settings.index_path,
        embeddings_path=settings.embeddings_path,
        model_name=settings.embedding_model,
        requested_sources=payload.sources or None,
    )
    clear_runtime_cache()
    return BuildIndexResponse(
        message="Index built successfully.",
        chunk_count=chunk_count,
        index_path=str(settings.index_path),
        embeddings_path=str(settings.embeddings_path),
    )


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    settings = load_settings()
    if not index_exists(settings):
        raise HTTPException(
            status_code=409,
            detail="RAG index is missing. Build the index before sending chat requests.",
        )

    response = await run_in_threadpool(
        answer_question_cached,
        settings,
        payload.question,
        payload.top_k,
        _history_from_request(payload),
    )
    return ChatResponse(
        question=response.question,
        answer=response.answer,
        citations=citations_markdown(response.results),
        used_generation=response.used_generation,
        rewritten_question=response.rewritten_question,
        timestamp_utc=response.timestamp_utc,
        contexts=[_serialize_result(item) for item in response.results],
    )

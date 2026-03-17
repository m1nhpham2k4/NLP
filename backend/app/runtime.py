from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from .bootstrap import PROJECT_ROOT
from .settings import AppSettings
from nlp_rag.generate import (
    build_prompt,
    fallback_answer,
    rewrite_query_with_gemini,
    try_generate_with_gemini,
)
from nlp_rag.retrieve import RagRetriever
from nlp_rag.service import ChatTurn, RagResponse


def index_exists(settings: AppSettings) -> bool:
    return settings.index_path.exists() and settings.embeddings_path.exists()


def _format_history(chat_history: list[ChatTurn] | None) -> str:
    if not chat_history:
        return ""
    return "\n".join(f"{turn.role}: {turn.content}" for turn in chat_history[-8:])


def _mtime_ns(path: Path) -> int:
    return path.stat().st_mtime_ns


@lru_cache(maxsize=8)
def _get_retriever(
    index_path_str: str,
    embeddings_path_str: str,
    model_name: str,
    index_mtime_ns: int,
    embeddings_mtime_ns: int,
) -> RagRetriever:
    del index_mtime_ns, embeddings_mtime_ns
    return RagRetriever(
        Path(index_path_str),
        Path(embeddings_path_str),
        model_name=model_name,
    )


def clear_runtime_cache() -> None:
    _get_retriever.cache_clear()


def answer_question_cached(
    settings: AppSettings,
    question: str,
    top_k: int | None = None,
    chat_history: list[ChatTurn] | None = None,
) -> RagResponse:
    if not index_exists(settings):
        raise FileNotFoundError("RAG index files are missing. Build the index first.")

    cleaned_question = question.strip()
    history_text = _format_history(chat_history)
    rewritten_question = cleaned_question
    if settings.enable_query_rewrite:
        rewritten_question = rewrite_query_with_gemini(
            question=cleaned_question,
            chat_history=history_text,
            api_key=settings.google_api_key,
            model_name=settings.query_rewrite_model,
        ) or rewritten_question

    retriever = _get_retriever(
        str(settings.index_path),
        str(settings.embeddings_path),
        settings.embedding_model,
        _mtime_ns(settings.index_path),
        _mtime_ns(settings.embeddings_path),
    )
    results = retriever.search(rewritten_question, top_k=top_k or settings.top_k)
    prompt = build_prompt(cleaned_question, rewritten_question, results, chat_history=history_text)
    generated = try_generate_with_gemini(
        prompt,
        settings.google_api_key,
        settings.gemini_model,
    )
    answer = generated or fallback_answer(cleaned_question, results)

    return RagResponse(
        question=cleaned_question,
        answer=answer,
        results=results,
        used_generation=bool(generated),
        rewritten_question=rewritten_question,
        timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def load_settings() -> AppSettings:
    return AppSettings.from_env(PROJECT_ROOT)

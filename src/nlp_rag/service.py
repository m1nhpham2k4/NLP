from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

from .config import Settings
from .generate import (
    build_prompt,
    fallback_answer,
    rewrite_query_with_gemini,
    try_generate_with_gemini,
)
from .retrieve import RagRetriever, SearchResult


@dataclass(slots=True)
class RagResponse:
    question: str
    answer: str
    results: list[SearchResult]
    used_generation: bool
    rewritten_question: str
    timestamp_utc: str


@dataclass(slots=True)
class ChatTurn:
    role: str
    content: str


def index_exists(settings: Settings) -> bool:
    return settings.index_path.exists() and settings.embeddings_path.exists()


def _format_history(chat_history: list[ChatTurn] | None) -> str:
    if not chat_history:
        return ""
    return "\n".join(f"{turn.role}: {turn.content}" for turn in chat_history[-8:])


def _format_citation_line(index: int, item: SearchResult) -> str:
    title = item.metadata.get("title", "Untitled")
    section = item.metadata.get("section_path", "root")
    return f"[{index}] {title} | {section} | score={item.score:.3f}"


def citations_markdown(results: list[SearchResult]) -> str:
    if not results:
        return "Không có trích dẫn."
    return "\n".join(f"- {_format_citation_line(idx, item)}" for idx, item in enumerate(results, start=1))


def export_answer_markdown(response: RagResponse) -> str:
    lines = [
        "# IUH RAG Demo Export",
        "",
        f"- Timestamp (UTC): {response.timestamp_utc}",
        f"- Original question: {response.question}",
        f"- Normalized question: {response.rewritten_question}",
        f"- Generation mode: {'Gemini' if response.used_generation else 'Fallback retrieval'}",
        "",
        "## Answer",
        response.answer,
        "",
        "## Citations",
        citations_markdown(response.results),
        "",
        "## Retrieved Contexts",
    ]
    for idx, item in enumerate(response.results, start=1):
        lines.extend(
            [
                f"### Context {idx}",
                f"- Source: {item.source}",
                f"- Title: {item.metadata.get('title', 'Untitled')}",
                f"- Section: {item.metadata.get('section_path', 'root')}",
                f"- Blended score: {item.score:.3f}",
                f"- Semantic score: {item.metadata.get('semantic_score', 'n/a')}",
                f"- Lexical score: {item.metadata.get('lexical_score', 'n/a')}",
                "",
                item.text,
                "",
            ]
        )
    return "\n".join(lines)


def export_answer_json(response: RagResponse) -> str:
    payload = {
        "question": response.question,
        "rewritten_question": response.rewritten_question,
        "answer": response.answer,
        "used_generation": response.used_generation,
        "timestamp_utc": response.timestamp_utc,
        "results": [
            {
                "score": item.score,
                "source": item.source,
                "text": item.text,
                "metadata": item.metadata,
            }
            for item in response.results
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def export_chat_txt(chat_turns: list[ChatTurn]) -> str:
    if not chat_turns:
        return "No chat history."
    return "\n\n".join(f"{turn.role.upper()}:\n{turn.content}" for turn in chat_turns)


def answer_question(
    settings: Settings,
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

    retriever = RagRetriever(
        settings.index_path,
        settings.embeddings_path,
        model_name=settings.embedding_model,
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

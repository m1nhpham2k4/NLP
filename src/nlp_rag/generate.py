from __future__ import annotations

from typing import Iterable

from .retrieve import SearchResult


def build_prompt(question: str, normalized_question: str, contexts: Iterable[SearchResult], chat_history: str = "") -> str:
    context_block = "\n\n".join(
        (
            f"[{idx}] Source: {item.source}\n"
            f"[{idx}] Section: {item.metadata.get('section_path', 'root')}\n"
            f"[{idx}] Title: {item.metadata.get('title', 'Untitled')}\n"
            f"[{idx}] Content: {item.text}"
        )
        for idx, item in enumerate(contexts, start=1)
    )
    return (
        "You are an IUH academic assistant. Answer in Vietnamese, grounded only in the provided context.\n"
        "If the context is insufficient, say clearly that the indexed data does not contain enough information.\n"
        "Prefer concise, factual answers.\n"
        "When citing evidence, use bracket citations like [1], [2] that refer to the provided contexts.\n"
        "Do not invent policies, dates, or contact details beyond the contexts.\n\n"
        f"Conversation history:\n{chat_history or '(empty)'}\n\n"
        f"Original question:\n{question}\n\n"
        f"History-aware normalized question for retrieval:\n{normalized_question}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer:"
    )


def fallback_answer(question: str, contexts: list[SearchResult]) -> str:
    if not contexts:
        return "Không tìm thấy dữ liệu phù hợp trong chỉ mục hiện tại."

    lines = [
        f"Câu hỏi: {question}",
        "Các ngữ cảnh liên quan nhất:",
    ]
    for idx, item in enumerate(contexts, start=1):
        title = item.metadata.get("title", "Untitled")
        section = item.metadata.get("section_path", "root")
        lines.append(f"[{idx}] [{item.score:.3f}] {title} | {section}: {item.text[:320]}")
    return "\n".join(lines)


def rewrite_query_with_gemini(
    question: str,
    chat_history: str,
    api_key: str | None,
    model_name: str,
) -> str | None:
    if not api_key:
        return None

    try:
        from google import genai
    except ImportError:
        return None

    prompt = (
        "You rewrite Vietnamese questions for history-aware retrieval over IUH university documents.\n"
        "Tasks:\n"
        "1. Use the conversation history to resolve pronouns or omitted subjects.\n"
        "2. Fix spelling mistakes.\n"
        "3. Expand abbreviations when obvious, such as CNTT -> Công nghệ thông tin.\n"
        "4. Preserve Vietnamese language.\n"
        "5. Keep the meaning unchanged while making the query standalone and retrieval-friendly.\n"
        "6. Return only the rewritten query, no explanation.\n\n"
        f"Conversation history:\n{chat_history or '(empty)'}\n\n"
        f"Question:\n{question}\n\n"
        "Rewritten query:"
    )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    text = getattr(response, "text", None)
    if not text:
        return None
    return text.strip()


def try_generate_with_gemini(prompt: str, api_key: str | None, model_name: str) -> str | None:
    if not api_key:
        return None

    try:
        from google import genai
    except ImportError:
        return None

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    return getattr(response, "text", None)

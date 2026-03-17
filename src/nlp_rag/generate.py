from __future__ import annotations

from typing import Iterable

from .retrieve import SearchResult


def build_prompt(query: str, contexts: Iterable[SearchResult]) -> str:
    context_block = "\n\n".join(
        f"Source: {item.source}\nContent: {item.text}" for item in contexts
    )
    return (
        "You are answering questions using the provided university knowledge base.\n"
        "Answer in Vietnamese when the question is in Vietnamese.\n"
        "If the context is insufficient, say that the answer is not available in the indexed data.\n\n"
        f"Question:\n{query}\n\n"
        f"Context:\n{context_block}\n\n"
        "Answer:"
    )


def fallback_answer(query: str, contexts: list[SearchResult]) -> str:
    if not contexts:
        return "Khong tim thay du lieu phu hop trong chi muc."

    lines = [
        f"Cau hoi: {query}",
        "Thong tin lien quan nhat trong chi muc:",
    ]
    for idx, item in enumerate(contexts, start=1):
        lines.append(f"{idx}. [{item.score:.3f}] {item.text[:280]}")
    return "\n".join(lines)


def try_generate_with_gemini(prompt: str, api_key: str | None) -> str | None:
    if not api_key:
        return None

    try:
        from google import genai
    except ImportError:
        return None

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return getattr(response, "text", None)

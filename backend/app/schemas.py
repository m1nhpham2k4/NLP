from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ChatMessage(StrictModel):
    role: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class SearchResultResponse(StrictModel):
    score: float
    source: str
    text: str
    metadata: dict[str, str]


class StatusResponse(StrictModel):
    app_name: str
    index_ready: bool
    gemini_configured: bool
    query_rewrite_enabled: bool
    top_k: int
    gemini_model: str
    query_rewrite_model: str
    embedding_model: str
    index_path: str
    embeddings_path: str
    frontend_origins: list[str]


class BuildIndexRequest(StrictModel):
    sources: list[str] = Field(default_factory=list)


class BuildIndexResponse(StrictModel):
    message: str
    chunk_count: int
    index_path: str
    embeddings_path: str


class ChatRequest(StrictModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=20)
    history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(StrictModel):
    question: str
    answer: str
    citations: str
    used_generation: bool
    rewritten_question: str
    timestamp_utc: str
    contexts: list[SearchResultResponse]


class MessageResponse(StrictModel):
    message: str

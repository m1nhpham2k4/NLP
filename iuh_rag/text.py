import re
import unicodedata
from typing import Iterable, List, Sequence, Set

from .config import VIETNAMESE_STOPWORDS


WHITESPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def remove_accents(text: str) -> str:
    text = str(text).replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


def clean_text(text: str) -> str:
    text = str(text).replace("\ufeff", " ").replace("\u200b", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [WHITESPACE_RE.sub(" ", line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines).strip()


def normalize_text(text: str, *, keep_digits: bool = True) -> str:
    text = remove_accents(str(text).lower())
    if keep_digits:
        text = re.sub(r"[^a-z0-9\s]", " ", text)
    else:
        text = re.sub(r"[^a-z\s]", " ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def slugify(text: str, max_len: int = 90) -> str:
    text = normalize_text(text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:max_len].strip("_") or "document"


def tokenize(text: str, *, stopwords: Set[str] | None = None) -> List[str]:
    stopwords = VIETNAMESE_STOPWORDS if stopwords is None else stopwords
    tokens = normalize_text(text).split()
    return [token for token in tokens if len(token) > 1 and token not in stopwords]


def sentence_split(text: str) -> List[str]:
    parts = SENTENCE_SPLIT_RE.split(clean_text(text))
    return [part.strip() for part in parts if part.strip()]


def chunk_text(text: str, max_chars: int = 1400, overlap_chars: int = 220) -> List[str]:
    """Sentence-aware chunking with a small character overlap."""
    text = clean_text(text)
    if not text:
        return []
    sentences = sentence_split(text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence_len = len(sentence)
        if current and current_len + sentence_len + 1 > max_chars:
            chunks.append(" ".join(current).strip())
            overlap = chunks[-1][-overlap_chars:] if overlap_chars > 0 else ""
            current = [overlap, sentence] if overlap else [sentence]
            current_len = sum(len(item) + 1 for item in current)
        elif sentence_len > max_chars:
            if current:
                chunks.append(" ".join(current).strip())
                current = []
                current_len = 0
            for start in range(0, sentence_len, max_chars - overlap_chars):
                part = sentence[start : start + max_chars].strip()
                if part:
                    chunks.append(part)
        else:
            current.append(sentence)
            current_len += sentence_len + 1

    if current:
        chunks.append(" ".join(current).strip())

    return [chunk for chunk in chunks if chunk]


def token_overlap_score(query: str, text: str) -> float:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return 0.0
    text_tokens = set(tokenize(text))
    if not text_tokens:
        return 0.0
    overlap = query_tokens & text_tokens
    return len(overlap) / max(1, len(query_tokens))


def keyword_hit_ratio(text: str, keywords: Sequence[str]) -> float:
    if not keywords:
        return 0.0
    text_norm = normalize_text(text)
    hits = 0
    for keyword in keywords:
        keyword_norm = normalize_text(keyword)
        if keyword_norm and keyword_norm in text_norm:
            hits += 1
    return hits / len(keywords)


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen = set()
    output = []
    for item in items:
        if item not in seen:
            output.append(item)
            seen.add(item)
    return output

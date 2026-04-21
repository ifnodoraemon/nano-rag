from __future__ import annotations

import math
import re
from collections import Counter
from hashlib import sha256


def mock_embeddings(texts: list[str], dimensions: int = 32) -> dict:
    return {
        "data": [
            {"index": index, "embedding": _text_to_vector(text, dimensions)}
            for index, text in enumerate(texts)
        ]
    }


def mock_rerank(query: str, documents: list[str], top_n: int) -> dict:
    query_terms = _tokenize(query)
    scored = []
    for index, document in enumerate(documents):
        doc_terms = _tokenize(document)
        overlap = len(query_terms & doc_terms)
        score = overlap + _keyword_bias(query, document)
        scored.append({"index": index, "relevance_score": float(score)})
    scored.sort(key=lambda item: item["relevance_score"], reverse=True)
    return {"results": scored[:top_n]}


def mock_chat(messages: list[dict[str, str]]) -> dict:
    user_message = next((message["content"] for message in reversed(messages) if message["role"] == "user"), "")
    contexts = _extract_contexts(user_message)
    question_match = re.search(r"问题：(.+?)\n", user_message, re.S)
    question = question_match.group(1).strip() if question_match else ""

    selected_chunk_id = None
    selected_text = None
    best_score = -1.0
    for chunk_id, text in contexts:
        candidate = _best_snippet(question, text.strip())
        score = _keyword_bias(question, candidate)
        if score > best_score:
            best_score = score
            selected_chunk_id = chunk_id
            selected_text = candidate.strip()

    if not selected_text:
        answer = "无法确认。当前没有可用上下文。"
    elif best_score <= 0:
        answer = f"无法确认。现有证据不足。[{selected_chunk_id}]"
    else:
        answer = f"{selected_text} [{selected_chunk_id}]"

    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": answer,
                },
                "finish_reason": "stop",
            }
        ],
        "model": "mock",
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _text_to_vector(text: str, dimensions: int) -> list[float]:
    counter = Counter(_tokenize_ordered(text))
    vector = [0.0] * dimensions
    for token, count in counter.items():
        digest = sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:2], "big") % dimensions
        vector[bucket] += float(count)
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [round(value / norm, 6) for value in vector]


def _tokenize(text: str) -> set[str]:
    return set(_tokenize_ordered(text))


def _tokenize_ordered(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.split(r"[^0-9A-Za-z\u4e00-\u9fff]+", text.lower()):
        if not raw:
            continue
        if re.fullmatch(r"[0-9a-z]+", raw):
            tokens.append(raw)
            continue
        tokens.extend([char for char in raw if _is_cjk(char)])
    return tokens


def _keyword_bias(query: str, text: str) -> float:
    terms = _tokenize(query)
    text_terms = _tokenize(text)
    score = 0.0
    for term in terms:
        if term and term in text_terms:
            score += 1.0
    return score


def _best_snippet(query: str, text: str) -> str:
    sentences = [segment.strip() for segment in re.split(r"[。！？\n]+", text) if segment.strip()]
    if not sentences:
        return text
    return max(sentences, key=lambda sentence: _keyword_bias(query, sentence))


def _is_cjk(char: str) -> bool:
    return "\u4e00" <= char <= "\u9fff"


def _extract_contexts(user_message: str) -> list[tuple[str, str]]:
    marker = "可用上下文："
    if marker not in user_message:
        return []
    block = user_message.split(marker, 1)[1].strip()
    matches = list(re.finditer(r"\[([^\]]+)\]\s*", block))
    contexts: list[tuple[str, str]] = []
    for index, match in enumerate(matches):
        chunk_id = match.group(1)
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(block)
        text = block[start:end].strip()
        if text:
            contexts.append((chunk_id, text))
    return contexts

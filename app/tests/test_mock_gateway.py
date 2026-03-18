from app.model_client.mock_gateway import mock_chat, mock_embeddings, mock_rerank


def test_mock_embeddings_returns_vectors() -> None:
    payload = mock_embeddings(["hello world"])
    assert len(payload["data"]) == 1
    assert len(payload["data"][0]["embedding"]) == 32


def test_mock_rerank_prioritizes_overlap() -> None:
    payload = mock_rerank("报销多久提交", ["请假规则", "报销需要在15天内提交"], 1)
    assert payload["results"][0]["index"] == 1


def test_mock_chat_uses_best_context() -> None:
    payload = mock_chat(
        [
            {
                "role": "user",
                "content": "问题：差旅报销多久内提交？\n\n可用上下文：\n[c1] 员工应在出差结束后 15 个自然日内提交差旅报销申请。",
            }
        ]
    )
    assert "15 个自然日内" in payload["choices"][0]["message"]["content"]

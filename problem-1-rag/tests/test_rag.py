import pytest
from unittest.mock import MagicMock
from src.rag import RAGService, INSUFFICIENT_CONTEXT_MESSAGE


def test_rag_fallback_insufficient_context():
    mock_store = MagicMock()
    mock_store.search.return_value = []  # No points match

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 384

    service = RAGService(store=mock_store, embedder=mock_embedder)
    res = service.query(question="What is unknown?")

    assert res["answer"] == INSUFFICIENT_CONTEXT_MESSAGE
    assert res["citations"] == []
    assert res["chunk_count"] == 0
    assert res["is_grounded"] is False
    assert res["token_usage"]["total_tokens"] == 0


def test_rag_query_with_context_and_citations():
    mock_store = MagicMock()
    mock_store.search.return_value = [
        {
            "id": "11111111-1111-1111-1111-111111111111",
            "score": 0.88,
            "payload": {
                "text": "Vector databases index high-dimensional embeddings efficiently.",
                "document_id": "guide.md",
                "chunk_index": 0
            }
        }
    ]

    mock_embedder = MagicMock()
    mock_embedder.embed_query.return_value = [0.1] * 384

    service = RAGService(store=mock_store, embedder=mock_embedder)

    # Mock Groq client
    mock_groq = MagicMock()
    mock_completion = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Vector databases index embeddings efficiently [Chunk 1]."
    mock_completion.choices = [mock_choice]
    mock_completion.usage.prompt_tokens = 50
    mock_completion.usage.completion_tokens = 15
    mock_completion.usage.total_tokens = 65
    mock_groq.chat.completions.create.return_value = mock_completion

    service._groq_client = mock_groq

    res = service.query(question="What do vector databases do?")

    assert "[Chunk 1]" in res["answer"]
    assert res["chunk_count"] == 1
    assert len(res["citations"]) == 1
    assert res["citations"][0]["citation"] == "[Chunk 1]"
    assert res["citations"][0]["document_id"] == "guide.md"
    assert res["token_usage"]["total_tokens"] == 65
